from typing import List, TYPE_CHECKING, TypedDict, Optional
import asyncio
import time
import os


from src.agents.base_agent import BaseAgent
from src.agents.agenda_manager.prompts import get_prompt
from src.agents.agenda_manager.tools import UpdateSessionNote, UpdateSubtopicNotes, UpdateSubtopicCoverage, FeedbackSubtopicCoverage, \
    UpdateMemoryBankAndSession
from src.agents.shared.memory_tools import Recall
from src.utils.data_process import read_from_pdf
from src.utils.llm.prompt_utils import format_prompt
from src.utils.logger.session_logger import SessionLogger
from src.interview_session.session_models import Participant, Message
from src.content.memory_bank.memory import Memory

if TYPE_CHECKING:
    from src.interview_session.interview_session import InterviewSession



class AgendaManagerConfig(TypedDict, total=False):
    """Configuration for the AgendaManager agent."""
    user_id: str


class AgendaManager(BaseAgent, Participant):
    def __init__(self, config: AgendaManagerConfig, interview_session: 'InterviewSession'):
        BaseAgent.__init__(
            self, name="AgendaManager",
            description="Agent that takes notes and manages the user's memory bank",
            config=config
        )
        Participant.__init__(self, title="AgendaManager",
                             interview_session=interview_session)
        
        # Current unprocessed memories
        self._new_memories: List[Memory] = []
        # All memories from this session
        self._all_session_memories: List[Memory] = []
        # Mapping from temporary memory IDs to real IDs
        self._memory_id_map = {}

        # Track last interviewer message
        self._last_interviewer_message = None

        # Locks and processing flags
        self.processing_in_progress = False # If processing is in progress
        self._pending_tasks = 0             # Track number of pending tasks
        self._notes_lock = asyncio.Lock()   # Lock for memory writes
        self._session_agenda_lock = asyncio.Lock()  # Lock for session agenda
        self._tasks_lock = asyncio.Lock()   # Lock for updating task counter

        # Tools agent can use
        self.tools = {
            "update_memory_bank_and_session": UpdateMemoryBankAndSession(
                memory_bank=self.interview_session.memory_bank,
                on_memory_added=self._add_new_memory,
                update_memory_map=self._update_memory_map,
                get_current_qa=self._get_recent_qa,
                session_agenda=self.interview_session.session_agenda
            ),
            "update_session_agenda": UpdateSessionNote(
                session_agenda=self.interview_session.session_agenda
            ),
            "update_subtopic_coverage": UpdateSubtopicCoverage(
                session_agenda=self.interview_session.session_agenda
            ),
            "feedback_subtopic_coverage": FeedbackSubtopicCoverage(
                session_agenda=self.interview_session.session_agenda
            ),
            "update_subtopic_notes": UpdateSubtopicNotes(
                session_agenda=self.interview_session.session_agenda
            ),
            "recall": Recall(
                memory_bank=self.interview_session.memory_bank
            ),
        }

    async def on_message(self, message: Message):
        '''Handle incoming messages'''
        SessionLogger.log_to_file(
            "execution_log",
            f"[NOTIFY] Agenda manager received message from {message.role}"
        )

        if message.role == "Interviewer":
            self._last_interviewer_message = message
            # Add question to session agenda
            self._add_question_to_session_agenda()
        elif message.role == "User":
            if self._last_interviewer_message:
                asyncio.create_task(self._process_qa_pair(
                    interviewer_message=self._last_interviewer_message,
                    user_message=message
                ))
                self._last_interviewer_message = None
     
    async def augment_session_agenda(self, additional_context_path: Optional[str] = None):
        # If there is existing user profile, we load them
        if additional_context_path and os.path.exists(additional_context_path):
            if additional_context_path.endswith('.txt') or additional_context_path.endswith('.md'):
                with open(additional_context_path, 'r', encoding='utf-8') as f:
                    additional_context = f.read()
            elif additional_context_path.endswith('.pdf'):
                additional_context = read_from_pdf(additional_context_path)
            else:
                SessionLogger.log_to_file(
                    "execution_log", f"[INIT] Existing user profile is IGNORED, currently only supports .txt, .md, and .pdf files"
                )
            
            # Found initial context to be initialized with
            SessionLogger.log_to_file(
                "execution_log", f"[RUN] Found initial context to be initialized with, preparing an optimized session!"
            )

            # Get user portrait and last meeting summary
            await asyncio.gather(
                self.interview_session.agenda_manager._update_user_portrait(
                    additional_context=additional_context
                ),
                self.interview_session.agenda_manager._update_last_meeting_summary(
                    additional_context=additional_context
                ),
                self.interview_session.agenda_manager._update_subtopic_notes(
                    additional_context=additional_context
                )
            )

            # TODO update memory?
            # Update session agenda notes and eventually coverage
            await self._update_list_of_subtopics(additional_context=additional_context)
            await self._update_subtopic_coverage()

    def _add_question_to_session_agenda(self):
        if self._last_interviewer_message:
            subtopic_id = str(self._last_interviewer_message.metadata.get('subtopic_id', ""))
            question_text = self._last_interviewer_message.content.strip()

            # Add question to QuestionBank if exists
            adding_status = False
            if self.interview_session.proposed_question_bank:
                question = self.interview_session.proposed_question_bank.add_question(content=question_text,
                                                         subtopic_id=subtopic_id)

                # Add question to SessionAgenda
                adding_status = self.interview_session.session_agenda.add_interview_question(question=question)
            else:
                # Add question to SessionAgenda
                adding_status = self.interview_session.session_agenda.add_interview_question_raw(
                    subtopic_id=subtopic_id,
                    question=question_text,
                )
                
            if not adding_status:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[NOTIFY] SessionAgenda failed/skipped to add question to session agenda.",
                )

    async def _process_qa_pair(self, interviewer_message: Message, user_message: Message):
        """Process a Q&A pair with task tracking"""
        await self._increment_pending_tasks()
        try:
            await self._locked_write_memory_notes_and_question_bank(interviewer_message, user_message)
            await self._locked_update_subtopic_coverage(interviewer_message, user_message)
        finally:
            await self._decrement_pending_tasks()

    async def _locked_write_memory_notes_and_question_bank(self, interviewer_message: Message, user_message: Message) -> None:
        """Wrapper to handle update_memory_bank_and_session with lock"""
        async with self._notes_lock:
            self.add_event(sender=interviewer_message.role,
                        tag="memory_lock_message", 
                        content=interviewer_message.content)
            self.add_event(sender=user_message.role,
                        tag="memory_lock_message", 
                        content=user_message.content)
            await self._write_memory_notes_and_question_bank()
            
    async def _locked_update_subtopic_coverage(self, interviewer_message: Message, user_message: Message) -> None:
        """Wrapper to handle update_subtopic_coverage with lock"""
        async with self._session_agenda_lock:
            self.add_event(sender=interviewer_message.role,
                        tag="agenda_lock_message", 
                        content=interviewer_message.content)
            self.add_event(sender=user_message.role,
                        tag="agenda_lock_message", 
                        content=user_message.content)
            await self._update_subtopic_coverage()

    async def _update_last_meeting_summary(self, additional_context: str):
        prompt = self._get_formatted_prompt("update_last_meeting_summary",
                                            additional_context=additional_context)
        self.add_event(
            sender=self.name, 
            tag="update_last_meeting_summary_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_last_meeting_summary_response", 
            content=response
        )
        
        async with self._session_agenda_lock:
            self.interview_session.session_agenda.update_last_meeting_summary_str(response)
    
    async def _update_user_portrait(self, additional_context: str):
        prompt = self._get_formatted_prompt("update_user_portrait",
                                            additional_context=additional_context)
        self.add_event(
            sender=self.name,
            tag="update_user_portrait_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_user_portrait_response", 
            content=response
        )
        
        async with self._session_agenda_lock:
            self.interview_session.session_agenda.update_user_portrait_str(response)
            
    async def _update_subtopic_notes(self, additional_context: str) -> None:
        """Process the latest conversation and update both memory and question banks."""
        prompt = self._get_formatted_prompt("update_subtopic_notes", additional_context=additional_context)
        self.add_event(
            sender=self.name, 
            tag="update_subtopic_notes_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_subtopic_notes_response", 
            content=response
        )
        
        async with self._session_agenda_lock:
            self.handle_tool_calls(response)

    async def _write_memory_notes_and_question_bank(self) -> None:
        """Process the latest conversation and update both memory and question banks."""
        prompt = self._get_formatted_prompt("update_memory_and_session")
        self.add_event(
            sender=self.name, 
            tag="update_memory_question_bank_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_memory_question_bank_response", 
            content=response
        )
        self.handle_tool_calls(response)
        
    async def _update_subtopic_coverage(self, active_topics_only: bool = False) -> None:
        """Process the latest conversation and update subtopic coverage and possibly move to the next topic."""
        prompt = self._get_formatted_prompt("update_subtopic_coverage", active_topics_only=active_topics_only)
        self.add_event(
            sender=self.name, 
            tag="update_subtopic_coverage_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_subtopic_coverage_response", 
            content=response
        )
        self.handle_tool_calls(response)
        
        # Decide if need to proceed
        self.interview_session.session_agenda.revise_agenda_after_update()
        
    async def _update_list_of_subtopics(self, active_topics_only: bool = False, additional_context: Optional[str] = None) -> None:
        """Process the latest conversation and update list of subtopics."""
        prompt = self._get_formatted_prompt("update_list_of_subtopics", active_topics_only=active_topics_only,
                                            additional_context=additional_context)
        self.add_event(
            sender=self.name, 
            tag="update_list_of_subtopics_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_list_of_subtopics_response", 
            content=response
        )
        self.handle_tool_calls(response)

    async def _update_session_agenda(self) -> None:
        """Update session agenda with user's response"""
        prompt = self._get_formatted_prompt("update_session_agenda")
        self.add_event(
            sender=self.name,
            tag="update_session_agenda_prompt",
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name,
            tag="update_session_agenda_response",
            content=response
        )
        self.handle_tool_calls(response)

    def _get_formatted_prompt(self, prompt_type: str, **kwargs) -> str:
        '''Gets the formatted prompt for the AgendaManager agent.'''
        prompt = get_prompt(prompt_type)
        if prompt_type == "update_memory_and_session":
            events = self.get_event_stream_str(filter=[
                {"tag": "memory_lock_message"},
            ], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []
            previous_events = events[:-2] if len(events) >= 2 else events

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "previous_events": "\n".join(previous_events),
                "current_qa": "\n".join(current_qa),
                "topics_list": self.interview_session.session_agenda.get_all_topics_and_subtopics(),
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_memory_bank_and_session"]
                )
            })
        elif prompt_type == "update_list_of_subtopics":
            events = self.get_event_stream_str(filter=[
                {"tag": "agenda_lock_message"},
            ], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []
            previous_events = events[:-2] if len(events) >= 2 else events

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            active_topics_only = kwargs.get("active_topics_only", True)
            topics_list = self.interview_session.session_agenda.get_questions_and_notes_str(hide_answered="all",
                                                                                            active_topics_only=active_topics_only)

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "interview_description": self.interview_session.session_agenda.interview_description,
                "previous_events": "\n".join(previous_events),
                "additional_context": kwargs.get("additional_context", None),
                "current_qa": "\n".join(current_qa),
                "last_meeting_summary": self.interview_session.session_agenda.get_last_meeting_summary_str(),
                "topics_list": topics_list,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_subtopic_coverage"]
                )
            })
        elif prompt_type == "update_session_agenda":
            events = self.get_event_stream_str(
                filter=[{"tag": "notes_lock_message"}], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []
            previous_events = events[:-2] if len(events) >= 2 else events

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "previous_events": "\n".join(previous_events),
                "current_qa": "\n".join(current_qa),
                "questions_and_notes": (
                    self.interview_session.session_agenda \
                        .get_questions_and_notes_str(
                            hide_answered="qa"
                        )
                ),
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_session_agenda"]
                )
            })
        elif prompt_type == "update_subtopic_coverage":
            events = self.get_event_stream_str(
                filter=[{"tag": "agenda_lock_message"}], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []
            previous_events = events[:-2] if len(events) >= 2 else events

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            active_topics_only = kwargs.get("active_topics_only", True)
            topics_list = self.interview_session.session_agenda.get_questions_and_notes_str(hide_answered="all",
                                                                                            active_topics_only=active_topics_only)

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "previous_events": "\n".join(previous_events),
                "current_qa": "\n".join(current_qa),
                "last_meeting_summary": self.interview_session.session_agenda.get_last_meeting_summary_str(),
                "topics_list": topics_list,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_subtopic_coverage"]
                )
            })
        elif prompt_type == "update_subtopic_notes":
            return format_prompt(prompt, {
                "additional_context": kwargs.get("additional_context"),
                "topics_list": (
                    self.interview_session.session_agenda.get_all_topics_and_subtopics(active_topics_only=False)
                ),
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_subtopic_notes"] # Note disable feedback as tools
                )
            })
        elif prompt_type == "update_last_meeting_summary":
            return format_prompt(prompt, {
                "additional_context": kwargs.get("additional_context"),
                "interview_description": self.interview_session.session_agenda.interview_description,
            })
        elif prompt_type == "update_user_portrait":
            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "additional_context": kwargs.get("additional_context"),
                "interview_description": self.interview_session.session_agenda.interview_description,
            })

    async def get_session_memories(self, clear_processed=False, wait_for_processing=True, include_processed=False) -> List[Memory]:
        """Get memories added by agenda manager during current session.
        
        Args:
            clear_processed: 
                - If True, clears the list of unprocessed memories after returning
            wait_for_processing: 
                - If True, waits for all pending memory updates to complete
            include_processed: 
                - If True, returns all memories from the session
                - If False, returns only the currently unprocessed memories
        
        Returns:
            List of Memory objects based on the include_processed parameter
        """
        if wait_for_processing:
            start_time = time.time()

            SessionLogger.log_to_file(
                "execution_log",
                f"[MEMORY] Waiting for memory updates to complete..."
            )
            
            while self.processing_in_progress:
                await asyncio.sleep(0.1)
                if time.time() - start_time > 300:  # 5 minutes timeout
                    SessionLogger.log_to_file(
                        "execution_log",
                        f"[MEMORY] Timeout waiting for memory updates"
                    )
                    break
        elif self.processing_in_progress:
            SessionLogger.log_to_file(
                "execution_log",
                f"[MEMORY] Retrieving memories..."
            )

        if include_processed:
            memories = self._all_session_memories.copy()
            memory_source = "all session"
        else:
            memories = self._new_memories.copy()
            memory_source = "unprocessed"
        
        if clear_processed:
            SessionLogger.log_to_file(
                "execution_log",
                f"[MEMORY] Clearing {len(self._new_memories)} unprocessed memories"
            )
            self._new_memories = []
            
        SessionLogger.log_to_file(
            "execution_log",
            (
                f"[MEMORY] Collected {len(memories)} {memory_source} memories "
                f"from current session"
            )
        )
        return memories

    def _add_new_memory(self, memory: Memory):
        """Callback to track newly added memory in the session"""
        self._new_memories.append(memory)
        self._all_session_memories.append(memory)  # Also add to all memories list

    def _update_memory_map(self, temp_id: str, real_id: str) -> None:
        """Callback to update the memory ID mapping"""
        self._memory_id_map[temp_id] = real_id
        SessionLogger.log_to_file("execution_log",
                                  f"[MEMORY] Write a new memory with {real_id}")

    async def _increment_pending_tasks(self):
        """Increment the pending tasks counter"""
        async with self._tasks_lock:
            self._pending_tasks += 1
            self.processing_in_progress = True

    async def _decrement_pending_tasks(self):
        """Decrement the pending tasks counter"""
        async with self._tasks_lock:
            self._pending_tasks -= 1
            if self._pending_tasks <= 0:
                self._pending_tasks = 0
                self.processing_in_progress = False

    def _get_recent_qa(self) -> str:
        """Safely get the current user response, with error handling."""
        interviewer_question = "No interviewer question available"
        user_response = "No user response available"
        
        try:
            messages = self.get_event_stream_str(filter=[
                {"tag": "memory_lock_message", "sender": "Interviewer"}
            ], as_list=True)
                        
            if messages:
                last_message = messages[-1]
                interviewer_question = last_message. \
                    removeprefix("<Interviewer>\n"). \
                    removesuffix("\n</Interviewer>")
        except Exception as e:
            return "Error retrieving interviewer's question"
        
        try:
            messages = self.get_event_stream_str(filter=[
                {"tag": "memory_lock_message", "sender": "User"}
            ], as_list=True)
                        
            if messages:
                last_message = messages[-1]
                user_response = last_message. \
                    removeprefix("<User>\n"). \
                    removesuffix("\n</User>")
        except Exception as e:
            return "Error retrieving user response"
        
        return interviewer_question, user_response
