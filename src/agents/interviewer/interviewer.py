import os
import re
from typing import TYPE_CHECKING, TypedDict, Tuple

import json

from src.agents.base_agent import BaseAgent
from src.agents.interviewer.prompts import get_prompt
from src.agents.interviewer.tools import EndConversation, RespondToUser
from src.agents.shared.memory_tools import Recall
from src.utils.llm.prompt_utils import format_prompt
from src.interview_session.session_models import Participant, Message
from src.utils.llm.xml_formatter import parse_rubric_call
from src.utils.logger.session_logger import SessionLogger
from src.utils.constants.colors import GREEN, RESET
from src.content.question_bank.question import Rubric

if TYPE_CHECKING:
    from src.interview_session.interview_session import InterviewSession



class TTSConfig(TypedDict, total=False):
    """Configuration for text-to-speech."""
    enabled: bool
    provider: str  # e.g. 'openai'
    voice: str     # e.g. 'alloy'


class InterviewerConfig(TypedDict, total=False):
    """Configuration for the Interviewer agent."""
    user_id: str
    tts: TTSConfig
    interview_description: str


class Interviewer(BaseAgent, Participant):
    '''Inherits from BaseAgent and Participant. Participant is a class that all agents in the interview session inherit from.'''

    def __init__(self, config: InterviewerConfig, interview_session: 'InterviewSession'):
        BaseAgent.__init__(
            self, name="Interviewer",
            description="The agent that holds the interview and asks questions.",
            config=config)
        Participant.__init__(
            self, title="Interviewer",
            interview_session=interview_session)

        self.interview_description = config.get("interview_description")
        self.tools = {
            "recall": Recall(memory_bank=self.interview_session.memory_bank),
            "respond_to_user": RespondToUser(
                tts_config=config.get("tts", {}),
                base_path= \
                    f"{os.getenv('DATA_DIR', 'data')}/{config.get('user_id')}/",
                on_response=self._handle_response,
                on_turn_complete=lambda: setattr(
                    self, '_turn_to_respond', False)
            ),
            # "end_conversation": EndConversation(
            #     on_goodbye=lambda goodbye: (
            #         self.add_event(sender=self.name,
            #                        tag="goodbye", content=goodbye),
            #         self.interview_session.add_message_to_chat_history(
            #             role=self.title, content=goodbye)
            #     ),
            #     on_end=lambda: (
            #         setattr(self, '_turn_to_respond', False),
            #         self.interview_session.end_session()
            #     )
            # )
        }

        self._turn_to_respond = False

    def _handle_quantify_response(self, quantified_response: str,
                                  original_response: str) -> Tuple[str, Rubric]:
        # 2. Parse the <tool_calls> block from the response
        final_question_text = original_response
        final_rubric = None
        try:
            parsed_calls = parse_rubric_call(quantified_response)
            for parsed_call in parsed_calls:
                is_quantifiable = str(parsed_call.get('quantifiable', 'false')).lower() == 'true'

                if is_quantifiable:
                    final_question_text = parsed_call.get('question', original_response)
                    rubric_data_str = parsed_call.get('rubric')
                    if rubric_data_str and isinstance(rubric_data_str, str):
                        # The rubric might be a string representation of JSON
                        try:
                            rubric_data = json.loads(rubric_data_str)
                            final_rubric = Rubric(**rubric_data) # Need to be in string
                        except json.JSONDecodeError:
                            SessionLogger.log_to_file(
                                "execution_log",
                                f"Could not parse rubric JSON string: {rubric_data_str}",
                                log_level="warning"
                            )
                    elif isinstance(rubric_data_str, dict): # Sometimes it's already a dict
                        final_rubric = Rubric(**rubric_data_str) # Need to be in string

        except Exception as e:
            # If parsing fails for any reason, log it but fall back gracefully
            SessionLogger.log_to_file(
                "execution_log",
                f"Could not parse rubric response: {e}. Using original question.",
                log_level="warning"
            )
            final_question_text = original_response
            final_rubric = None

        return final_question_text, final_rubric

    async def _handle_response(self, response: str, subtopic_id: str = "") -> str:
        """Handle responses from the RespondToUser tool by quantifying it and adding them to chat history.
        
        Args:
            response: The response text to add to chat history
            topic_id: The topic ID of the response
            subtopic_id: The subtopic ID of the response
        """
        # # Quantify question even further
        # quantify_prompt = format_prompt(get_prompt("quantify_question"), {"question_text": response})
        # self.add_event(sender=self.name, tag="llm_prompt", content=quantify_prompt)
        # quantified_response = await self.call_engine_async(quantify_prompt)
        # print(f"{GREEN}Interviewer Quantified:\n{quantified_response}{RESET}")
        # quantified_question, rubric = self._handle_quantify_response(quantified_response=quantified_response,
        #                                                              original_response=response)
        
        # If we disable quantification
        quantified_question = response
        rubric = None
        
        self.interview_session.add_message_to_chat_history(
            role=self.title,
            content=quantified_question,
            metadata={'subtopic_id': str(subtopic_id), "rubric": rubric},
        )
        self.add_event(sender=self.name, tag="message",
                       content=quantified_question)
        
        return quantified_question

    async def on_message(self, message: Message):

        if message:
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] Interviewer received message from {message.role}"
            )
            self.add_event(sender=message.role, tag="message",
                           content=message.content)
        
        self._turn_to_respond = True
        iterations = 0

        while self._turn_to_respond and iterations < self._max_consideration_iterations:
            prompt = self._get_prompt()
            self.add_event(sender=self.name, tag="llm_prompt", content=prompt)
            response = await self.call_engine_async(prompt)
            print(f"{GREEN}Interviewer:\n{response}{RESET}")
            try:
                await self.handle_tool_calls_async(response)
            except Exception as e:
                print(f"Error calling tool: {e}. Use the raw response as the output.")
                await self._handle_response(response)

            iterations += 1
            if iterations >= self._max_consideration_iterations:
                self.add_event(
                    sender="system",
                    tag="error",
                    content=f"Exceeded maximum number of consideration "
                    f"iterations ({self._max_consideration_iterations})"
                )

    def _get_prompt(self):
        '''Gets the prompt for the interviewer. '''
        # Get user portrait and last meeting summary from session agenda
        user_portrait_str = self.interview_session.session_agenda \
            .get_user_portrait_str()
        last_meeting_summary_str = (
            self.interview_session.session_agenda
            .get_last_meeting_summary_str()
        )

        # Get chat history from event stream where these are the senders
        chat_history_events = self.get_event_stream_str(
            [
                {"sender": "Interviewer", "tag": "message"},
                {"sender": "User", "tag": "message"},
                {"sender": "system", "tag": "recall"},
            ],
            as_list=True
        )

        recent_events = chat_history_events[-self._max_events_len:] if \
            len(chat_history_events) > self._max_events_len else chat_history_events
        current_events = recent_events[-2:] if len(recent_events) >= 2 else recent_events

        all_interviewer_messages = self.get_event_stream_str(
            [{"sender": "Interviewer", "tag": "message"}],
            as_list=True
        )
        recent_interviewer_messages = all_interviewer_messages[-5:] if \
            len(all_interviewer_messages) >= 5 else all_interviewer_messages

        # Start with all available tools
        tools_set = set(self.tools.keys())
        
        # if self.interview_session.api_participant:
        #     # Don't end_conversation directly if API participant is present
        #     tools_set.discard("end_conversation")
        
        if self.use_baseline:
            # For baseline mode, remove recall tool
            tools_set.discard("recall")

        # Create format parameters based on prompt type
        format_params = {
            "user_portrait": user_portrait_str,
            "interview_description": self.interview_description,
            "last_meeting_summary": last_meeting_summary_str,
            "chat_history": '\n'.join(recent_events),
            "current_events": '\n'.join(current_events),
            "recent_interviewer_messages": '\n'.join(
                [msg for msg in recent_interviewer_messages]),
            "tool_descriptions": self.get_tools_description(list(tools_set))
        }

        # Only add questions_and_notes for normal mode
        if not self.use_baseline:
            questions_and_notes_str = self.interview_session.session_agenda \
                .get_questions_and_notes_str(
                    hide_answered="all", active_topics_only=True
                )
            format_params["questions_and_notes"] = questions_and_notes_str

            # Get strategic question suggestions from ExplorationPlanner (only if not stale)
            # Staleness is checked before formatting to avoid unnecessary work
            if self._should_include_strategic_questions():
                strategic_questions_str = self._format_strategic_questions()
                format_params["strategic_questions"] = strategic_questions_str

        # Use the baseline prompt if enabled
        if len(all_interviewer_messages) == 0 and len(last_meeting_summary_str) == 0:
            main_prompt = get_prompt("introduction")
        elif len(all_interviewer_messages) == 0:
            main_prompt = get_prompt("introduction_continue_session")
        elif self.use_baseline:
            main_prompt = get_prompt("baseline")
        else:
            main_prompt = get_prompt("normal")

            # Remove STRATEGIC_QUESTIONS section from template if stale
            if not self.use_baseline and not self._should_include_strategic_questions():
                # Remove the {STRATEGIC_QUESTIONS} line to exclude the section entirely
                main_prompt = main_prompt.replace("\n{STRATEGIC_QUESTIONS}\n", "\n")
                # Don't provide strategic_questions key in format_params (already omitted above)

        return format_prompt(main_prompt, format_params)

    def _format_strategic_questions(self) -> str:
        """
        Format strategic question suggestions from ExplorationPlanner.

        Returns formatted string with strategic questions or empty state message.
        Handles case where suggestions may be stale (from 3-5 turns ago).
        """
        # Access strategic state from ExplorationPlanner
        strategic_state = self.interview_session.exploration_planner.strategic_state
        suggestions = strategic_state.strategic_question_suggestions

        if not suggestions:
            return "No strategic question suggestions available yet. Use coverage-based heuristics to select questions from the topics list."

        # Get top rollout if available
        top_rollout = None
        if strategic_state.rollout_predictions:
            top_rollout = strategic_state.rollout_predictions[0]

        formatted_lines = []

        # Add top rollout context if available
        if top_rollout:
            formatted_lines.append("**Highest-Utility Conversation Path Predicted:**")
            formatted_lines.append(f"Utility Score: {top_rollout.utility_score:.3f} (Higher is better)")
            formatted_lines.append(f"- Expected new subtopics covered: {top_rollout.expected_coverage_delta}")
            formatted_lines.append(f"- Emergence potential: {top_rollout.emergence_potential:.2f}")
            formatted_lines.append(f"- Cost (turns): {top_rollout.cost_estimate}")
            formatted_lines.append("")
            formatted_lines.append("The questions below are optimized to align with this high-utility path.")
            formatted_lines.append("")

        # Format suggestions by priority (high to low)
        sorted_suggestions = sorted(suggestions, key=lambda x: x.get('priority', 0), reverse=True)

        formatted_lines.append("**Strategic Question Suggestions (sorted by priority):**")
        formatted_lines.append("")
        for i, suggestion in enumerate(sorted_suggestions, 1):
            formatted_lines.append(f"{i}. **{suggestion['content']}**")
            formatted_lines.append(f"   - Target: Subtopic {suggestion['subtopic_id']}")
            formatted_lines.append(f"   - Strategy: {suggestion['strategy_type']}")
            formatted_lines.append(f"   - Priority: {suggestion['priority']}/10")
            formatted_lines.append(f"   - Reasoning: {suggestion['reasoning']}")
            formatted_lines.append("")  # Blank line between suggestions

        return "\n".join(formatted_lines)

    def _should_include_strategic_questions(self) -> bool:
        """
        Determine if strategic questions should be included in the prompt.

        Strategic questions become stale after exceeding the rollout horizon.
        Only include them if they are fresh (within horizon + buffer).

        Returns:
            bool: True if strategic questions should be included, False if stale
        """
        strategic_state = self.interview_session.exploration_planner.strategic_state

        # If no suggestions exist, don't include
        if not strategic_state.strategic_question_suggestions:
            return False

        # Calculate current turn (count User messages)
        current_turn = len([
            m for m in self.interview_session.chat_history
            if m.role == "User"
        ])

        # Get last planning turn from strategic state
        last_planning_turn = strategic_state.last_planning_turn

        # If planning hasn't run yet (turn 0), don't include
        if last_planning_turn == 0:
            return False

        # Get rollout horizon from exploration planner
        rollout_horizon = self.interview_session.exploration_planner.rollout_horizon

        # Calculate staleness: questions are stale if we're beyond horizon + buffer
        # Buffer of 2 turns accounts for: 1) planning completes after trigger, 2) grace period
        staleness_threshold = last_planning_turn + rollout_horizon + 2

        # Include questions only if NOT stale
        is_fresh = current_turn <= staleness_threshold

        if not is_fresh:
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] (Interviewer) Strategic questions are stale "
                f"(last_planning_turn={last_planning_turn}, current_turn={current_turn}, "
                f"threshold={staleness_threshold}). Excluding from prompt."
            )

        return is_fresh
