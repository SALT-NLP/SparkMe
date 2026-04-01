import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict
import signal
import contextlib

import time
from tiktoken import get_encoding

from src.agents.base_agent import BaseAgent
from src.interview_session.session_models import Message, MessageType, Participant
from src.agents.interviewer.interviewer import Interviewer, InterviewerConfig, TTSConfig
from src.agents.agenda_manager.agenda_manager import AgendaManager, AgendaManagerConfig
from src.agents.exploration_planner.exploration_planner import ExplorationPlanner, ExplorationPlannerConfig
from src.agents.user.user_agent import UserAgent
from src.content.session_agenda.session_agenda import SessionAgenda
from src.utils.data_process import save_feedback_to_csv
from src.utils.logger.session_logger import SessionLogger, setup_logger
from src.utils.logger.evaluation_logger import EvaluationLogger
from src.interview_session.user.user import User
from src.interview_session.user.dummy_participant import UserDummyParticipant
from src.agents.report_team.orchestrator import ReportOrchestrator
from src.agents.report_team.base_report_agent import ReportConfig
from src.content.memory_bank.memory_bank_vector_db import VectorMemoryBank
from src.content.memory_bank.memory import Memory
from src.content.question_bank.question_bank_vector_db import QuestionBankVectorDB
from src.interview_session.prompts.conversation_summarize import summarize_conversation
from src.utils.token_tracker import TokenUsageTracker




class UserConfig(TypedDict, total=False):
    """Configuration for user settings.
    """
    user_id: str
    enable_voice: bool
    report_style: str


class InterviewConfig(TypedDict, total=False):
    """Configuration for interview settings."""
    enable_voice: bool
    interview_description: str
    interview_plan_path: str
    interview_evaluation: str
    additional_context_path: str
    initial_user_portrait_path: str

class BankConfig(TypedDict, total=False):
    """Configuration for memory and question banks."""
    memory_bank_type: str  # "vector_db", "graph_rag", etc.
    historical_question_bank_type: str  # "vector_db", "graph", "semantic", etc.


class InterviewSession:

    def __init__(self, interaction_mode: str = 'terminal', user_config: UserConfig = {},
                 interview_config: InterviewConfig = {}, bank_config: BankConfig = {},
                 use_baseline: Optional[bool] = None, max_turns: Optional[int] = None):
        """Initialize the interview session.

        Args:
            interaction_mode: How to interact with user 
                Options: 'terminal', 'agent', or 'api'
            user_config: User configuration dictionary
                user_id: User identifier (default: 'default_user')
                enable_voice: Enable voice input (default: False)
            interview_config: Interview configuration dictionary
                enable_voice: Enable voice output (default: False)
            bank_config: Bank configuration dictionary
                memory_bank_type: Type of memory bank 
                    Options: "vector_db", etc.
                historical_question_bank_type: Type of question bank 
                    Options: "vector_db", etc.
            use_baseline: Whether to use baseline prompt (default: read from .env)
            max_turns: Optional maximum number of turns before ending session
                      If None, session continues until manually ended
        """

        # Set the baseline mode for all agents
        if use_baseline is not None:
            # Set the class variable directly to affect all agent instances
            BaseAgent.use_baseline = use_baseline
        else:
            BaseAgent.use_baseline = \
                os.getenv("USE_BASELINE_PROMPT", "false").lower() == "true"
        
        # User setup
        self.user_id = user_config.get("user_id", "default_user")
        self._initial_additional_context_path = interview_config.get("additional_context_path", None)
        self._interview_description = interview_config.get("interview_description", "any topic")

        # Session agenda setup
        self.session_agenda = SessionAgenda.get_last_session_agenda(self.user_id,
                                                                    initial_user_portrait_path=interview_config.get('initial_user_portrait_path'),
                                                                    interview_plan_path=interview_config.get('interview_plan_path'),
                                                                    interview_description=self._interview_description,
                                                                    interview_evaluation=interview_config.get('interview_evaluation'))
        self.session_id = self.session_agenda.session_id + 1

        # Memory bank setup
        memory_bank_type = bank_config.get("memory_bank_type", "vector_db")
        if memory_bank_type == "vector_db":
            self.memory_bank = VectorMemoryBank.load_from_file(self.user_id)
            self.memory_bank.set_session_id(self.session_id)
        else:
            raise ValueError(f"Unknown memory bank type: {memory_bank_type}")

        # Question bank setup
        historical_question_bank_type = \
            bank_config.get("historical_question_bank_type", "vector_db")
        if historical_question_bank_type == "vector_db":
            self.historical_question_bank = \
                QuestionBankVectorDB.load_from_file(
                    self.user_id)
            self.historical_question_bank.set_session_id(self.session_id)
            self.proposed_question_bank = QuestionBankVectorDB()
        else:
            raise ValueError(
                f"Unknown question bank type: {historical_question_bank_type}")

        # Logger setup
        setup_logger(self.user_id, self.session_id,
                     console_output_files=["execution_log"])
        EvaluationLogger.setup_logger(self.user_id, self.session_id)

        # Token usage tracking setup
        self.token_tracker = TokenUsageTracker(
            session_id=str(self.session_id),
            user_id=self.user_id
        )
        # Set the class variable so all agents can access it
        BaseAgent.token_tracker = self.token_tracker

        # Chat history
        self.chat_history: list[Message] = []

        # Session states signals
        self.interaction_mode = interaction_mode
        self.session_in_progress = True
        self.session_completed = False
        self._session_timeout = False
        self.max_turns = max_turns

        # Report auto-update states
        self.auto_report_update_in_progress = False
        self.memory_threshold = int(
            os.getenv("MEMORY_THRESHOLD_FOR_UPDATE", 10))
        
        # Conversation summary for auto-updates
        self.conversation_summary = ""
        
        # Counter for user messages to trigger auto-updates check
        self._user_message_count = 0
        self._check_interval = max(1, self.memory_threshold // 5)
        self._accumulated_auto_update_time = 0

        # Last message timestamp tracking for session timeout
        self._last_message_time = datetime.now()
        self._last_user_message = None
        self.timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", 10))

        # User in the interview session
        if interaction_mode == 'agent':
            self.user: User = UserAgent(
                user_id=self.user_id, interview_session=self, 
                config=user_config)
        elif interaction_mode == 'terminal':
            self.user: User = User(user_id=self.user_id, interview_session=self,
                                   enable_voice_input=user_config \
                                   .get("enable_voice", False))
        elif interaction_mode == 'api':
            self.user: User = UserDummyParticipant(user_id=self.user_id, interview_session=self) # No direct user interface for API mode
        else:
            raise ValueError(f"Invalid interaction_mode: {interaction_mode}")

        # Agents in the interview session
        self._interviewer: Interviewer = Interviewer(
            config=InterviewerConfig(
                user_id=self.user_id,
                tts=TTSConfig(enabled=interview_config.get(
                    "enable_voice", False)),
                interview_description=self._interview_description,
            ),
            interview_session=self
        )
        # AgendaManager config with optional dedicated model
        scribe_config = AgendaManagerConfig(user_id=self.user_id)

        # Use dedicated scribe model if configured
        scribe_model = os.getenv("AGENDA_MANAGER_MODEL_NAME")
        if scribe_model:
            scribe_config["model_name"] = scribe_model
            # Pass base_url if configured (for vLLM)
            scribe_base_url = os.getenv("AGENDA_MANAGER_VLLM_BASE_URL")
            if scribe_base_url:
                scribe_config["base_url"] = scribe_base_url

        self.agenda_manager = AgendaManager(
            config=scribe_config,
            interview_session=self
        )
        
        # ExplorationPlanner config
        # TODO: Tune exploration planner parameters
        planner_config = ExplorationPlannerConfig(
                user_id=self.user_id,
                turn_trigger=int(os.getenv("EXPLORATION_PLANNER_TURN_TRIGGER", "3")),
                num_rollouts=int(os.getenv("EXPLORATION_PLANNER_NUM_ROLLOUTS", "3")),
                rollout_horizon=int(os.getenv("EXPLORATION_PLANNER_ROLLOUT_HORIZON", "3")),
                max_strategic_questions=int(os.getenv("EXPLORATION_PLANNER_MAX_QUESTIONS", "5")),
                alpha=float(os.getenv("EXPLORATION_PLANNER_ALPHA", "0.5")),  # Coverage weight
                beta=float(os.getenv("EXPLORATION_PLANNER_BETA", "0.3")),   # Cost penalty
                gamma=float(os.getenv("EXPLORATION_PLANNER_GAMMA", "0.2"))   # Emergence reward
        )
        
        # Use dedicated planner model if configured
        planner_model = os.getenv("EXPLORATION_PLANNER_MODEL_NAME")
        if planner_model:
            planner_config["model_name"] = planner_model
            # Pass base_url if configured (for vLLM)
            planner_base_url = os.getenv("EXPLORATION_PLANNER_VLLM_BASE_URL")
            if planner_base_url:
                planner_config["base_url"] = planner_base_url
        
        self.exploration_planner: ExplorationPlanner = ExplorationPlanner(
            config=planner_config,
            interview_session=self
        )
        self.report_orchestrator = ReportOrchestrator(
            config=ReportConfig(
                user_id=self.user_id,
                report_style=user_config.get(
                    "report_style", "chronological")
            ),
            interview_session=self
        )

        # Subscriptions of participants to each other
        self._subscriptions: Dict[str, List[Participant]] = {
            # Subscribers of Interviewer: Note-taker and User (in following code)
            "Interviewer": [self.agenda_manager],
            # Subscribers of User: Interviewer, AgendaManager, and ExplorationPlanner
            "User": [self._interviewer, self.agenda_manager, self.exploration_planner]
        }

        # User participant for terminal interaction
        if self.user:
            self._subscriptions["Interviewer"].append(self.user)

        # User API participant for backend API interaction
        # self.api_participant = None
        # if interaction_mode == 'api':
        #     self.api_participant = UserDummyParticipant(interview_session=self)
        #     self._subscriptions["Interviewer"].append(self.api_participant)
        #     self._subscriptions["User"].append(self.api_participant)

        # Shutdown signal handler - only for agent mode
        if interaction_mode == 'agent':
            self._setup_signal_handlers()
        
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Interview session initialized")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] User ID: {self.user_id}")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Session ID: {self.session_id}")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Use baseline: {BaseAgent.use_baseline}")
        
        self.tokenizer = get_encoding("cl100k_base")

    async def _notify_participants(self, message: Message):
        """Notify subscribers asynchronously"""
        # Gets subscribers for the user that sent the message.
        subscribers = self._subscriptions.get(message.role, [])
        SessionLogger.log_to_file(
            "execution_log", 
            (
                f"[NOTIFY] Notifying {len(subscribers)} subscribers "
                f"for message from {message.role}"
            )
        )

        # Create independent tasks for each subscriber
        tasks = []
        for sub in subscribers:
            if self.session_in_progress:
                task = asyncio.create_task(sub.on_message(message))
                tasks.append(task)
        
        # Allow tasks to run concurrently without waiting for each other
        await asyncio.sleep(0)  # Explicitly yield control

        # Special handling for user messages after notifying participants
        if message.role == "User":
            self._last_user_message = message
            self._user_message_count += 1

            # Update the turn counter for token tracking
            BaseAgent.current_turn = self._user_message_count

            # Save token usage snapshot every turn
            snapshot_path = self.token_tracker.save_snapshot()
            SessionLogger.log_to_file(
                "execution_log",
                f"[TOKEN_TRACKING] Saved token usage snapshot to {snapshot_path}",
                log_level="info"
            )

            # Check if we need to trigger a report update
            if (self._user_message_count % self._check_interval == 0 and
                not self.auto_report_update_in_progress):
                asyncio.create_task(self._check_and_trigger_report_update())

            # Check if max turns reached
            if self.max_turns is not None and \
                    self._user_message_count >= self.max_turns:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TURNS] Maximum turns ({self.max_turns}) reached. "
                    f"Ending session."
                )
                self.session_in_progress = False
                # Save final token usage summary
                final_summary_path = self.token_tracker.save_final_summary()
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                    log_level="info"
                )
            elif self.session_agenda.all_core_topics_completed():
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TOPICS] All topics for this session have been completed. "
                    f"Ending session."
                )
                self.session_in_progress = False
                # Save final token usage summary
                final_summary_path = self.token_tracker.save_final_summary()
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                    log_level="info"
                )

    def add_message_to_chat_history(self, role: str, content: str = "", 
                                    message_type: str = MessageType.CONVERSATION,
                                    metadata: dict = {}):
        """Add a message to the chat history"""

        # Reject messages if session is not in progress
        if not self.session_in_progress:
            return

        # Set fixed content for skip and like messages
        if message_type == MessageType.SKIP:
            content = "Skip the question"
        elif message_type == MessageType.LIKE:
            content = "Like the question"

        # Create message object
        message = Message(
            id=str(uuid.uuid4()),
            type=message_type,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        if role == "User":
            self._last_message_time = message.timestamp
        elif role == "Interviewer" and self._last_user_message is not None:
            self._last_user_message = None
        
        # Log feedback
        if message_type != MessageType.CONVERSATION:
            save_feedback_to_csv(
                self.chat_history[-1], message, self.user_id, self.session_id)

        # Notify participants if message is a skip or conversation
        if message_type == MessageType.SKIP or \
              message_type == MessageType.CONVERSATION:
            
            # Add message to chat history
            self.chat_history.append(message)
            SessionLogger.log_to_file(
                "chat_history", f"{message.role}: {message.content}")
            
            # Notify participants
            asyncio.create_task(self._notify_participants(message))


        SessionLogger.log_to_file(
            "execution_log", 
            (
                f"[CHAT_HISTORY] {message.role}'s message has been added "
                f"to chat history."
            )
        )

    async def run(self):
        """Run the interview session"""
        # Augment session agenda with existing profile if applicable
        await self.agenda_manager.augment_session_agenda(additional_context_path=self._initial_additional_context_path)

        SessionLogger.log_to_file(
            "execution_log", f"[RUN] Starting interview session")
        self.session_in_progress = True

        # In-interview Processing
        try:
            # Interviewer initiate the conversation (if not in API mode)
            if self.user is not None:
                await self._interviewer.on_message(None)

            # Monitor the session for completion and timeout
            while self.session_in_progress or \
                self.agenda_manager.processing_in_progress:
                await asyncio.sleep(0.1)

                # Check for timeout
                if datetime.now() - self._last_message_time \
                        > timedelta(minutes=self.timeout_minutes):
                    SessionLogger.log_to_file(
                        "execution_log", 
                        (
                            f"[TIMEOUT] Session timed out after "
                            f"{self.timeout_minutes} minutes of inactivity"
                        )
                    )
                    self.session_in_progress = False
                    self._session_timeout = True
                    break

        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log", f"[RUN] Unexpected error: {str(e)}")
            raise e

        # Post-interview Processing
        finally:
            try:
                self.session_in_progress = False

                # Update report (API mode handles this separately)
                if self.interaction_mode != 'api' or self._session_timeout:
                    with contextlib.suppress(KeyboardInterrupt):
                        SessionLogger.log_to_file(
                            "execution_log", 
                            (
                                f"[REPORT] Trigger final report update. "
                                f"Waiting for agenda manager to finish processing..."
                            )
                        )
                        await self.final_update_report_and_agenda(
                            selected_topics=[])

                # Wait for report update to complete if it's in progress
                start_time = time.time()
                while (self.report_orchestrator.report_update_in_progress or 
                       self.report_orchestrator.session_agenda_update_in_progress):
                    await asyncio.sleep(0.1)
                    if time.time() - start_time > 600:  # 10 minutes timeout
                        SessionLogger.log_to_file(
                            "execution_log", 
                            (
                                f"[REPORT] Timeout waiting for report update"
                            )
                        )
                        break

            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log", f"[RUN] Error during report update: \
                          {str(e)}")
            finally:
                # Save memory bank
                self.memory_bank.save_to_file(self.user_id)
                SessionLogger.log_to_file(
                    "execution_log", f"[COMPLETED] Memory bank saved")
                
                # Save historical question bank
                self.historical_question_bank.save_to_file(self.user_id)
                SessionLogger.log_to_file(
                    "execution_log", f"[COMPLETED] Question bank saved")
                       
                self.session_completed = True
                SessionLogger.log_to_file(
                    "execution_log", f"[COMPLETED] Session completed")

    async def get_session_memories(self, include_processed=True) -> List[Memory]:
        """Get memories added during this session
        
        Args:
            include_processed: If True, returns all memories from the session
                              If False, returns only the unprocessed memories
        """
        return await self.agenda_manager.get_session_memories(
            clear_processed=False, 
            wait_for_processing=True,
            include_processed=include_processed
        )

    async def _check_and_trigger_report_update(self):
        """Check if we have enough memories to trigger a report update"""
        # Skip if report update already in progress or session not in progress
        if self.auto_report_update_in_progress or \
           not self.session_in_progress or \
           self.report_orchestrator.report_update_in_progress:
            return
            
        # Get current memory count without clearing or waiting
        memories = await self.agenda_manager \
            .get_session_memories(clear_processed=False,
                                   wait_for_processing=False)
        
        # Check if we've reached the threshold
        if len(memories) >= self.memory_threshold:
            SessionLogger.log_to_file(
                "execution_log",
                f"[AUTO-UPDATE] Triggering report update "
                f"with {len(memories)} memories"
            )
            
            try:
                self.auto_report_update_in_progress = True
                
                # Generate a summary of recent conversation
                await self._update_conversation_summary()
                
                # Get memories and clear them from the agenda manager
                memories_to_process = \
                    await self.agenda_manager.get_session_memories(
                        clear_processed=True, wait_for_processing=False)
                
                # Measure the time auto-update would take
                start_time = time.time()
                
                # Update report with these memories and the conversation summary
                # await self.report_orchestrator.update_report_with_memories(
                #     memories_to_process,
                #     is_auto_update=True
                # )
                
                # Record the time it took
                update_time = time.time() - start_time
                self._accumulated_auto_update_time += update_time
                
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[AUTO-UPDATE] Report update completed "
                    f"for {len(memories_to_process)} memories"
                )
                
            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log", 
                    f"[AUTO-UPDATE] Error during report update: {str(e)}"
                )
            finally:
                self.auto_report_update_in_progress = False
    
    async def _update_conversation_summary(self):
        """Generate a summary of recent conversation messages"""
        
        # Extract recent messages from chat history
        recent_messages: List[Message] = []
        for msg in self.chat_history[-self.agenda_manager._max_events_len:]:
            if msg.type == MessageType.CONVERSATION:
                recent_messages.append(msg)
        
        # Generate summary if we have messages
        if recent_messages:
            self.conversation_summary = \
                summarize_conversation(recent_messages)
    
    async def final_update_report_and_agenda(self, selected_topics: Optional[List[str]] = None):
        """Trigger final report update"""
        # Record start time
        start_time = time.time()
        
        try:
            # Proceed with the final update
            await self.report_orchestrator.final_update_report_and_agenda(
                selected_topics=selected_topics,
                wait_time=self._accumulated_auto_update_time if \
                    (BaseAgent.use_baseline and 
                    self.interaction_mode == "api") else None
                # Simulate baseline mode without auto-updates for web user testing
            )
        finally:
            # Calculate and log duration
            duration = time.time() - start_time
            eval_logger = EvaluationLogger.setup_logger(
                self.user_id, self.session_id)
            eval_logger.log_report_update_time(
                update_type="final",
                duration=duration if not BaseAgent.use_baseline \
                    else (duration + self._accumulated_auto_update_time),
                accumulated_auto_time=self._accumulated_auto_update_time
                # Simulate baseline mode without auto-updates
            )

    def end_session(self):
        """End the session without triggering report update"""
        self.session_in_progress = False

        # Save final token usage summary
        if hasattr(self, 'token_tracker'):
            final_summary_path = self.token_tracker.save_final_summary()
            SessionLogger.log_to_file(
                "execution_log",
                f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                log_level="info"
            )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

    def _signal_handler(self):
        """Handle shutdown signals"""
        self.session_in_progress = False
        SessionLogger.log_to_file(
            "execution_log", f"[SIGNAL] Shutdown signal received")
        SessionLogger.log_to_file(
            "execution_log", f"[SIGNAL] Waiting for interview session to finish...")
    
    def set_db_session_id(self, db_session_id: int):
        """Set the database session ID. Used for server mode"""
        self.db_session_id = db_session_id

    def get_db_session_id(self) -> int:
        """Get the database session ID. Used for server mode"""
        return self.db_session_id
        