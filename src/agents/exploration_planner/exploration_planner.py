"""
Exploration Planner Agent for Long-Term Interview Planning

This agent runs every X conversation turns to provide strategic guidance
through rollout prediction, coverage analysis, and emergence detection.
"""

import asyncio
import os
import random
import json
from typing import TYPE_CHECKING, Dict, List, Tuple
from typing_extensions import TypedDict

from src.agents.base_agent import BaseAgent
from src.interview_session.session_models import Message, Participant
from src.utils.logger.session_logger import SessionLogger

from src.agents.exploration_planner.strategic_state import StrategicState, ConversationRollout
from src.agents.exploration_planner.tools import (
    SuggestStrategicQuestions,
    AddEmergentSubtopic,
    IdentifyEmergentInsights
)
from src.content.session_agenda.topic_evaluator import get_registry
from src.agents.exploration_planner.prompts import get_prompt
from src.utils.llm.prompt_utils import format_prompt

if TYPE_CHECKING:
    from src.interview_session.interview_session import InterviewSession


class ExplorationPlannerConfig(TypedDict, total=False):
    """Configuration for Exploration Planner.

    Most configuration comes from environment variables.
    Only required fields are included here.
    """
    user_id: str


class ExplorationPlanner(BaseAgent, Participant):
    """
    Long-term strategic planning agent that complements AgendaManager.

    Runs every 3-5 conversation turns to:
    - Analyze strategic coverage across all topics
    - Predict conversation rollouts with utility scoring
    - Detect emergent insights (novel, counter-intuitive findings)
    - Generate strategic questions optimized for U = α·Coverage - β·Cost + γ·Emergence

    This provides predictive, goal-oriented planning while AgendaManager
    handles reactive, short-term planning.
    """

    def __init__(
        self,
        config: ExplorationPlannerConfig,
        interview_session: 'InterviewSession'
    ):
        """
        Initialize Exploration Planner.

        Args:
            config: Configuration dictionary
            interview_session: Reference to parent interview session
        """
        # Initialize BaseAgent
        BaseAgent.__init__(
            self,
            name="ExplorationPlanner",
            description="Long-term strategic planning agent for interview optimization",
            config=config
        )

        # Initialize Participant
        Participant.__init__(
            self,
            title="ExplorationPlanner",
            interview_session=interview_session
        )

        # Configuration
        self.user_id = config["user_id"]

        # Load config from environment variables with fallbacks
        self.turn_trigger = int(os.getenv("EXPLORATION_PLANNER_TURN_TRIGGER", "3"))
        self.num_rollouts = int(os.getenv("EXPLORATION_PLANNER_NUM_ROLLOUTS", "3"))
        self.rollout_horizon = int(os.getenv("EXPLORATION_PLANNER_ROLLOUT_HORIZON", "3"))
        self.max_strategic_questions = int(os.getenv("EXPLORATION_PLANNER_MAX_QUESTIONS", "5"))

        # Utility function weights
        self.alpha = float(os.getenv("EXPLORATION_PLANNER_ALPHA", "0.5"))  # Coverage weight
        self.beta = float(os.getenv("EXPLORATION_PLANNER_BETA", "0.3"))  # Cost penalty
        self.gamma = float(os.getenv("EXPLORATION_PLANNER_GAMMA", "0.2"))  # Emergence reward
        self.min_novelty_score = int(os.getenv("EXPLORATION_PLANNER_MIN_NOVELTY", "3"))

        # Strategic state (NOT loaded from file, starts fresh each session)
        session_id = interview_session.session_id
        self.strategic_state = StrategicState(
            session_id=session_id,
            last_planning_turn=0
        )

        # Planning control
        self._planning_in_progress = False
        self._planning_lock = asyncio.Lock()
        self._last_planning_turn = 0

        # Initialize tools
        self.tools = {
            "suggest_strategic_questions": SuggestStrategicQuestions(
                strategic_state=self.strategic_state,
                session_agenda=self.interview_session.session_agenda,
                alpha=self.alpha,
                gamma=self.gamma,
            ),
            "add_emergent_subtopic": AddEmergentSubtopic(
                session_agenda=self.interview_session.session_agenda
            ),
            "identify_emergent_insights": IdentifyEmergentInsights(
                session_agenda=self.interview_session.session_agenda,
                min_novelty_score=3
            ),
        }

    async def on_message(self, message: Message):
        """
        Handle incoming messages and trigger strategic planning when appropriate.

        Args:
            message: Message from the interview session
        """
        # Only process user messages (turn tracking)
        if message.role != "User":
            return

        # Count total user turns
        current_turn = len([
            m for m in self.interview_session.chat_history
            if m.role == "User"
        ])

        # Log if planning is in progress
        if self._planning_in_progress:
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] ({self.name}) Planning still in progress from turn {self._last_planning_turn}"
            )

        # Check if should trigger planning
        if self._should_trigger_planning(current_turn):
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] ({self.name}) Triggering strategic planning at turn {current_turn}"
            )
            # Run planning in background (non-blocking)
            asyncio.create_task(self._run_strategic_planning())

    def _should_trigger_planning(self, current_turn: int) -> bool:
        """
        Determine if strategic planning should run.

        Args:
            current_turn: Current conversation turn number

        Returns:
            True if planning should trigger
        """
        # Don't trigger if already planning
        if self._planning_in_progress:
            return False

        # Calculate turns since last planning
        turns_since_last = current_turn - self._last_planning_turn

        # Trigger if minimum turns have passed (deterministic)
        # Use turn_trigger_min as the consistent threshold
        return turns_since_last >= self.turn_trigger

    async def _run_strategic_planning(self):
        """
        Main strategic planning workflow with proper locking.

        Runs in background with 5-10s latency. Executes:
        1. Conversation rollout prediction
        2. Emergent insight identification
        3. Brainstorm emergent subtopics
        4. Strategic question generation
        5. Update SessionAgenda with strategic data and state persistence
        """
        # Check if already planning (outside lock for efficiency)
        if self._planning_in_progress:
            SessionLogger.log_to_file(
                "execution_log",
                f"({self.name}) Planning already in progress, skipping..."
            )
            return

        async with self._planning_lock:
            # Double-check after acquiring lock
            if self._planning_in_progress:
                return

            self._planning_in_progress = True

            try:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[NOTIFY] ({self.name}) === Starting Strategic Planning ==="
                )

                # Update last planning turn
                current_turn = len([
                    m for m in self.interview_session.chat_history
                    if m.role == "User"
                ])
                self._last_planning_turn = current_turn
                self.strategic_state.last_planning_turn = current_turn

                # Phase 1-2: Run all analysis in parallel (no dependencies)
                await asyncio.gather(
                    self._brainstorm_emergent_subtopic(), # Strategic: New subtopics based on patterns
                    self._identify_emergent_insights(),  # Strategic: New insights
                    self._predict_conversation_rollout()   # Future: Predicted trajectories
                )

                # Phase 3: Generate strategic questions (depends on all above)
                await self._generate_strategic_questions()

                # Phase 4: Save state
                await self._save_state_snapshot()

                SessionLogger.log_to_file(
                    "execution_log",
                    f"({self.name}) === Strategic Planning Complete ==="
                )

            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"({self.name}) Error during strategic planning: {e}",
                    log_level="error"
                )
                raise

            finally:
                self._planning_in_progress = False

    async def _draft_multiple_rollouts(self) -> List[Dict[str, any]]:
        """
        Draft multiple conversation rollouts in a SINGLE LLM call.

        This replaces the previous parallel multi-call approach with a single
        prompt that generates N rollouts at once.

        Returns:
            List of rollout_data dicts, each containing:
            - rollout_id: str
            - predicted_turns: List of {question, predicted_response, subtopics_covered, emergence_potential, reasoning}
        """
        # Build prompt for multi-rollout generation
        prompt = self._get_formatted_prompt("draft_rollouts")

        # Log prompt to event stream
        self.add_event(
            sender=self.name,
            tag="draft_rollouts_prompt",
            content=prompt
        )

        response = await self.call_engine_async(prompt)

        # Log response to event stream
        self.add_event(
            sender=self.name,
            tag="draft_rollouts_response",
            content=response
        )

        # Parse JSON response containing multiple rollouts
        try:
            # Extract JSON from response (may be wrapped in markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            rollouts_data = json.loads(json_str)

            # Validate structure
            if not isinstance(rollouts_data, dict) or "rollouts" not in rollouts_data:
                raise ValueError("Response must contain 'rollouts' key")

            return rollouts_data["rollouts"]

        except (json.JSONDecodeError, ValueError) as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"({self.name}) Error parsing rollout JSON: {e}",
                log_level="error"
            )
            return []

    async def _judge_coverage_impact(self, rollout_data: Dict[str, any]) -> Dict[str, any]:
        """
        Judge predicted coverage impact for a rollout using LLM evaluation.

        Uses AgendaManager's STAR framework to validate which subtopics would
        actually be covered by the predicted Q&A exchanges.

        Args:
            rollout_data: Dict containing rollout_id, predicted_turns, confidence_score

        Returns:
            List[str] of subtopic IDs predicted to be covered
        """
        # Build coverage judging prompt
        prompt = self._get_formatted_prompt(
            "judge_coverage",
            rollout_data=rollout_data
        )

        # Log prompt to event stream
        self.add_event(
            sender=self.name,
            tag=f"judge_coverage_prompt_{rollout_data.get('rollout_id', 'unknown')}",
            content=prompt
        )

        response = await self.call_engine_async(prompt)

        # Log response to event stream
        self.add_event(
            sender=self.name,
            tag=f"judge_coverage_response_{rollout_data.get('rollout_id', 'unknown')}",
            content=response
        )

        # Parse JSON response (now expects array of {subtopic_id, coverage_rationale})
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            coverage_array = json.loads(json_str)

            # Extract subtopic IDs from the simplified array format
            if isinstance(coverage_array, list):
                subtopics_covered = [item["subtopic_id"] for item in coverage_array if "subtopic_id" in item]
            else:
                subtopics_covered = []

            return subtopics_covered

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"({self.name}) Error parsing coverage judgment JSON: {e}",
                log_level="error"
            )
            return []

    def _calculate_hypothetical_utility(
        self,
        subtopics_to_cover: List[str],
        emergence_potential: float,
        cost_estimate: int
    ) -> Tuple[int, float]:
        """
        Calculate hypothetical utility based on marginal coverage.

        Simpler approach: Count how many subtopics would be newly covered
        (exist + not already covered) and use that as the coverage delta.

        U = α·(newly_covered_count) - β·Cost + γ·Emergence

        Args:
            subtopics_to_cover: List of subtopic IDs predicted to be covered
            emergence_potential: Predicted emergence score (0-1)
            cost_estimate: Number of turns in this rollout

        Returns:
            Tuple of (newly_covered_count, utility_score)
        """
        manager = self.interview_session.session_agenda.interview_topic_manager

        # Track validation: already covered, newly covered
        newly_covered = []

        # Check each predicted subtopic
        for subtopic_id in subtopics_to_cover:
            # Check all topics for this subtopic
            for topic in manager.get_all_topics():
                # Check required subtopics
                if self.alpha > 0 and subtopic_id in topic.required_subtopics:
                    if not topic.required_subtopics[subtopic_id].check_coverage():
                        newly_covered.append(subtopic_id)
                    break
                # Check emergent subtopics
                elif self.gamma > 0 and subtopic_id in topic.emergent_subtopics:
                    if not topic.emergent_subtopics[subtopic_id].check_coverage():
                        newly_covered.append(subtopic_id)
                    break

        if newly_covered:
            SessionLogger.log_to_file(
                "execution_log",
                f"({self.name}) Judge predicted {len(newly_covered)} newly-covered subtopics (marginal): {newly_covered}"
            )

        # Coverage delta is simply the count of newly covered subtopics
        newly_covered_count = len(newly_covered)

        # Calculate utility: U = α·(newly_covered_count) - β·Cost + γ·Emergence
        utility_score = (
            self.alpha * newly_covered_count -
            self.beta * cost_estimate +
            self.gamma * emergence_potential
        )

        return newly_covered_count, utility_score

    async def _predict_conversation_rollout(self):
        """
        Workflow:
        1. Draft N rollouts in SINGLE LLM call
        2. Judge coverage impact for each rollout using LLM
        3. Calculate hypothetical utility using evaluator
        4. Store rollouts ranked by utility

        Uses utility function: U = α·Coverage - β·Cost + γ·Emergence
        """
        SessionLogger.log_to_file(
            "execution_log",
            f"[NOTIFY] {self.name}: Generating {self.num_rollouts} rollouts in single call..."
        )

        # Step 1: Draft multiple rollouts in one LLM call
        rollouts_data = await self._draft_multiple_rollouts()

        if not rollouts_data:
            SessionLogger.log_to_file(
                "execution_log",
                f"({self.name}) No rollouts generated, skipping prediction",
                log_level="warning"
            )
            return

        SessionLogger.log_to_file(
            "execution_log",
            f"({self.name}) Generated {len(rollouts_data)} rollout drafts"
        )

        # Step 2-3: Judge coverage and calculate utility for each rollout
        self.strategic_state.rollout_predictions = []

        # Step 2: Judge coverage impact for ALL rollouts in parallel
        SessionLogger.log_to_file(
            "execution_log",
            f"({self.name}) Judging coverage for {len(rollouts_data)} rollouts in parallel..."
        )

        judgment_tasks = [
            self._judge_coverage_impact(rollout_data)
            for rollout_data in rollouts_data
        ]
        all_subtopics_covered = await asyncio.gather(*judgment_tasks)

        # Step 3: Calculate utility for each rollout using parallel judgment results
        for i, rollout_data in enumerate(rollouts_data):
            # Extract basic info from draft
            rollout_id = rollout_data.get("rollout_id", "unknown")
            predicted_turns = rollout_data.get("predicted_turns", [])

            # Aggregate emergence potential from turns
            emergence_potential = sum(
                turn.get("emergence_potential", 0.0) for turn in predicted_turns
            ) if predicted_turns else 0.0

            # Get pre-computed coverage judgment from parallel execution
            subtopics_covered = all_subtopics_covered[i]

            # Calculate hypothetical utility
            coverage_delta, utility_score = self._calculate_hypothetical_utility(
                subtopics_to_cover=subtopics_covered,
                emergence_potential=emergence_potential,
                cost_estimate=len(predicted_turns)
            )

            # Create rollout object with calculated metrics
            rollout = ConversationRollout(
                rollout_id=rollout_id,
                predicted_turns=predicted_turns,
                expected_coverage_delta=coverage_delta,
                emergence_potential=emergence_potential,
                cost_estimate=len(predicted_turns),
                utility_score=utility_score
            )

            self.strategic_state.rollout_predictions.append(rollout)

            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] ({self.name}) Rollout {rollout_id}: utility={utility_score:.3f}, "
                f"newly_covered={coverage_delta} subtopics, emergence={emergence_potential:.2f}"
            )

        # Sort rollouts by utility score (highest first)
        self.strategic_state.rollout_predictions.sort(
            key=lambda r: r.utility_score,
            reverse=True
        )

        if self.strategic_state.rollout_predictions:
            top_rollout = self.strategic_state.rollout_predictions[0]
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] ({self.name}) Rollout prediction complete. "
                f"Top utility score: {top_rollout.utility_score:.3f} "
                f"(newly_covered={top_rollout.expected_coverage_delta} subtopics, "
                f"emergence={top_rollout.emergence_potential:.2f})"
            )

    async def _brainstorm_emergent_subtopic(self):
        """
        Identifies new subtopic areas to explore based on conversation flow
        and adds them to SessionAgenda.
        """
        SessionLogger.log_to_file(
            "execution_log",
            f"[NOTIFY] {self.name}: Brainstorming emergent subtopics..."
        )

        # Use centralized prompt builder
        prompt = self._get_formatted_prompt("brainstorm_emergent_subtopic")

        # Log prompt to event stream
        self.add_event(
            sender=self.name,
            tag="brainstorm_emergent_subtopic_prompt",
            content=prompt
        )

        response = await self.call_engine_async(prompt)

        # Log response to event stream
        self.add_event(
            sender=self.name,
            tag="brainstorm_emergent_subtopic_response",
            content=response
        )

        await self.handle_tool_calls_async(response)

        SessionLogger.log_to_file(
            "execution_log",
            f"({self.name}) Emergent subtopic brainstorming complete"
        )
        
    async def _identify_emergent_insights(self):
        """
        Identify emergent insights from the recent Q&A pair.

        Quick analysis to detect counter-intuitive findings that contradict
        conventional wisdom or reveal unexpected patterns.
        """
        prompt = self._get_formatted_prompt("identify_emergent_insights")
        self.add_event(
            sender=self.name,
            tag="identify_emergent_insights_prompt",
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name,
            tag="identify_emergent_insights_response",
            content=response
        )

        # Handle tool calls (LLM decides whether to call tool or not)
        self.handle_tool_calls(response)

    async def _generate_strategic_questions(self):
        """
        Generates questions optimized for the highest-utility rollout path,
        targeting coverage gaps, emergent insights, and strategic progression.
        """
        SessionLogger.log_to_file(
            "execution_log",
            f"[NOTIFY] {self.name}: Generating strategic questions..."
        )

        # Use centralized prompt builder
        prompt = self._get_formatted_prompt(
            "generate_strategic_questions"
        )

        # Log prompt to event stream
        self.add_event(
            sender=self.name,
            tag="generate_strategic_questions_prompt",
            content=prompt
        )

        response = await self.call_engine_async(prompt)

        # Log response to event stream
        self.add_event(
            sender=self.name,
            tag="generate_strategic_questions_response",
            content=response
        )

        await self.handle_tool_calls_async(response)

        SessionLogger.log_to_file(
            "execution_log",
            f"({self.name}) Strategic question generation complete"
        )

    async def _save_state_snapshot(self):
        """Save snapshot of strategic state to file."""
        current_turn = len([
            m for m in self.interview_session.chat_history
            if m.role == "User"
        ])
        self.strategic_state.save_snapshot(self.user_id, current_turn)
        
        SessionLogger.log_to_file(
            "execution_log",
            f"[NOTIFY] {self.name}: Strategic state snapshot saved (turn {current_turn})"
        )

    def _get_formatted_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        Get formatted prompt for strategic planning tasks.

        Args:
            prompt_type: Type of prompt to generate
            **kwargs: Additional context parameters

        Returns:
            Formatted prompt string ready for LLM
        """
        # Get base prompt template from factory
        prompt = get_prompt(prompt_type)
        all_topics = self.interview_session.session_agenda.get_questions_and_notes_str(hide_answered="all",
                                                                                       active_topics_only=False)

        if prompt_type == "draft_rollouts":
            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "topics_list": all_topics,
                "previous_events": self._get_recent_conversation(),
                "interview_description": self.interview_session.session_agenda.interview_description,
                "num_rollouts": self.num_rollouts,
                "num_horizon": self.rollout_horizon,
            })
        elif prompt_type == "judge_coverage":
            rollout_data = kwargs.get("rollout_data", {})
            return format_prompt(prompt, {
                "rollout_data": rollout_data,
                "topics_list": all_topics,
            })
        elif prompt_type == "brainstorm_emergent_subtopic":
            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "interview_description": self.interview_session.session_agenda.interview_description,
                "previous_events": self._get_recent_conversation(),
                "last_meeting_summary": self.interview_session.session_agenda.get_last_meeting_summary_str(),
                "topics_list": all_topics,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["add_emergent_subtopic"]
                )
            })
        elif prompt_type == "generate_strategic_questions":
            active_topics_only = kwargs.get("active_topics_only", True)
            topics_list = self.interview_session.session_agenda.get_questions_and_notes_str(hide_answered="all",
                                                                                            active_topics_only=active_topics_only)

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "interview_description": self.interview_session.session_agenda.interview_description,
                "max_questions": self.max_strategic_questions,
                "rollout_predictions": self._format_rollouts(),
                "previous_events": self._get_recent_conversation(),
                "last_meeting_summary": self.interview_session.session_agenda.get_last_meeting_summary_str(),
                "topics_list": topics_list,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["suggest_strategic_questions"]
                ),
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma
            })
        elif prompt_type == "identify_emergent_insights":
            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "interview_description": self.interview_session.session_agenda.interview_description,
                "previous_events": self._get_recent_conversation(),
                "last_meeting_summary": self.interview_session.session_agenda.get_last_meeting_summary_str(),
                "topics_list": all_topics,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["identify_emergent_insights"]
                )
            })
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    def _get_recent_conversation(self, n: int = None) -> str:
        """
        Get recent conversation turns.

        Args:
            n: Number of recent turns to include (defaults to _max_events_len)

        Returns:
            Formatted conversation string
        """
        # Use _max_events_len from BaseAgent if n not specified
        if n is None:
            n = self._max_events_len

        recent_messages = self.interview_session.chat_history[-min(n, self._max_events_len):]
        formatted = []

        for msg in recent_messages:
            formatted.append(f"{msg.role}: {msg.content}")

        return "\n".join(formatted)

    def _format_rollouts(self) -> str:
        """Format rollout predictions with full turn details"""
        if not self.strategic_state.rollout_predictions:
            return "No rollouts predicted yet"

        lines = []
        for i, rollout in enumerate(self.strategic_state.rollout_predictions):
            lines.append(
                f"\n=== Rollout {i+1} (utility={rollout.utility_score:.3f}) ===\n"
            )
            
            # Add each predicted turn
            for turn in rollout.predicted_turns:
                turn_num = turn.get('turn_number', '?')
                question = turn.get('question', 'N/A')
                predicted_response = turn.get('predicted_response', 'N/A')
                subtopics = turn.get('subtopics_covered', [])
                emergence = turn.get('emergence_potential', 0.0)
                rationale = turn.get('strategic_rationale', 'N/A')
                
                lines.append(f"\nTurn {turn_num}:")
                lines.append(f"  Q: {question}")
                lines.append(f"  Predicted A: {predicted_response}")
                lines.append(f"  Potential Subtopics Covered: {', '.join(subtopics) if subtopics else 'None'}")
                lines.append(f"  Potential Emergence Score: {emergence:.2f}")
                lines.append(f"  Rationale: {rationale}")

        return "\n".join(lines)

