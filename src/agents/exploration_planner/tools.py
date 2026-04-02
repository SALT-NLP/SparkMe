from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, field_validator

from src.content.session_agenda.session_agenda import SessionAgenda
from src.agents.exploration_planner.strategic_state import (
    StrategicState,
)

class SuggestStrategicQuestionsInput(BaseModel):
    """Input for strategic question suggestion tool"""
    questions: List[Dict[str, Any]] = Field(
        description=(
            "List of strategic questions. Each question should be a dictionary with: "
            "content (str), subtopic_id (str), strategy_type (str), "
            "priority (int 1-10), reasoning (str)"
        )
    )

    @field_validator('questions', mode='before')
    @classmethod
    def validate_questions(cls, v):
        """Validate questions structure"""
        if not isinstance(v, list):
            raise ValueError("questions must be a list")

        for i, q in enumerate(v):
            if not isinstance(q, dict):
                raise ValueError(f"Question {i} must be a dictionary")

            required_fields = ["content", "subtopic_id", "strategy_type", "priority", "reasoning"]
            for field in required_fields:
                if field not in q:
                    raise ValueError(f"Question {i} missing required field '{field}'")

            # Validate strategy type
            valid_types = ["coverage_gap", "emergent_insight"]
            if q["strategy_type"] not in valid_types:
                raise ValueError(
                    f"Question {i} strategy_type must be one of: {valid_types}"
                )

            # Validate priority range
            priority = q["priority"]
            if not isinstance(priority, int) or not (0 <= priority <= 10):
                raise ValueError(f"Question {i} priority must be between 0 and 10")

        return v


class SuggestStrategicQuestions(BaseTool):
    """
    Tool for suggesting strategic questions as guidance.

    Generates question suggestions optimized for:
    - Filling coverage gaps
    - Exploring emergent insights

    Questions are stored as suggestions for AgendaManager to consider,
    not directly added to the question bank.
    """
    name: str = "suggest_strategic_questions"
    description: str = (
        "Suggest strategic questions as guidance for AgendaManager. "
        "Questions are optimized for coverage, emergence, and utility. "
        "AgendaManager will consider these suggestions but won't blindly use them if already covered."
    )
    args_schema: Type[BaseModel] = SuggestStrategicQuestionsInput
    strategic_state: StrategicState = Field(...)
    session_agenda: SessionAgenda = Field(...)
    alpha: float = Field(...)
    gamma: float = Field(...)

    def _run(
        self,
        questions: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute strategic question suggestion"""
        try:
            # Store suggestions in strategic state (not question bank)
            self.strategic_state.strategic_question_suggestions = []

            # Get current coverage status from session agenda
            manager = self.session_agenda.interview_topic_manager
            covered_subtopics = set()

            for topic in manager.get_all_topics():
                for subtopic_id, subtopic in topic.required_subtopics.items():
                    if self.alpha > 0 and subtopic.is_covered:
                        covered_subtopics.add(subtopic_id)
                for subtopic_id, subtopic in topic.emergent_subtopics.items():
                    if self.gamma > 0 and subtopic.is_covered:
                        covered_subtopics.add(subtopic_id)

            filtered_count = 0
            for q_data in questions:
                subtopic_id = q_data["subtopic_id"]

                # Skip if subtopic is already covered
                if subtopic_id in covered_subtopics:
                    filtered_count += 1
                    continue

                suggestion = {
                    'content': q_data["content"],
                    'subtopic_id': subtopic_id,
                    'strategy_type': q_data["strategy_type"],
                    'priority': q_data["priority"],
                    'reasoning': q_data["reasoning"],
                }
                self.strategic_state.strategic_question_suggestions.append(suggestion)

            # Draft result msg
            result_msg = (
                f"Stored {len(self.strategic_state.strategic_question_suggestions)} strategic question suggestions. "
            )
            if filtered_count > 0:
                result_msg += f"Filtered out {filtered_count} suggestions for already-covered subtopics. "
            result_msg += "These will be included in guidance for AgendaManager to consider."

            return result_msg
        except Exception as e:
            raise ToolException(f"Error suggesting strategic questions: {e}")

class AddEmergentSubtopicInput(BaseModel):
    topic_id: str = Field(description="The topic ID under which this subtopic should be added")
    subtopic_description: str = Field(description="A brief description of the emergent subtopic")
    
class AddEmergentSubtopic(BaseTool):
    """Tool for adding emergent subtopics to the corresponding topic in session agenda."""
    name: str = "add_emergent_subtopic"
    description: str = (
        "A tool for adding emergent subtopics that arise during the interview. "
        "Use this when a new subtopic comes up that wasn't part of the original agenda in the list of subtopics of a topic and can gauge more emergent insights."
    )
    args_schema: Type[BaseModel] = AddEmergentSubtopicInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        topic_id: str,
        subtopic_description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            added = self.session_agenda.add_emergent_subtopic(
                topic_id=str(topic_id),
                subtopic_description=str(subtopic_description)
            )
            if added:
                return f"Successfully added emergent subtopic '{subtopic_description}'."
            else:
                return f"Emergent subtopic '{subtopic_description}' was not added (duplicate or invalid topic)."
        except Exception as e:
            raise ToolException(f"Error adding emergent subtopic: {e}")

class IdentifyEmergentInsightsInput(BaseModel):
    """Input for emergent insight identification tool"""

    emergent_insights: List[Dict[str, Any]] = Field(
        description=(
            "List of emergent insights. Each insight must be a dictionary with: "
            "subtopic_id (str), description (str), novelty_score from 1 (mildly unexpected) to 5 (highly counter-intuitive), "
            "evidence (str), conventional_belief (str)."
        )
    )

    @field_validator("emergent_insights", mode="before")
    @classmethod
    def validate_insights(cls, v):
        """Validate emergent insights structure"""
        if not isinstance(v, list):
            raise ValueError("emergent_insights must be a list")

        for i, insight in enumerate(v):
            if not isinstance(insight, dict):
                raise ValueError(f"Insight {i} must be a dictionary")

            required_fields = [
                "subtopic_id",
                "description",
                "novelty_score",
                "evidence",
                "conventional_belief",
            ]
            for field in required_fields:
                if field not in insight:
                    raise ValueError(f"Insight {i} missing required field '{field}'")

            # Validate novelty_score (1–5 integer)
            novelty = insight["novelty_score"]
            if not isinstance(novelty, int) or not (1 <= novelty <= 5):
                raise ValueError(
                    f"Insight {i} novelty_score must be an integer between 1 and 5"
                )

        return v

class IdentifyEmergentInsights(BaseTool):
    """
    Tool for identifying emergent insights (novel, counter-intuitive findings).

    Detects insights that:
    - Are unexpected or counter-intuitive
    - Contradict conventional wisdom
    - Are within topic scope but expand beyond existing subtopics
    """

    name: str = "identify_emergent_insights"
    description: str = (
        "Identify emergent insights that are novel and counter-intuitive. "
        "Only report insights with novelty_score >= min_novelty_score."
    )

    args_schema: Type[BaseModel] = IdentifyEmergentInsightsInput
    session_agenda: SessionAgenda = Field(...)
    min_novelty_score: int = Field(
        default=3,
        description="Minimum novelty threshold (1–5)"
    )

    def _run(
        self,
        emergent_insights: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute emergent insight identification"""
        added_count = 0

        for insight_data in emergent_insights:
            novelty_score = insight_data["novelty_score"]

            # Only add insights that meet the novelty threshold
            if novelty_score < self.min_novelty_score:
                continue

            # Construct insight payload
            insight = {
                "description": insight_data["description"],
                "novelty_score": novelty_score,
                "evidence": insight_data.get("evidence", ""),
                "conventional_belief": insight_data.get("conventional_belief", ""),
            }

            # Attach insight at the SUBTOPIC level
            self.session_agenda.add_emergent_insight(
                subtopic_id=insight_data["subtopic_id"],
                insight_data=insight,
            )

            added_count += 1

        return (
            f"Identified {added_count} emergent insight(s) "
            f"(minimum novelty_score: {self.min_novelty_score})"
        )
