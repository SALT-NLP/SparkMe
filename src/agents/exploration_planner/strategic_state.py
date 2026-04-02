import json
import os
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, Field


class ConversationRollout(BaseModel):
    """
    Predicted conversation trajectory with utility scoring.

    Represents a simulated conversation path that predicts:
    - Which topics will be discussed
    - Expected coverage improvement
    - Likelihood of discovering emergent insights
    - Cost (number of turns)
    - Overall utility score (U = α·Coverage - β·Cost + γ·Emergence)
    """

    rollout_id: str = Field(default_factory=lambda: f"rollout_{uuid4().hex[:8]}")
    predicted_turns: List[Dict[str, Any]] = Field(default_factory=list)
    """
    Each turn contains:
    {
        "turn": int,
        "topic_id": str,
        "subtopic_id": str,
        "question_type": str,  # "follow_up", "new_topic", "clarification"
        "likelihood": float,   # 0-1
        "coverage_impact": float  # Expected coverage contribution
    }
    """

    expected_coverage_delta: float = 0.0
    """Expected coverage improvement from this path (0-1)"""

    emergence_potential: float = 0.0
    """Likelihood of discovering novel insights on this path (0-1)"""

    cost_estimate: int = 0
    """Number of turns needed for this path"""

    utility_score: float = 0.0
    """
    Utility score: U = α·Coverage - β·Cost + γ·Emergence
    Higher scores indicate more valuable conversation paths
    """

    created_at: datetime = Field(default_factory=datetime.now)

class StrategicState(BaseModel):
    """
    Persisted strategic planning state for a session.
    """

    session_id: int
    last_planning_turn: int = 0
    """Last conversation turn when strategic planning was executed"""

    rollout_predictions: List[ConversationRollout] = Field(default_factory=list)
    """Predicted conversation trajectories, ranked by utility score (highest first)"""

    strategic_question_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    """
    Suggested strategic questions for AgendaManager to consider.
    Each suggestion contains:
    - content: Question text
    - subtopic_id: Target subtopic
    - strategy_type: "coverage_gap" or "emergent_insight"
    - priority: Priority score (1-10)
    - reasoning: Why this question is strategic
    """

    def save_to_file(self, user_id: str) -> None:
        """
        Save strategic state to session directory.

        Args:
            user_id: User ID for directory path
        """
        file_path = self._get_file_path(user_id)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(
                self.model_dump(mode='json'),
                f,
                indent=2,
                default=str  # Handle datetime serialization
            )

    @classmethod
    def load_from_file(cls, user_id: str, session_id: int) -> 'StrategicState':
        """
        Load strategic state from session directory.

        Args:
            user_id: User ID for directory path
            session_id: Session ID

        Returns:
            StrategicState instance (new if file doesn't exist)
        """
        file_path = cls._get_file_path_static(user_id, session_id)

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        else:
            # Initialize new state
            return cls(session_id=session_id, last_planning_turn=0)

    def save_snapshot(self, user_id: str, turn_number: int) -> None:
        """
        Save strategic state snapshot for debugging.

        Similar to SessionAgenda.save(save_type="snapshot").
        Saves as strategic_state_turn_X.json in session directory.

        Args:
            user_id: User ID for directory path
            turn_number: Current conversation turn
        """
        logs_dir = os.getenv("LOGS_DIR", "logs")
        session_dir = os.path.join(
            logs_dir,
            user_id,
            "execution_logs",
            f"session_{self.session_id}"
        )

        # Create directory if needed
        os.makedirs(session_dir, exist_ok=True)

        # Save with turn number in filename
        file_path = os.path.join(
            session_dir,
            f"strategic_state_turn_{turn_number}.json"
        )

        with open(file_path, 'w') as f:
            json.dump(
                self.model_dump(mode='json'),
                f,
                indent=2,
                default=str
            )

    def _get_file_path(self, user_id: str) -> str:
        """Get file path for this strategic state"""
        return self._get_file_path_static(user_id, self.session_id)

    @staticmethod
    def _get_file_path_static(user_id: str, session_id: int) -> str:
        """Get file path for strategic state"""
        logs_dir = os.getenv("LOGS_DIR", "logs")
        return os.path.join(
            logs_dir,
            user_id,
            "execution_logs",
            f"session_{session_id}",
            "strategic_state.json"
        )
