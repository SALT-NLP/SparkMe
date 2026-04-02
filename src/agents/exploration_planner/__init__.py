"""
Exploration Planner Agent for Long-Term Interview Planning

This module provides strategic planning capabilities that complement the
AgendaManager's short-term reactive planning with predictive, goal-oriented
planning.

Key Features:
- Strategic coverage analysis across all topics
- Conversation rollout prediction with utility scoring
- Novel emergence detection (counter-intuitive insights)
- Strategic question generation optimized for U = α·Coverage - β·Cost + γ·Emergence
"""

from .exploration_planner import ExplorationPlanner, ExplorationPlannerConfig
from .strategic_state import (
    ConversationRollout,
    StrategicState,
)

__all__ = [
    "ExplorationPlanner",
    "ExplorationPlannerConfig",
    "ConversationRollout",
    "StrategicState",
]
