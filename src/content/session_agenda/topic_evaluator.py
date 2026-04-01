"""
Topic Evaluator Factory Registry

A flexible registry-based factory system that supports both
built-in and user-defined evaluator strategies.
"""
import os
from typing import Callable, Dict, Optional
from src.content.session_agenda.core_topic import CoreTopic

class TopicEvaluator:
    """Base interface for topic completion evaluators."""
    def is_complete(self, core_topic: CoreTopic) -> bool:
        raise NotImplementedError

    def get_coverage_score(self, core_topic: CoreTopic) -> float:
        raise NotImplementedError
    
    def get_all_statistics(self, core_topic: CoreTopic) -> float:
        raise NotImplementedError

class MinimumThresholdSubtopicsEvaluator(TopicEvaluator):
    """Requires a minimum ratio of number of subtopics to be covered."""
    def __init__(self, minimum_threshold: float = 0.9, gamma: Optional[float] = None):
        self.minimum_threshold = minimum_threshold
        self.gamma = gamma

    def is_complete(self, core_topic: CoreTopic) -> bool:
        # All required subtopics are covered and the coverage score is above the threshold
        required_topic_coverage = all(st.is_covered for st in core_topic.required_subtopics.values())
        emergent_topic_coverage = all(st.is_covered for st in core_topic.emergent_subtopics.values()) 

        if self.gamma > 0:
            return (required_topic_coverage and emergent_topic_coverage) or self.get_coverage_score(core_topic) >= self.minimum_threshold
        else:
            return required_topic_coverage or self.get_coverage_score(core_topic) >= self.minimum_threshold

    def get_coverage_score(self, core_topic: CoreTopic) -> float:
        required_covered = sum(st.is_covered for st in core_topic.required_subtopics.values())
        emergence_covered = sum(st.is_covered for st in core_topic.emergent_subtopics.values())
        covered = required_covered + emergence_covered
        return covered / (len(core_topic.required_subtopics) + len(core_topic.emergent_subtopics))
    
    def get_all_statistics(self, core_topic: CoreTopic) -> float:
        curr_coverage_stats = {
            "total_required_subtopics_covered": sum(st.is_covered for st in core_topic.required_subtopics.values()),
            "total_required_subtopics": sum(1 for st in core_topic.required_subtopics.values()),
            "total_emergent_subtopics_covered": sum(st.is_covered for st in core_topic.emergent_subtopics.values()),
            "total_emergent_subtopics": sum(1 for st in core_topic.emergent_subtopics.values()),
            "current_coverage_score": self.get_coverage_score(core_topic)
        }
        return curr_coverage_stats

class EvaluatorFactoryRegistry:
    """Registry-based factory for TopicEvaluators."""

    def __init__(self):
        self._registry: Dict[str, Callable[..., TopicEvaluator]] = {}

    def register(self, name: str, factory_fn: Callable[..., TopicEvaluator]):
        """Register a new evaluator factory."""
        self._registry[name] = factory_fn

    def unregister(self, name: str):
        """Remove a registered evaluator."""
        if name in self._registry:
            del self._registry[name]

    def create(self, name: str, **kwargs) -> TopicEvaluator:
        """Instantiate an evaluator by name."""
        if name not in self._registry:
            raise KeyError(f"Evaluator '{name}' not registered")
        return self._registry[name](**kwargs)

    def list_evaluators(self):
        """Return all registered evaluator names."""
        return list(self._registry.keys())


# ===== Global Registry =====

_global_registry = EvaluatorFactoryRegistry()

# Register built-ins
_global_registry.register("minimum_threshold", lambda minimum_threshold=0.9: MinimumThresholdSubtopicsEvaluator(minimum_threshold, gamma=float(os.getenv("EXPLORATION_PLANNER_GAMMA", 0.0))))

def get_registry() -> EvaluatorFactoryRegistry:
    return _global_registry

def register_evaluator(name: str, factory_fn: Callable[..., TopicEvaluator]):
    """Register a UDF or custom evaluator globally."""
    _global_registry.register(name, factory_fn)