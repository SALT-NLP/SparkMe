"""
Interview Topic Manager

This module defines the InterviewTopicManager class, which tracks core topics with their completion status.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import faiss
import numpy as np
from pydantic import BaseModel, Field
from src.content.embeddings.embedding_service import EmbeddingService
from src.content.question_bank.question import InterviewQuestion
from src.content.session_agenda.core_topic import CoreTopic, SubTopic, EmergentInsight
from src.content.session_agenda.topic_evaluator import TopicEvaluator, MinimumThresholdSubtopicsEvaluator, get_registry

class InterviewTopicManager(BaseModel):
    """Tracks core topics with their completion status."""
    core_topic_dict: Dict[str, CoreTopic] = Field(default_factory=dict)
    interview_evaluator: TopicEvaluator = Field(default_factory=MinimumThresholdSubtopicsEvaluator)
    model_config = {
        "arbitrary_types_allowed": True
    }
    active_topic_id_list: List[str] = Field(default_factory=list)
    
    # Statistics only for recording
    coverage_stats: Dict[str, Any] = Field(default_factory=list)
    enable_emergent_subtopics: bool = False

    # Embedding-based dedup for emergent subtopics
    embedding_service: Optional[Any] = Field(default=None, exclude=True)
    subtopic_embeddings: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    similarity_threshold: float = 0.7
    
    def __iter__(self):
        """Iterate over all core topics."""
        return iter(self.core_topic_dict.values())
    
    def __len__(self):
        """Return the number of core topics."""
        return len(self.core_topic_dict)
    
    def __contains__(self, topic_id: str) -> bool:
        """
        Implements the 'in' operator to check if a topic_id
        exists in the core_topic_dict.
        """
        return topic_id in self.core_topic_dict
    
    def __str__(self) -> str:
        """Returns a tree visualization of topics and questions.
        
        Example output:
        Topics
        ├── General
        │   └── How old are you?
        ├── Professional
        │   ├── How did you choose your career path?
        │   └── What specific rare plant species did you cultivate?
        │       └── Did you face any challenges?
        └── Personal
            └── Where did you grow up?
        """
        # TODO: This is unverified

        lines = ["Topics"]        
        def add_question_prefix(question: InterviewQuestion, 
                         prefix: str, is_last: bool) -> None:
            # Add the current question
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{question.question}")
            
            # Handle sub-questions
            if question.sub_questions:
                new_prefix = prefix + ("    " if is_last else "│   ")
                sub_questions = question.sub_questions
                for i, sub_q in enumerate(sub_questions):
                    add_question_prefix(sub_q, new_prefix, i == len(sub_questions) - 1)
        
        # Process each topic
        for topic_idx, core_topic in enumerate(self.core_topic_dict.values()):
            # Add topic
            topic_prefix = "└── " if topic_idx == len(self.core_topic_dict) - 1 else "├── "
            lines.append(f"{topic_prefix}{core_topic.description}")
            
            for subtopic_idx, subtopic in enumerate(core_topic):
                # Add topic
                subtopic_prefix = "    └── " if subtopic_idx == len(core_topic) - 1 else "    ├── "
                lines.append(f"{subtopic_prefix}{subtopic.description}")
            
                # Process questions under this topic
                question_prefix = "        " if subtopic_idx == len(subtopic) - 1 else "│       "
                for q_idx, question in enumerate(subtopic):
                    add_question_prefix(question, question_prefix, q_idx == len(subtopic) - 1)
        
        return "\n".join(lines)
    
    @classmethod
    def init_from_interview_plan(cls, interview_plan: List[Dict[str, Any]] = [],
                                 interview_evaluator: Optional[str] = None):
        # Initialize InterviewTopicManager using the interview plan provided
        manager = cls()
        for i, topic_dict in enumerate(interview_plan):
            topic_id = str(i + 1)
            
            curr_core_topic = CoreTopic(
                topic_id=topic_id,
                description=topic_dict['topic'],
                required_subtopics={},
                emergent_subtopics={},
                keywords=[],
            )

            for j, subtopic in enumerate(topic_dict.get('subtopics', [])):
                subtopic_id = f"{topic_id}.{j + 1}"
                curr_subtopic = SubTopic(
                    subtopic_id=subtopic_id,
                    core_topic_id=topic_id,
                    description=subtopic,
                    questions=[],
                    is_covered=False
                )
                curr_core_topic.add_required_subtopic(curr_subtopic)

            manager.add_core_topic(curr_core_topic)
            
            if i == 0 or i == 1:
                manager.add_topic_id_as_active_topic(topic_id)
        
        # Register evaluator
        evaluator_registry = get_registry()
        if interview_evaluator:
            manager.interview_evaluator = evaluator_registry.create(interview_evaluator)
        else:
            manager.interview_evaluator = evaluator_registry.create("minimum_threshold")
            
        return manager
    
    def add_topic_id_as_active_topic(self, core_topic_id: str):
        if core_topic_id in self.core_topic_dict and core_topic_id not in self.active_topic_id_list:
            self.active_topic_id_list.append(core_topic_id)
    
    def add_core_topic(self, core_topic: CoreTopic):
        self.core_topic_dict[core_topic.topic_id] = core_topic
        
    def get_core_topic(self, core_topic_id: str) -> Optional[CoreTopic]:
        return self.core_topic_dict.get(core_topic_id, None)
    
    def _get_embedding_service(self) -> Optional[EmbeddingService]:
        """Get or lazily create the embedding service."""
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()
        return self.embedding_service

    def _ensure_subtopic_embeddings(self):
        """Compute and cache embeddings for any subtopics not yet in the cache."""
        service = self._get_embedding_service()
        if service is None or service.is_noop():
            return

        missing_ids = []
        missing_descriptions = []
        for core_topic in self.core_topic_dict.values():
            for subtopic in core_topic:
                if subtopic.subtopic_id not in self.subtopic_embeddings:
                    missing_ids.append(subtopic.subtopic_id)
                    missing_descriptions.append(subtopic.description)

        if missing_descriptions:
            embeddings = service.get_embeddings_batch(missing_descriptions)
            for sid, emb in zip(missing_ids, embeddings):
                self.subtopic_embeddings[sid] = emb

    def _check_duplicate_subtopic(self, description: str) -> Tuple[bool, float]:
        """Check if a subtopic description is too similar to any existing subtopic.

        Returns:
            Tuple of (is_duplicate, similarity_score).
        """
        service = self._get_embedding_service()
        if service is None or service.is_noop():
            return False, None, 0.0

        self._ensure_subtopic_embeddings()

        if not self.subtopic_embeddings:
            return False, None, 0.0

        # Build FAISS index from cached embeddings
        dim = service.get_embedding_dimension()
        index = faiss.IndexFlatL2(dim)

        id_list = list(self.subtopic_embeddings.keys())
        embedding_matrix = np.array(
            [self.subtopic_embeddings[sid] for sid in id_list]
        ).astype(np.float32)
        index.add(embedding_matrix)

        # Compute embedding for the new description
        new_embedding = service.get_embedding(description).reshape(1, -1).astype(np.float32)

        # Search for nearest neighbor
        distances, indices = index.search(new_embedding, 1)

        if indices[0][0] == -1:
            return False, None, 0.0

        # Convert L2 distance to similarity score (consistent with codebase pattern)
        similarity = float(1 / (1 + distances[0][0]))

        is_duplicate = similarity >= self.similarity_threshold
        return is_duplicate, similarity

    def add_emergent_subtopic(self, core_topic_id: str, new_subtopic_description: str) -> bool:
        core_topic = self.get_core_topic(core_topic_id)
        if core_topic is None:
            return False

        # Embedding-based deduplication check
        is_duplicate, similarity = self._check_duplicate_subtopic(new_subtopic_description)
        if is_duplicate:
            return False

        subtopic_list_length = len(core_topic)
        subtopic_id = f"{core_topic_id}.{subtopic_list_length + 1}"
        new_subtopic = SubTopic(
            subtopic_id=subtopic_id,
            core_topic_id=core_topic_id,
            description=new_subtopic_description,
            questions=[],
            is_covered=False
        )
        added = core_topic.add_emergent_subtopic(new_subtopic)

        # Cache embedding for the newly added subtopic
        if added:
            service = self._get_embedding_service()
            if service is not None and not service.is_noop():
                self.subtopic_embeddings[subtopic_id] = service.get_embedding(new_subtopic_description)

        return added

    def add_question(self, subtopic_id: str, new_question: InterviewQuestion) -> bool:
        """Add a question to a core topic then to a subtopic. Returns True is adding succeeded."""
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        return subtopic.add_question(new_question)
    
    def add_note_to_subtopic(self, subtopic_id: str, note: str) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        return subtopic.add_note(note)
    
    def get_question(self, topic_id: str, subtopic_id: str, question_id: str) -> Optional[InterviewQuestion]:
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return None
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return None
        
        return subtopic.get_question(question_id)
    
    def reset(self):
        """Clear all non-default elements inside CoreTopic and SubTopic."""
        for core_topic in self.core_topic_dict.values():
            core_topic.reset()
            
    def check_core_topic_completion(self, core_topic_id: str) -> bool:
        """Check the completion of a core topic."""
        core_topic = self.get_core_topic(core_topic_id)
        if core_topic is None:
            return False
        
        return self.interview_evaluator.is_complete(core_topic)
    
    def check_core_topic_score(self, core_topic_id: str) -> float:
        """Check the score of a core topic."""
        core_topic = self.get_core_topic(core_topic_id)
        if core_topic is None:
            return False
        
        return self.interview_evaluator.get_coverage_score(core_topic)
    
    def check_all_core_topic_completion(self) -> bool:
        """Check the completion of all core topics."""
        return all(self.check_core_topic_completion(topic_id) for topic_id in self.core_topic_dict.keys())
    
    def get_all_incomplete_core_topic(self) -> List[CoreTopic]:
        """Get all incomplete core topics."""
        core_topic_list = []
        for core_topic in self.core_topic_dict.values():
            if not self.check_core_topic_completion(core_topic.topic_id):
                core_topic_list.append(core_topic)
            
        return core_topic_list
    
    def use_emergent_subtopics(self):
        self.enable_emergent_subtopics = True
    
    def get_active_topics(self) -> List[CoreTopic]:
        """Get all current active topics."""
        if self.enable_emergent_subtopics:
            required_only = False
        else:
            required_only = True
        
        list_active_topics = []
        for topic_id in self.active_topic_id_list:
            copy_topic = CoreTopic.get_topic_with_active_subtopics(self.get_core_topic(topic_id), required_only=required_only)
            list_active_topics.append(copy_topic)
        return list_active_topics
    
    def get_all_topics(self) -> List[CoreTopic]:
        """Get all topics."""
        if self.enable_emergent_subtopics:
            required_only = False
        else:
            required_only = True
            
        list_core_topics = []
        for core_topic in self.core_topic_dict.values():
            copy_topic = CoreTopic.get_copy_of_core_topic(core_topic, required_only=required_only)
            list_core_topics.append(copy_topic)
            
        return list_core_topics
    
    def update_subtopic_coverage(self, subtopic_id: str, aggregated_notes: str) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        return subtopic.update_coverage_with_summary(aggregated_notes=aggregated_notes)
    
    def add_emergent_insight_subtopic(self, subtopic_id: str, insight_data: Dict[str, Any]) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        insight = EmergentInsight.from_dict(insight_data)
        return subtopic.add_emergent_insight(insight=insight)
    
    def give_feedback_subtopic_coverage(self, subtopic_id: str, feedback: str) -> bool:
        topic_id = subtopic_id.split(".")[0]
        core_topic = self.get_core_topic(topic_id)
        if core_topic is None:
            return False
        
        subtopic = core_topic.get_subtopic(subtopic_id)
        if subtopic is None:
            return False
        
        return subtopic.update_coverage_feedback_gap(feedback=feedback)
    
    def revise_agenda_after_update(self):
        # TODO basic rule, only 3 active topic at a time
        self.active_topic_id_list = []
        for topic_id, core_topic in self.core_topic_dict.items():
            if not self.check_core_topic_completion(topic_id):
                self.active_topic_id_list.append(topic_id)
            
            if len(self.active_topic_id_list) == 3:
                break
            
        # Stats gathering only
        curr_coverage_stats = {}
        for topic_id, core_topic in self.core_topic_dict.items():
            curr_coverage_stats[topic_id] = self.interview_evaluator.get_all_statistics(core_topic)
        self.coverage_stats = curr_coverage_stats

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'core_topic_dict': {k: v.to_dict() for k, v in self.core_topic_dict.items()},
            'active_topic_id_list': self.active_topic_id_list,
            'coverage_stats': self.coverage_stats, # Stats only
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InterviewTopicManager':
        """Create from dictionary."""
        # Assuming CoreTopic has a from_dict method, adjust as needed
        manager = cls()

        core_topic_dict_data = data.get("core_topic_dict", {})
        active_topic_id_list_data = data.get("active_topic_id_list", [])
        for topic_id, core_topic_data in core_topic_dict_data.items():
            # CoreTopic.from_dict should reconstruct its SubTopics and Questions
            core_topic = CoreTopic.from_dict(core_topic_data)
            manager.add_core_topic(core_topic)
            
        for topic_id in active_topic_id_list_data:
            manager.add_topic_id_as_active_topic(topic_id)

        return manager
