"""
Core Topic Schema

This module defines the core topic that needs to be captured during each interview section.
Core topic represents milestones rather than literal questions to be asked.
Subtopic represents the sub-topic occuring within core topic that needs to be covered
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from src.content.question_bank.question import InterviewQuestion


class EmergentInsight(BaseModel):
    """
    Represents a novel, unexpected insight discovered during the interview.

    Emergent insights are:
    - Within a topic scope
    - Counter-intuitive or unexpected
    - Not captured by existing subtopics
    - Different from conventional wisdom

    Example:
        Expected: "AI helps with code completion"
        Emergent: "Developers treat AI as junior teammates, delegating
                   architectural decisions" (counter-intuitive shift)
    """
    subtopic_id: str
    description: str
    novelty_score: int
    evidence: str
    conventional_belief: str = ""

    def to_dict(self) -> dict:
        """Convert EmergentInsight object to dictionary."""
        return {
            'subtopic_id': self.subtopic_id,
            'description': self.description,
            'novelty_score': self.novelty_score,
            'evidence': self.evidence,
            'conventional_belief': self.conventional_belief,
        }

    @classmethod
    def from_dict(cls, emergent_insight_dict: dict) -> 'EmergentInsight':
        """Create SubTopic object from dictionary."""
        return cls(
            subtopic_id=emergent_insight_dict["subtopic_id"],
            description=emergent_insight_dict["description"],
            novelty_score=emergent_insight_dict["novelty_score"],
            evidence=emergent_insight_dict["evidence"],
            conventional_belief=emergent_insight_dict["conventional_belief"],
        )

class SubTopic(BaseModel):
    """Represents a subtopic of a topic to be collected."""
    subtopic_id: str
    core_topic_id: str
    description: str
    questions: List[InterviewQuestion] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    emergent_insights: List[EmergentInsight] = Field(default_factory=list)
    final_summary: str = "" # Essentially only available when subtopic is marked as covered
    feedback_gap: str = ""
    is_covered: bool = False
    
    def __iter__(self):
        """Iterate over questions in this subtopic."""
        return iter(self.questions)
    
    def __len__(self):
        """Return the number of questions."""
        return len(self.questions)

    def mark_covered(self):
        """Mark this subtopic as covered."""
        self.is_covered = True
        
    def reset_coverage(self):
        """Mark this subtopic as not covered."""
        self.is_covered = False
        
    def check_coverage(self):
        """Check if it's covered or not."""
        return self.is_covered
    
    def update_coverage_with_summary(self, aggregated_notes: str):
        """Update coverage and with summary."""
        self.mark_covered()
        self.final_summary = aggregated_notes
        
    def update_coverage_feedback_gap(self, feedback: str):
        """Update coverage feedback gap."""
        self.feedback_gap = feedback
        
    def get_final_summary(self) -> str:
        return self.final_summary
    
    def get_coverage_feedback_gap(self) -> str:
        """Update coverage feedback gap."""
        return self.feedback_gap
        
    def get_question(self, question_id: str) -> Optional[InterviewQuestion]:
        for q_item in self.questions:
            if q_item.question_id == question_id:
                return q_item
            
        return None
    
    def add_question(self, new_question: InterviewQuestion) -> bool:
        """Return True if success"""
        new_question_id = new_question.question_id
        if '.' not in new_question_id:
            # Top-level question
            self.questions.append(new_question)
            return True
        else:
            # Sub-question
            parent_id = new_question_id.split('.')[0]  
            current_index = new_question_id.split('.', 1)[1]  # e.g., "1.2.3" -> "2.3"
        
            parent_question = self.get_question(parent_id)
            if parent_question:
                parent_question.add_sub_question(current_index, new_question)
                return True
            else:
                return False
            
    def add_note(self, note: str) -> bool:
        """Return True if success"""
        self.notes.append(note)
        return True
    
    def add_emergent_insight(self, insight: EmergentInsight) -> bool:
        """Return True if success"""
        self.emergent_insights.append(insight)
        return True
            
    def reset(self):
        """Clears all questions."""
        self.questions = []
    
    def to_dict(self) -> dict:
        """Convert SubTopic object to dictionary."""
        return {
            'subtopic_id': self.subtopic_id,
            'core_topic_id': self.core_topic_id,
            'description': self.description,
            'questions': [q.to_dict() for q in self.questions],
            'notes': self.notes,
            'emergent_insights': [insight.to_dict() for insight in self.emergent_insights],
            'final_summary': self.final_summary,
            'is_covered': self.is_covered,
        }

    @classmethod
    def from_dict(cls, subtopic_dict: dict) -> 'SubTopic':
        """Create SubTopic object from dictionary."""
        return cls(
            subtopic_id=subtopic_dict['subtopic_id'],
            core_topic_id=subtopic_dict['core_topic_id'],
            description=subtopic_dict['description'],
            questions=[InterviewQuestion.from_dict(sub_q) for sub_q in subtopic_dict['questions']],
            notes=subtopic_dict['notes'],
            emergent_insights=[EmergentInsight.from_dict(insight) for insight in subtopic_dict['emergent_insights']],
            final_summary=subtopic_dict['final_summary'],
            is_covered=subtopic_dict['is_covered'],
        )

class CoreTopic(BaseModel):
    """Represents a core piece of topic to be collected."""
    topic_id: str
    description: str
    required_subtopics: Dict[str, SubTopic] = Field(default_factory=dict)
    emergent_subtopics: Dict[str, SubTopic] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    
    def __iter__(self):
        """Iterate over all subtopics (required first, then emergent)."""
        yield from self.required_subtopics.values()
        yield from self.emergent_subtopics.values()
    
    def __len__(self):
        """Return total number of subtopics."""
        return len(self.required_subtopics) + len(self.emergent_subtopics)
    
    def iter_required_subtopics(self):
        """Iterate over only required subtopics."""
        return iter(self.required_subtopics.values())
    
    def iter_emergent_subtopics(self):
        """Iterate over only emergent subtopics."""
        return iter(self.emergent_subtopics.values())
    
    def add_required_subtopic(self, subtopic: SubTopic) -> bool:
        if subtopic.subtopic_id in self.required_subtopics or subtopic.subtopic_id in self.emergent_subtopics:
            return False
        else:
            self.required_subtopics[subtopic.subtopic_id] = subtopic
            return True
    
    def add_emergent_subtopic(self, subtopic: SubTopic) -> bool:
        if subtopic.subtopic_id in self.required_subtopics or subtopic.subtopic_id in self.emergent_subtopics:
            return False
        else:
            self.emergent_subtopics[subtopic.subtopic_id] = subtopic
            return True
            
    def get_subtopic(self, subtopic_id: str):
        if subtopic_id in self.required_subtopics:
            return self.required_subtopics.get(subtopic_id)
        elif subtopic_id in self.emergent_subtopics:
            return self.emergent_subtopics.get(subtopic_id)
        else:
            return None
        
    def reset(self):
        """Clears all emergent subtopics and all questions in required subtopics."""
        self.emergent_subtopics = {}
        for subtopic in self.required_subtopics.values():
            subtopic.reset()
    
    def to_dict(self) -> dict:
        """Convert CoreTopic object to dictionary."""
        return {
            'topic_id': self.topic_id,
            'description': self.description,
            'required_subtopics': {k: v.to_dict() for k, v in self.required_subtopics.items()},
            'emergent_subtopics': {k: v.to_dict() for k, v in self.emergent_subtopics.items()},
            'keywords': self.keywords,
        }

    @classmethod
    def from_dict(cls, core_topic_dict: dict) -> 'CoreTopic':
        """Create CoreTopic object from dictionary."""
        required_subtopics = {k: SubTopic.from_dict(v) for k, v in core_topic_dict.get('required_subtopics', {}).items()}
        emergent_subtopics = {k: SubTopic.from_dict(v) for k, v in core_topic_dict.get('emergent_subtopics', {}).items()}
        return cls(
            topic_id=core_topic_dict['topic_id'],
            description=core_topic_dict['description'],
            required_subtopics=required_subtopics,
            emergent_subtopics=emergent_subtopics,
            keywords=core_topic_dict['keywords'],
        )
        
    @classmethod
    def get_topic_with_active_subtopics(cls, core_topic: 'CoreTopic', required_only: bool = False) -> 'CoreTopic':
        """Create new CoreTopic that only contains incomplete subtopics."""
        active_required_subtopics = {k: v for k, v in core_topic.required_subtopics.items() if not v.check_coverage()}
        if required_only:
            active_emergent_subtopics = {}
        else:
            active_emergent_subtopics = {k: v for k, v in core_topic.emergent_subtopics.items() if not v.check_coverage()}
        return cls(
            topic_id=core_topic.topic_id,
            description=core_topic.description,
            required_subtopics=active_required_subtopics,
            emergent_subtopics=active_emergent_subtopics,
            keywords=core_topic.keywords,
        )
        
    @classmethod
    def get_copy_of_core_topic(cls, core_topic: 'CoreTopic', required_only: bool = False) -> 'CoreTopic':
        """Create new CoreTopic that only contains incomplete subtopics."""
        active_required_subtopics = {k: v for k, v in core_topic.required_subtopics.items()}
        if required_only:
            active_emergent_subtopics = {}
        else:
            active_emergent_subtopics = {k: v for k, v in core_topic.emergent_subtopics.items()}
        return cls(
            topic_id=core_topic.topic_id,
            description=core_topic.description,
            required_subtopics=active_required_subtopics,
            emergent_subtopics=active_emergent_subtopics,
            keywords=core_topic.keywords,
        )
