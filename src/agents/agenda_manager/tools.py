from typing import Type, Optional, List, Callable, Dict, Any, Union
import json


from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, SkipValidation, field_validator

from src.content.memory_bank.memory_bank_base import MemoryBankBase, Memory
from src.content.session_agenda.session_agenda import SessionAgenda


class UpdateSessionNoteInput(BaseModel):
    subtopic_id: str = Field(
        description=(
            "The ID of the subtopic"
        )
    )
    note: str = Field(
        description="A concise note to be added to the question, or as an additional note.")


class UpdateSessionNote(BaseTool):
    """Tool for updating the session agenda."""
    name: str = "update_session_agenda"
    description: str = "A tool for updating the session agenda."
    args_schema: Type[BaseModel] = UpdateSessionNoteInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        question_id: str,
        note: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        self.session_agenda.add_note(question_id=str(question_id), note=note)
        target_question = question_id if question_id else "additional note"
        return f"Successfully added the note for `{target_question}`."


class UpdateMemoryBankAndSessionInput(BaseModel):
    title: str = Field(description="A concise but descriptive title for the memory")
    text: str = Field(description="A clear summary of the information")
    subtopic_links: Union[str, List[Dict[str, Any]]] = Field(
        description=(
            "List of subtopics this memory relates to. "
            "Format: (a list of JSON dictionary) where "
            "each entry must contain: subtopic_id (from topics_list), "
            "importance (1-10 for that subtopic), and relevance (explanation). "
            "Example: '[{\"subtopic_id\": \"the id\", \"importance\": 1-10, \"relevance\": \"why it matters\"}, ...]'"
        )
    )
    metadata: Optional[dict] = Field(description=(
        "Additional metadata about the memory. "
        "Format: A valid JSON dictionary."
        "This can include topics, people mentioned, emotions, locations, dates, relationships, life events, achievements, goals, aspirations, beliefs, values, preferences, hobbies, interests, education, work experience, skills, challenges, fears, dreams, etc. "
        ),
        default={}
    )
    
    @field_validator('subtopic_links', mode='before')
    @classmethod
    def parse_subtopic_links(cls, v):
        """Parse subtopic_links from string (JSON array or NDJSON) to list."""
        # If it's a string, parse it
        if isinstance(v, str):
            v = v.strip()
            # Remove markdown code blocks
            v = v.removeprefix('```json').removeprefix('```').removesuffix('```').strip()

            # Try parsing as JSON array first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    v = parsed
                elif isinstance(parsed, dict):
                    # Single object, wrap in list
                    v = [parsed]
                else:
                    raise ValueError(f"Expected list or dict, got {type(parsed).__name__}")
            except json.JSONDecodeError:
                # Failed as JSON array, try parsing multiple JSON objects
                # LLM can output various formats:
                # 1. Newline-separated: {...}\n{...}
                # 2. Space-separated: {...} {...}
                # 3. Mixed: {...}\n{...} {...}

                v_list = []
                # Use a simple approach: find all {...} patterns
                # Match JSON objects (simple heuristic: balanced braces)
                current_obj = ""
                brace_count = 0

                for char in v:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1

                    current_obj += char

                    # When braces are balanced and we have content, parse it
                    if brace_count == 0 and current_obj.strip():
                        try:
                            obj = json.loads(current_obj.strip())
                            v_list.append(obj)
                            current_obj = ""
                        except json.JSONDecodeError:
                            # Skip invalid JSON, continue accumulating
                            pass

                if not v_list:
                    raise ValueError(f"Could not parse any valid JSON objects from: {v[:200]}...")

                v = v_list

        if isinstance(v, dict):
            v = [v]

        # Validate it's a list
        if not isinstance(v, list):
            raise ValueError(f"subtopic_links must be a list, got {type(v).__name__}")

        if len(v) == 0:
            raise ValueError("subtopic_links must contain at least one link")

        for i, link in enumerate(v):
            if not isinstance(link, dict):
                raise ValueError(f"Link {i} must be a dictionary, got {type(link).__name__}")

            # Required fields
            for field in ("subtopic_id", "importance", "relevance"):
                if field not in link:
                    raise ValueError(f"Link {i} missing required field '{field}'")

            # Validate importance range
            importance = link["importance"]
            if not isinstance(importance, (int, float)) or not (1 <= importance <= 10):
                raise ValueError(
                    f"Link {i} importance must be between 1-10, got {importance}"
                )

        return v

class UpdateMemoryBankAndSession(BaseTool):
    """Tool for updating the memory bank and session agenda."""
    name: str = "update_memory_bank_and_session"
    description: str = "A tool for storing new memories in the memory bank and updating the session agenda."
    args_schema: Type[BaseModel] = UpdateMemoryBankAndSessionInput
    memory_bank: MemoryBankBase = Field(...)
    session_agenda: SessionAgenda = Field(...)
    on_memory_added: SkipValidation[Callable[[Memory], None]] = Field(...)
    update_memory_map: SkipValidation[Callable[[str, str], None]] = Field(
        description="Callback function to update the memory ID mapping"
    )
    get_current_qa: SkipValidation[Callable[[], str]] = Field(
        description="Function to get the current interviewer's question and user's response"
    )

    def _run(
        self,
        title: str,
        text: str,
        subtopic_links: Union[str, List[Dict[str, Any]]],
        metadata: Optional[dict] = {},
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # Parse subtopic_links using the same logic as the validator
            # (LangChain bypasses the validator and passes raw strings to _run)
            subtopic_links = UpdateMemoryBankAndSessionInput.parse_subtopic_links(subtopic_links)

            # Ensure metadata is a valid dict, default to empty if not
            if not isinstance(metadata, dict):
                metadata = {}

            # First add memory to memory bank
            question_text, response_text = self.get_current_qa()
            memory = self.memory_bank.add_memory(
                title=title,
                text=text,
                subtopic_links=subtopic_links,
                metadata=metadata,
                source_interview_question=question_text,
                source_interview_response=response_text,
            )
            
            # # Use callback to update the mapping
            # self.update_memory_map(memory.id)
            
            # Trigger callback to track newly added memory
            self.on_memory_added(memory)
            
            # Now, add notes to session agenda
            for item_link in subtopic_links:
                self.session_agenda.add_note(subtopic_id=item_link['subtopic_id'], note=text)

            return f"Successfully stored memory and note for: {title}"
        except Exception as e:
            raise ToolException(f"Error storing memory: {e}")


class UpdateSubtopicCoverageInput(BaseModel):
    subtopic_id: str = Field(
        description="The unique ID of the subtopic to mark as covered (must exist in topics_list). Example: '1.1'."
    )
    aggregated_notes: str = Field(
        description="Final synthesis of the discussion or notes for this subtopic."
    )
class UpdateSubtopicCoverage(BaseTool):
    """Tool for updating the subtopics coverage."""
    name: str = "update_subtopic_coverage"
    description: str = "A tool for updating the coverage of subtopics along with the summary."
    args_schema: Type[BaseModel] = UpdateSubtopicCoverageInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        subtopic_id: str,
        aggregated_notes: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # Ensure metadata is a valid dict, default to empty if not
            self.session_agenda.update_subtopic_coverage(subtopic_id=str(subtopic_id),
                                                         aggregated_notes=str(aggregated_notes))
                
            return f"Successfully updated the coverage for subtopic ID: {subtopic_id}"
        except Exception as e:
            raise ToolException(f"Error updating subtopic coverage: {e}")
        
        
class FeedbackSubtopicCoverageInput(BaseModel):
    subtopic_id: str = Field(
        description="The unique ID of the subtopic that is not yet fully covered (must exist in topics_list). Example: '1.1'."
    )
    feedback: str = Field(
        description="Concise feedback of the missing elements or reasoning gaps for current subtopic."
    )
class FeedbackSubtopicCoverage(BaseTool):
    """Tool for giving feedback regarding the subtopic's coverage."""
    name: str = "feedback_subtopic_coverage"
    description: str = "A tool for providing feedback about the coverage of subtopic."
    args_schema: Type[BaseModel] = FeedbackSubtopicCoverageInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        subtopic_id: str,
        feedback: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # Ensure metadata is a valid dict, default to empty if not
            self.session_agenda.give_feedback_subtopic_coverage(subtopic_id=str(subtopic_id),
                                                         feedback=str(feedback))
                
            return f"Successfully provide feedback regarding the coverage for subtopic ID: {subtopic_id}"
        except Exception as e:
            raise ToolException(f"Error providing feedback for subtopic coverage: {e}")


class UpdateSubtopicNotesInput(BaseModel):
    subtopic_id: str = Field(
        description="The unique ID of the subtopic to be associated with the notes."
    )
    note_list: List[str] = Field(
        description="List of notes taken."
    )
class UpdateSubtopicNotes(BaseTool):
    """Tool for updating the subtopics note."""
    name: str = "update_subtopic_notes"
    description: str = "A tool for updating the notes of subtopics."
    args_schema: Type[BaseModel] = UpdateSubtopicNotesInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        subtopic_id: str,
        note_list: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            for note in note_list:
                self.session_agenda.add_note(str(subtopic_id), note=note)
                
            return f"Successfully updated the coverage for subtopic ID: {subtopic_id}"
        except Exception as e:
            raise ToolException(f"Error updating subtopic coverage: {e}")