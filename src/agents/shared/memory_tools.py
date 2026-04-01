from typing import List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException


from src.content.memory_bank.memory import MemorySearchResult
from src.content.memory_bank.memory_bank_base import MemoryBankBase


"""
Shared tools to access the memory bank by:
- Interviewer
- Report section writer
- Session coordinator
- Agenda manager
"""
class RecallInput(BaseModel):
    reasoning: str = Field(
        description="Explain:\n"
        "1. What information you're looking for\n"
        "2. How this search will help your evaluation\n"
        "3. What decisions this search will inform"
    )
    query: str = Field(
        description="The search query to find relevant information. Make it broad enough to cover related topics."
    )

class Recall(BaseTool):
    """Tool for searching relevant memories."""
    name: str = "recall"
    description: str = "Search for relevant memories in all historical memories"
    args_schema: Type[BaseModel] = RecallInput
    memory_bank: MemoryBankBase = Field(default=None)

    def _run(self, query: str, reasoning: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            if self.memory_bank is None:
                raise ToolException("No memory bank available")

            memories: List[MemorySearchResult] = self.memory_bank.search_memories(query)
            memories_str = "\n".join([memory.to_xml(include_source=True) for memory in memories])
            return f"""\
<memory_search>
<query>{query}</query>
<reasoning>{reasoning}</reasoning>
<results>
{memories_str if memories_str else "No relevant memories found."}
</results>
</memory_search>"""
        except Exception as e:
            raise ToolException(f"Error searching memories: {e}")
