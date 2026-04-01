import os
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, ToolException
from typing import Type, Optional, Callable, Any
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field, SkipValidation

import json
import asyncio
# import asyncio
# from concurrent.futures import ThreadPoolExecutor


from src.content.question_bank.question_bank_base import QuestionBankBase, Rubric, Question
from src.content.session_agenda.session_agenda import SessionAgenda
from src.agents.report_team.models import FollowUpQuestion
from src.utils.llm.xml_formatter import parse_rubric_call
from src.utils.logger.session_logger import SessionLogger

# Create a global thread pool executor for background evaluation tasks
# _thread_pool = ThreadPoolExecutor(max_workers=4)

"""
Shared tools for updating the session agenda:
- Agenda manager
- Session coordinator
"""

# TODO NOTE ================== THIS IS DEPRECATED ==================
class AddInterviewQuestionInput(BaseModel):
    topic: str = Field(description="The core topic for the question")
    topic_id: str = Field(decsription="The core topic ID")
    question: str = Field(description="The actual question text")
    question_id: str = Field(description="The ID for the question (e.g., '1', '1.1', '2.3'). Max level is 4. NEVER include a level 5 question id like '1.1.1.1.1'.")
    parent_id: Optional[str] = Field(default=None, description="The ID of the parent question (e.g., '1', '2', etc.). No need to include if it is a top-level question.")
    parent_text: Optional[str] = Field(default=None, description="The text of the parent question. No need to include if it is a top-level question.")

# TODO NOTE ================== THIS IS DEPRECATED ==================
class AddInterviewQuestion(BaseTool):
    """Tool for adding new interview questions."""
    name: str = "add_interview_question"
    description: str = (
        "Adds a new interview question to the session agenda. "
    )
    args_schema: Type[BaseModel] = AddInterviewQuestionInput
    session_agenda: SessionAgenda = Field(...)
    historical_question_bank: QuestionBankBase = Field(...)
    proposed_question_bank: Optional[QuestionBankBase] = Field(default=None)
    proposer: str = Field(...)
    llm_engine: Any = Field(None, exclude=True)

    async def _run(
        self,
        subtopic_id: str,
        question_id: str,
        question: str,
        parent_id: Optional[str] = None,
        parent_text: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # TODO move this out; replace to other aspect?
            # if os.getenv("EVAL_MODE", "FALSE").lower() == "true":
            #     asyncio.create_task(
            #         self._evaluate_question_duplicate_async(question)
            #     )

            # TODO NOTE ================== THIS IS DEPRECATED ==================
            # 3. Add to proposed question bank (if it exists)
            if self.proposed_question_bank:
                self.proposed_question_bank.add_question(content=str(question).strip(), 
                                                         subtopic_id=subtopic_id,
                                                         rubric=None)

            # 4. Add the final question to the session agenda
            self.session_agenda.add_interview_question(
                subtopic_id=str(subtopic_id),
                question=str(question).strip(),
                question_id=str(question_id),
                rubric=None
            )

            message = f"Successfully added question {question_id}"
            return message
        except Exception as e:
            raise ToolException(f"Error adding interview question: {str(e)}")
        
    # async def _evaluate_question_duplicate_async(self, question: str):
    #     """Run question duplicate evaluation in background without blocking."""
    #     try:
    #         # Run CPU-bound operation in thread pool
    #         loop = asyncio.get_running_loop()
    #         await loop.run_in_executor(
    #             _thread_pool,
    #             lambda: self.historical_question_bank.evaluate_question_duplicate(
    #                 question, self.proposer
    #             )
    #         )
    #     except Exception as e:
    #         # Log error but don't propagate - this is a background task
    #         print(f"Background question evaluation error: {str(e)}")

"""
Shared tools for proposing follow-up questions by:
- Planner
- Section writer

* Note: This tool only proposes follow-up questions, it does not really add them to the session agenda, which is different from the add_interview_question tool.
"""

class ProposeFollowUpInput(BaseModel):
    content: str = Field(description="The question to ask")
    context: str = Field(description="Context explaining why this question is important")

class ProposeFollowUp(BaseTool):
    """Tool for adding follow-up questions."""
    name: str = "propose_follow_up"
    description: str = (
        "Add a follow-up question that needs to be asked to gather more information for the report. "
        "Include both the question and context explaining why this information is needed."
    )
    args_schema: Type[BaseModel] = ProposeFollowUpInput
    on_question_added: SkipValidation[Callable[[FollowUpQuestion], None]] = Field(
        description="Callback function to be called when a follow-up question is added"
    )

    def _run(
        self,
        content: str,
        context: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            question = FollowUpQuestion(
                content=content.strip(),
                context=context.strip()
            )
            self.on_question_added(question)
            return f"Successfully added follow-up question: {content}"
        except Exception as e:
            raise ToolException(f"Error adding follow-up question: {str(e)}")