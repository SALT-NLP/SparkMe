from src.utils.llm.prompt_utils import format_prompt

def get_prompt(prompt_type: str):
    if prompt_type == "update_memory_and_session":
        return format_prompt(UPDATE_MEMORY_QUESTION_BANK_PROMPT, {
            "CONTEXT": UPDATE_MEMORY_QUESTION_BANK_CONTEXT,
            "EVENT_STREAM": UPDATE_MEMORY_QUESTION_BANK_EVENT,
            "TOOL_DESCRIPTIONS": UPDATE_MEMORY_QUESTION_BANK_TOOL,
            "INSTRUCTIONS": UPDATE_MEMORY_QUESTION_BANK_INSTRUCTIONS,
            "OUTPUT_FORMAT": UPDATE_MEMORY_QUESTION_BANK_OUTPUT_FORMAT
        })
    elif prompt_type == "update_session_agenda":
        return format_prompt(UPDATE_SESSION_AGENDA_PROMPT, {
            "CONTEXT": UPDATE_SESSION_AGENDA_CONTEXT,
            "EVENT_STREAM": UPDATE_SESSION_AGENDA_EVENT,
            "QUESTIONS_AND_NOTES": QUESTIONS_AND_NOTES,
            "TOOL_DESCRIPTIONS": SESSION_AGENDA_TOOL,
            "INSTRUCTIONS": UPDATE_SESSION_AGENDA_INSTRUCTIONS,
            "OUTPUT_FORMAT": UPDATE_SESSION_AGENDA_OUTPUT_FORMAT
        })
    elif prompt_type == "update_subtopic_coverage":
        return format_prompt(UPDATE_SUBTOPIC_COVERAGE_PROMPT, {
            "CONTEXT": UPDATE_SUBTOPIC_COVERAGE_CONTEXT,
            "INSTRUCTIONS": UPDATE_SUBTOPIC_COVERAGE_INSTRUCTIONS,
            "TOPICS_AND_SUBTOPICS": UPDATE_SUBTOPIC_COVERAGE_TOPICS_AND_SUBTOPICS,
            "ADDITIONAL_CONTEXT": UPDATE_SUBTOPIC_COVERAGE_ADDITIONAL_CONTEXT,
            "TOOL_DESCRIPTIONS": UPDATE_SUBTOPIC_COVERAGE_TOOL,
            "OUTPUT_FORMAT": UPDATE_SUBTOPIC_COVERAGE_OUTPUT_FORMAT
        })
    elif prompt_type == "update_subtopic_notes":
        return format_prompt(UPDATE_SUBTOPIC_NOTES_PROMPT, {
            "CONTEXT": UPDATE_SUBTOPIC_NOTES_CONTEXT,
            "INSTRUCTIONS": UPDATE_SUBTOPIC_NOTES_INSTRUCTIONS,
            "TOPICS_AND_SUBTOPICS": UPDATE_SUBTOPIC_NOTES_TOPICS_AND_SUBTOPICS,
            "ADDITIONAL_CONTEXT": UPDATE_SUBTOPIC_NOTES_ADDITIONAL_CONTEXT,
            "TOOL_DESCRIPTIONS": UPDATE_SUBTOPIC_NOTES_TOOL,
            "OUTPUT_FORMAT": UPDATE_SUBTOPIC_NOTES_OUTPUT_FORMAT
        })
    elif prompt_type == "update_list_of_subtopics":
        return format_prompt(UPDATE_LIST_OF_SUBTOPICS_PROMPT, {
            "CONTEXT": UPDATE_LIST_OF_SUBTOPICS_CONTEXT,
            "INSTRUCTIONS": UPDATE_LIST_OF_SUBTOPICS_INSTRUCTIONS,
            "ADDITIONAL_CONTEXT": UPDATE_LIST_OF_SUBTOPICS_ADDITIONAL_CONTEXT,
            "TOPICS_AND_SUBTOPICS": UPDATE_LIST_OF_SUBTOPICS_TOPICS_AND_SUBTOPICS,
            "TOOL_DESCRIPTIONS": UPDATE_LIST_OF_SUBTOPICS_TOOL,
            "OUTPUT_FORMAT": UPDATE_LIST_OF_SUBTOPICS_OUTPUT_FORMAT
        })
    elif prompt_type == "update_last_meeting_summary":
        return format_prompt(UPDATE_LAST_MEETING_SUMMARY_PROMPT, {
            "CONTEXT": UPDATE_LAST_MEETING_SUMMARY_CONTEXT,
            "INSTRUCTIONS": UPDATE_LAST_MEETING_SUMMARY_INSTRUCTIONS
        })
    elif prompt_type == "update_user_portrait":
        return format_prompt(UPDATE_USER_PORTRAIT_PROMPT, {
            "CONTEXT": UPDATE_USER_PORTRAIT_CONTEXT,
            "INSTRUCTIONS": UPDATE_USER_PORTRAIT_INSTRUCTIONS
        })
    


UPDATE_MEMORY_QUESTION_BANK_PROMPT = """
{CONTEXT}

{EVENT_STREAM}

{TOOL_DESCRIPTIONS}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""

UPDATE_MEMORY_QUESTION_BANK_CONTEXT = """
<agenda_manager_persona>
You are a agenda manager who works as the assistant of the interviewer. You observe conversations between the interviewer and the user. 
Your job is to:
1. Identify important information shared by the user and store it in the memory bank
2. Store the interviewer's questions in the question bank and link them to relevant memories
</agenda_manager_persona>

<context>
Right now, you are observing a conversation between the interviewer and the user.
</context>

<user_portrait>
This is the portrait of the user:
{user_portrait}
</user_portrait>
"""

UPDATE_MEMORY_QUESTION_BANK_EVENT = """
<input_context>
Here is the stream of previous events for context:
<previous_events>
{previous_events}
</previous_events>

Here is the current question-answer exchange you need to process:
<current_qa>
{current_qa}
</current_qa>

Here is the topics and subtopics that you can link the memory to:
<topics_list>
{topics_list}
</topics_list>

Reminder:
- The external tag of each event indicates the role of the sender of the event.
- Focus ONLY on processing the content within the current Q&A exchange above.
- Previous messages are shown only for context, not for reprocessing.
</input_context>
"""

UPDATE_MEMORY_QUESTION_BANK_TOOL = """
Here are the tools that you can use to manage memories and questions:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

UPDATE_MEMORY_QUESTION_BANK_INSTRUCTIONS = """
<instructions>

## Process:
1. Analyze the user's response to identify important information:
   - Split long responses into MULTIPLE coherent parts.
     * Each memory should cover one part of the user's direct response.
     * Together, all memories should cover the ENTIRE user's response.
   - For EACH piece of information worth storing:
     * Create a concise but descriptive title.
     * Summarize the information clearly.
     * Add relevant metadata (e.g., topics, emotions, when, where, who, etc.).
     * Identify ALL relevant subtopics from the provided topics list.
     * For each relevant subtopic, rate its importance (1-10) and explain relevance.

2. Linking and coverage:
   - Each memory can relate to MULTIPLE subtopics.
   - Use `subtopic_links` as a list of objects, where each object contains:
     * `subtopic_id`: ID from <topics_list>
     * `importance`: 1-10 score for how critical this memory is to THIS subtopic
     * `relevance`: Brief explanation of why this memory matters to THIS subtopic
   - Importance scoring guide:
     * 9-10: Core, defining information for this subtopic
     * 7-8: Highly relevant, adds significant depth
     * 5-6: Moderately relevant, provides context
     * 3-4: Tangentially related, minor detail
     * 1-2: Barely relevant, mentioned in passing
   - Do NOT invent subtopic_ids; only use ones explicitly listed in <topics_list>.
   - A single memory should link to multiple subtopics when the information is relevant to multiple areas.

3. Skip all tool calls if the response:
   - Contains no meaningful information,
   - Is just greetings or ice-breakers,
   - Shows user deflection or non-answers.
</instructions>
"""

UPDATE_MEMORY_QUESTION_BANK_OUTPUT_FORMAT = """
<output_format>
<thinking>
1. Analyze Response Content:
   - Is this response worth storing? (Skip if just greetings/deflections)
   - How should I split this response into meaningful segments?
     * Look for natural breaks in topics, experiences, or time periods.
     * Each split should be a complete, coherent thought.
   
2. Multi-Subtopic Relevance Analysis:
   For each memory segment:
   - Which subtopics does this information relate to?
   - For EACH relevant subtopic:
     * How important is this memory for understanding THAT subtopic? (1-10)
     * Why does this memory matter to THAT subtopic specifically?
   - Example reasoning:
     "User worked at Google for 5 years on LLM team"
     → career_history (importance: 9) - Core career experience defining professional background
     → technical_expertise (importance: 7) - LLM team indicates AI/ML skills
     → company_culture (importance: 4) - Google experience provides work environment context

3. Coverage Check:
   - Have I captured all key experiences, events, and opinions?
   - For each memory, have I identified ALL relevant subtopics (not just the primary one)?
   - Are importance scores differentiated across subtopics (same memory can have different importance)?
   - Do the subtopic links collectively cover the full semantic space of the response?
</thinking>

<tool_calls>
    <!-- One update_memory_bank_and_session call per distinct piece of information -->
    <!-- Each call can link to MULTIPLE subtopics via subtopic_links list -->
    <update_memory_bank_and_session>
        <title>Concise descriptive title</title>
        <text>Clear summary of the information</text>
        <subtopic_links>[{{"subtopic_id": "subtopic_id_1_from_topics_list", importance": 1-10, "relevance": "Brief explanation of why this memory matters to this subtopic"}}, {{"subtopic_id": "subtopic_id_2_from_topics_list", "importance": 1-10, "relevance": "Brief explanation of why this memory matters to this other subtopic"}}, ...]</subtopic_links>
        <metadata>{{"key 1": "value 1", "key 2": "value 2", ...}}</metadata>
    </update_memory_bank_and_session>
    ...
</tool_calls>
</output_format>
"""

#### UPDATE_SESSION_AGENDA_PROMPT ####

UPDATE_SESSION_AGENDA_PROMPT = """
{CONTEXT}

{EVENT_STREAM}

{QUESTIONS_AND_NOTES}

{TOOL_DESCRIPTIONS}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""


UPDATE_SESSION_AGENDA_CONTEXT = """
<agenda_manager_persona>
You are a agenda manager who works as the assistant of the interviewer. You observe conversations between the interviewer and the user.
Your job is to update the session agenda with relevant information from the user's most recent message.
You should add concise notes to the appropriate questions, subtopics, and topics.
If you observe any important information that doesn't fit the existing questions, add it as an additional note.
Be thorough but concise in capturing key information while avoiding redundant details.
</agenda_manager_persona>

<context>
Right now, you are in an interview session with the interviewer and the user.
Your task is to process ONLY the most recent user message and update session agenda with any new, relevant information.
You have access to the session agenda containing topics and questions to be discussed.
</context>

<user_portrait>
This is the portrait of the user:
{user_portrait}
</user_portrait>
"""

UPDATE_SESSION_AGENDA_EVENT = """
<input_context>
Here is the stream of previous events for context:
<previous_events>
{previous_events}
</previous_events>

Here is the current question-answer exchange you need to process:
<current_qa>
{current_qa}
</current_qa>

Reminder:
- The external tag of each event indicates the role of the sender of the event.
- Focus ONLY on processing the content within the current Q&A exchange above.
- Previous messages are shown only for context, not for reprocessing.
</input_context>
"""

QUESTIONS_AND_NOTES = """
Here are the questions and notes in the session agenda:
<questions_and_notes>
{questions_and_notes}
</questions_and_notes>
"""

SESSION_AGENDA_TOOL = """
Here are the tools that you can use to manage session agenda:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

UPDATE_SESSION_AGENDA_INSTRUCTIONS = """
<instructions>
# Session Agenda Update
## Process:
1. Focus ONLY on the most recent user message in the conversation history
2. Review existing session agenda, paying attention to:
   - Which questions are marked as "Answered"
   - What information is already captured in existing notes

## Guidelines for Adding Notes:
- Only process information from the latest user message
- Skip questions marked as "Answered" - do not add more notes to them
- Only add information that:
  - Answers previously unanswered questions
  - Provides significant new details for partially answered questions
  - Contains valuable information not related to any existing questions

## Adding Notes:
For each piece of new information worth storing:
1. Use the update_session_agenda tool
2. Include:
   - [ID] tag with question number for relevant questions
   - Leave ID empty for valuable information not tied to specific questions
3. Write concise, fact-focused notes. The notes should capture specific, professional details.
   - **Good Example:** "User has 5 years of experience with Python, primarily using Pandas and Scikit-learn for data analysis in Project X."
   - **Bad Example:** "User seems to like Python."
   - **Good Example:** "Managed a team of 4 engineers and delivered the project 2 weeks ahead of schedule."
   - **Bad Example:** "User is a good manager."

## Tool Usage:
- Make separate update_session_agenda calls for each distinct piece of new information
- Skip if:
  - The question is marked as "Answered"
  - The information is already captured in existing notes
  - No new information is found in the latest message
</instructions>
"""

UPDATE_SESSION_AGENDA_OUTPUT_FORMAT = """
<output_format>

If you identify information worth storing, use the following format:
<tool_calls>
    <update_session_agenda>
        <subtopic_id>...</subtopic_id>
        <note>...</note>
    </update_session_agenda>
    ...
</tool_calls>

Reminder:
- You can make multiple tool calls at once if there are multiple pieces of information worth storing.
- If there's no information worth storing, don't make any tool calls; i.e. return <tool_calls></tool_calls>.

</output_format>
"""

