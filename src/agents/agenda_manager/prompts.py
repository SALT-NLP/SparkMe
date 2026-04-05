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

UPDATE_SUBTOPIC_COVERAGE_PROMPT = """
{CONTEXT}

{TOPICS_AND_SUBTOPICS}

{ADDITIONAL_CONTEXT}

{TOOL_DESCRIPTIONS}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""

UPDATE_SUBTOPIC_COVERAGE_CONTEXT = """
<agenda_manager_persona>
You are a agenda manager who assists an interviewer. You observe the dialogue between the interviewer and the candidate, and your role is to determine investigate each subtopic and its notes to determine whether the subtopic has achieved full coverage or not.

Your objectives:
1. Infer whether each subtopic is best evaluated using the STAR (Situation, Task, Action, Result) framework or a general descriptive evaluation.
2. If the subtopic is complete, and mark the subtopic as covered and aggregate the subtopic's notes succinctly and faithfully.
</agenda_manager_persona>
"""

UPDATE_SUBTOPIC_COVERAGE_TOPICS_AND_SUBTOPICS = """
Here are the topics and subtopics to review:
<topics_list>
{topics_list}
</topics_list>
"""

UPDATE_SUBTOPIC_COVERAGE_ADDITIONAL_CONTEXT = """
Here is last meeting summary that might be helpful:
<last_meeting_summary>
{last_meeting_summary}
</last_meeting_summary>
"""

UPDATE_SUBTOPIC_COVERAGE_TOOL = """
You have access to the following tool(s) for updating subtopic coverage:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

UPDATE_SUBTOPIC_COVERAGE_INSTRUCTIONS = """
<instructions>

## Process

1. **Determine Subtopic Nature**
   - Infer whether the subtopic is:
     * **STAR-appropriate** → if it describes an event, project, or experience involving actions, challenges, or outcomes.
     * **Descriptive** → if it focuses on background, motivation, interest, reasoning, or conceptual understanding rather than a specific event.

2. **Evaluate Completeness**
   - For **STAR-appropriate** subtopics:
       * Coverage requires STAR components:
         - **Situation:** Context or background
         - **Task:** Objective or responsibility
         - **Action:** Steps taken or reasoning
         - **Result:** Outcome, metric, or reflection
       * Fully covered when almost all components are clearly present and coherent.
       * However, if notes is already comprehensive, feel free to mark it as covered as there are more important subtopics to be covered in later section.
   - For **Descriptive** subtopics:
       * Coverage requires comprehensive factual, reflective, or conceptual detail.
       * Fully covered when the main question or theme is explained with sufficient clarity, logic, and completeness (even if not quantifiable).
       * However, if notes is already comprehensive, feel free to mark it as covered as there are more important subtopics to be covered in later section.

3. **Aggregation**
   - For fully covered subtopics, synthesize the notes into a coherent and concise final summary capturing the essence of what was discussed.
   - Avoid repetition or rephrasing—focus on integration and clarity.

4. **Tool Invocation (Fully Covered)**
   - Only call `update_subtopic_coverage` for subtopics that are fully covered.
   - Each call should include:
       * `subtopic_id`: the ID of the covered subtopic.
       * `aggregated_notes`: the aggregated summary notes.

</instructions>
"""

UPDATE_SUBTOPIC_COVERAGE_OUTPUT_FORMAT = """
<output_format>
<thinking>
For each subtopic, you should:
1. Review its notes.
2. Infer if STAR is relevant or not.
3. Evaluate completeness based on the inferred type.
4. For fully covered subtopics, aggregate the notes and call `update_subtopic_coverage`.
</thinking>

<tool_calls>
    <!-- One update_subtopic_coverage call per subtopic id, ONLY when the subtopic is considered fully covered -->
    <update_subtopic_coverage>
        <subtopic_id>The subtopic ID to be marked as covered</subtopic_id>
        <aggregated_notes>Aggregated notes from the subtopic's notes.</aggregated_notes>
    </update_subtopic_coverage>
    ...
</tool_calls>
</output_format>
"""

UPDATE_SUBTOPIC_NOTES_PROMPT = """
{CONTEXT}

{TOPICS_AND_SUBTOPICS}

{ADDITIONAL_CONTEXT}

{TOOL_DESCRIPTIONS}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""

UPDATE_SUBTOPIC_NOTES_CONTEXT = """
<agenda_manager_persona>
You are a agenda manager who assists an interviewer.
You observe the dialogue between the interviewer and the candidate, and your role is to look into each subtopic and update the subtopic's notes based on the given additional context.
Notes may be duplicated across subtopics if relevant.
</agenda_manager_persona>
"""

UPDATE_SUBTOPIC_NOTES_TOPICS_AND_SUBTOPICS = """
Here are the topics and subtopics to review:
<topics_list>
{topics_list}
</topics_list>
"""

UPDATE_SUBTOPIC_NOTES_ADDITIONAL_CONTEXT = """
Here is the context that you should refer into:
<context_reference>
{additional_context}
</context_reference>
"""

UPDATE_SUBTOPIC_NOTES_TOOL = """
You have access to the following tool(s) for updating subtopic's notes:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

UPDATE_SUBTOPIC_NOTES_INSTRUCTIONS = """
<instructions>

## Process

1. Review the context reference given.
2. For each subtopic, create or update a list of short notes (1-2 sentences each).
3. Include only relevant facts from the context; do not invent details.
4. Notes can repeat across subtopics if relevant to the subtopic and applicable.
5. Output only the structured notes (no extra commentary).
</instructions>
"""

UPDATE_SUBTOPIC_NOTES_OUTPUT_FORMAT = """
<output_format>

If you identify information worth storing that is relevant to the subtopic, use the following format:
<tool_calls>
    <!-- One update_subtopic_notes call per subtopic id -->
    <update_subtopic_notes>
        <subtopic_id>The subtopic ID.</subtopic_id>
        <note_list>["First note", "Second note", ...]</note_list>
    </update_subtopic_notes>
    ...
</tool_calls>
</output_format>
"""


UPDATE_LAST_MEETING_SUMMARY_PROMPT = """
{CONTEXT}

{INSTRUCTIONS}
"""

UPDATE_LAST_MEETING_SUMMARY_CONTEXT = """
<agenda_manager_persona>
You are a agenda manager who assists an interviewer. You maintain summaries of what has been discussed or reviewed so far, so the interviewer can recall context before continuing the next session. 
Right now, the interviewer is conducting an interview with the user about {interview_description}.
</agenda_manager_persona>
"""

UPDATE_LAST_MEETING_SUMMARY_INSTRUCTIONS = """
<context_to_summarize>
This is the context to be summarized:
```
{additional_context}
```
</context_to_summarize>

<instructions>
Given the content inside <context_to_summarize>, produce a summary highlighting key points that might be helpful for the interviewer about {interview_description}. 

Your goals:
- Capture main ideas, themes, or facts from the provided context.
- Emphasize points that could guide follow-up questions or exploration.
- Do not invent details; summarize only what is present.
- Keep it general enough to be useful regardless of the topic.
- Do not output anything else other than the summary.

Use neutral, professional language suitable for internal memory.
</instructions>
"""

UPDATE_USER_PORTRAIT_PROMPT = """
{CONTEXT}

{INSTRUCTIONS}
"""

UPDATE_USER_PORTRAIT_CONTEXT = """
<agenda_manager_persona>
You are a agenda manager who assists an interviewer. You maintain a structured user portrait to help the interviewer recall context and ask relevant questions. 
Right now, the interviewer is conducting an interview about {interview_description}.
The portrait should be updated based on any new context provided and in a valid JSON dictionary
</agenda_manager_persona>
"""

UPDATE_USER_PORTRAIT_INSTRUCTIONS = """
<user_portrait>
Current user portrait (may be partially filled):
{user_portrait}
</user_portrait>

<additional_context>
New context to incorporate:
```
{additional_context}
```
</additional_context>

<instructions>
Update the user portrait based on the new context. Produce a concise, structured summary in the same dictionary format as the current user portrait. 

Your goals:
- For existing fields: Update if new information significantly changes understanding.
- For new fields: Only add if revealing fundamental aspect of user, considering {interview_description}.
- Capture main ideas, themes, or facts from the additional context.
- Highlight points that could guide the interviewer in asking questions or retrieving relevant information.
- Do not invent details not present in the context.
- Output only the user portrait in a valid JSON dictionary; do not add explanatory text.
</instructions>
"""

UPDATE_LIST_OF_SUBTOPICS_PROMPT = """
{CONTEXT}

{TOPICS_AND_SUBTOPICS}

{ADDITIONAL_CONTEXT}

{TOOL_DESCRIPTIONS}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""

                
UPDATE_LIST_OF_SUBTOPICS_CONTEXT = """
<agenda_manager_persona>
You are a agenda manager assisting an interviewer. You observe the conversation and update the interview agenda based on the user's most recent message or additional context, while also considering the broader interview context.
The agenda consists of topics and subtopics that guide the interview.
Your role is to propose at most one NEW emergent subtopic to be added to the interview agenda if, and only if, the most recent user message or additional context introduces a clear, novel, and useful idea that:
1. Fits within one of the existing topics.
2. Cannot reasonably be covered by any existing subtopic.
3. Adds meaningful value to the interviewer.
Be concise and avoid redundancy; the agenda must remain clean, non-overlapping, and interpretable.

## Context
You are currently in an interview about: {interview_description}.
Use the user's most recent message as the primary trigger for evaluating emergent subtopics.
If there is no recent message, consider additional context or last meeting's summary.
</agenda_manager_persona>

This is the portrait of the user:
<user_portrait>
{user_portrait}
</user_portrait>
"""

UPDATE_LIST_OF_SUBTOPICS_TOPICS_AND_SUBTOPICS = """
Here is the topics and subtopics that you should consider when deciding to add new subtopics:
<topics_list>
{topics_list}
</topics_list>
"""

UPDATE_LIST_OF_SUBTOPICS_ADDITIONAL_CONTEXT = """
<additional_input_context>

Here is the summary of the last meeting:
<last_meeting_summary>
{last_meeting_summary}
</last_meeting_summary>

Here are previous interview events for additional context:
<previous_events>
{previous_events}
</previous_events>

Here is the most recent question-answer exchange:
<current_qa>
{current_qa}
</current_qa>

Here is some additional context that might be helpful:
<additional_context>
{additional_context}
</additional_context>

</additional_input_context>
"""

UPDATE_LIST_OF_SUBTOPICS_TOOL = """
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

UPDATE_LIST_OF_SUBTOPICS_INSTRUCTIONS = """
<instructions>
## Process
1. Read the topics and subtopics in `topics_list`.
2. Read the user's *most recent message* or *additional_context* carefully. Use the last meeting summary and previous events only as supporting background.
3. Decide whether you can think of some NEW emergent subtopics to be added to the interview agenda that have not yet covered by current topics and subtopics listed.
4. Add exactly one emergent subtopic—the strongest candidate—or none.

## Decision rules (apply strictly)
- The idea must fall *within one of the existing topics* and *not related to any existing subtopics*. If it does not clearly map to a parent topic, do NOT add it.
- The idea must be *novel*: the idea of emergence topic is uncommon, so if it can reasonably be addressed within any existing subtopic (even loosely), do NOT add it.
- If multiple candidate ideas appear, select **only the strongest single candidate**.
- If no candidate satisfies all rules, do not add any new subtopic.

## Ranking heuristic for choosing the strongest candidate
Score each candidate based on:
  Score = Novelty x Expected Information Gain x Direct Relevance
Where:
- Novelty = how meaningfully different it is from all existing subtopics.
- Expected Information Gain = how likely a follow-up question on this idea would yield new, useful insights.
- Direct Relevance = how clearly the idea aligns with its parent topic.

## Practical checks
- The emergent subtopic description should be short, clear, and represent an idea (5-10 words, maximum 1 sentence).
- Avoid redundancy, rephrasings, or overly narrow micro-subtopics.
- Do not add subtopics that drift outside the interview's intended scope.

## Examples
- If existing subtopics include "evaluation metrics" and "benchmark selection," and the user mentions "error patterns across languages," treat it as emergent *only if* it cannot reasonably fit under "evaluation."
- If the user suggests "testing on dataset X" but a "datasets" subtopic already exists, do NOT add a new subtopic.
</instructions>
"""

UPDATE_LIST_OF_SUBTOPICS_OUTPUT_FORMAT = """
<output_format>

<thinking>
Step-by-step reasoning (each step as a separate numbered line):
1. Identify candidate emergent idea(s) mentioned in the most recent message to be added as NOVEL subtopic to the current topics and subtopics list (explicitly list them or state "none"). If most recent message is empty, consider additional context or last meeting's summary.
2. Come up with this emergent idea(s) in 5-10 words, maximum 1 sentence.
3. For the selected candidate, review ALL listed topic along with their associated subtopics, and identify the topic ID under which this novel emergent subtopic best fits.
4. Explain, in one short sentence, why this candidate is NOVEL and cannot be reasonably grouped under any existing subtopic, especially since 'emergence' is uncommon.
5. Explain, in one short sentence, why this candidate is the strongest among candidates (use the ranking heuristic: Novelty x Expected Information Gain x Direct Relevance).
6. Conclude with a one-line decision: either "add" or "no_add" and a one-line justification.
7. If you decide to add, then perform the following tool call below of `add_emergent_subtopic`.
</thinking>

<!-- If and only if the decision is "add", produce exactly one tool call below. Otherwise, produce NO tool_calls section. -->
<tool_calls>
  <add_emergent_subtopic>
      <topic_id>The topic ID the emergent subtopic should belong to.</topic_id>
      <subtopic_description>Brief emergent subtopic description.</subtopic_description>
  </add_emergent_subtopic>
</tool_calls>
</output_format>
"""
