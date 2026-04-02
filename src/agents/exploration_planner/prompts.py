"""
Prompt templates for Exploration Planner Agent.

Contains prompts for:
- Conversation rollout prediction
- Emergent insight identification
- Emergent subtopic brainstorming
- Strategic question generation
"""

from src.utils.llm.prompt_utils import format_prompt


def get_prompt(prompt_type: str):
    """
    Factory function for exploration planner prompts.

    Args:
        prompt_type: Type of prompt to generate

    Returns:
        Formatted prompt template with placeholders
    """
    if prompt_type == "draft_rollouts":
        return format_prompt(DRAFT_ROLLOUTS_PROMPT, {
            "CONTEXT": DRAFT_ROLLOUTS_CONTEXT,
            "SESSION_STATE": DRAFT_ROLLOUTS_SESSION_STATE,
            "INSTRUCTIONS": DRAFT_ROLLOUTS_INSTRUCTIONS,
            "OUTPUT_FORMAT": DRAFT_ROLLOUTS_OUTPUT_FORMAT
        })

    elif prompt_type == "judge_coverage":
        return format_prompt(JUDGE_COVERAGE_PROMPT, {
            "CONTEXT": JUDGE_COVERAGE_CONTEXT,
            "ROLLOUT_DATA": JUDGE_COVERAGE_ROLLOUT_DATA,
            "INSTRUCTIONS": JUDGE_COVERAGE_INSTRUCTIONS,
            "OUTPUT_FORMAT": JUDGE_COVERAGE_OUTPUT_FORMAT
        })

    elif prompt_type == "brainstorm_emergent_subtopic":
        return format_prompt(BRAINSTORM_EMERGENT_SUBTOPIC_PROMPT, {
            "CONTEXT": BRAINSTORM_EMERGENT_SUBTOPIC_CONTEXT,
            "INSTRUCTIONS": BRAINSTORM_EMERGENT_SUBTOPIC_INSTRUCTIONS,
            "ADDITIONAL_CONTEXT": BRAINSTORM_EMERGENT_SUBTOPIC_ADDITIONAL_CONTEXT,
            "TOPICS_AND_SUBTOPICS": BRAINSTORM_EMERGENT_SUBTOPIC_TOPICS_AND_SUBTOPICS,
            "TOOL_DESCRIPTIONS": BRAINSTORM_EMERGENT_SUBTOPIC_TOOL,
            "OUTPUT_FORMAT": BRAINSTORM_EMERGENT_SUBTOPIC_OUTPUT_FORMAT
        })
    elif prompt_type == "identify_emergent_insights":
        return format_prompt(IDENTIFY_EMERGENT_INSIGHTS_PROMPT, {
            "CONTEXT": IDENTIFY_EMERGENT_INSIGHTS_CONTEXT,
            "INSTRUCTIONS": IDENTIFY_EMERGENT_INSIGHTS_INSTRUCTIONS,
            "TOPICS_AND_SUBTOPICS": IDENTIFY_EMERGENT_INSIGHTS_TOPICS_AND_SUBTOPICS,
            "ADDITIONAL_CONTEXT": IDENTIFY_EMERGENT_INSIGHTS_ADDITIONAL_CONTEXT,
            "TOOL_DESCRIPTIONS": IDENTIFY_EMERGENT_INSIGHTS_TOOL,
            "OUTPUT_FORMAT": IDENTIFY_EMERGENT_INSIGHTS_OUTPUT_FORMAT
        })
    elif prompt_type == "generate_strategic_questions":
        return format_prompt(GENERATE_STRATEGIC_QUESTIONS_PROMPT, {
            "CONTEXT": GENERATE_STRATEGIC_QUESTIONS_CONTEXT,
            "INSTRUCTIONS": GENERATE_STRATEGIC_QUESTIONS_INSTRUCTIONS,
            "ADDITIONAL_CONTEXT": GENERATE_STRATEGIC_QUESTIONS_ADDITIONAL_CONTEXT,
            "TOPICS_AND_SUBTOPICS": GENERATE_STRATEGIC_QUESTIONS_TOPICS_AND_SUBTOPICS,
            "TOOL_DESCRIPTIONS": GENERATE_STRATEGIC_QUESTIONS_TOOL,
            "OUTPUT_FORMAT": GENERATE_STRATEGIC_QUESTIONS_OUTPUT_FORMAT
        })

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

# =============================================================================
# BRAINSTORM EMERGENT SUBTOPICS PROMPT COMPONENTS
# =============================================================================

BRAINSTORM_EMERGENT_SUBTOPIC_PROMPT = """
{CONTEXT}

{TOPICS_AND_SUBTOPICS}

{ADDITIONAL_CONTEXT}

{TOOL_DESCRIPTIONS}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""


BRAINSTORM_EMERGENT_SUBTOPIC_CONTEXT = """
<exploration_planner_persona>
You are a strategic sub-topic brainstormer for semi-structured interviews. You observe the conversation and update the interview agenda based on the user's most recent message or additional context, while also considering the broader interview context.
The agenda consists of topics and subtopics that guide the interview.
Your role is to propose at most one NEW emergent subtopic to be added to the interview agenda if, and only if, the most recent user message or additional context introduces a clear, novel, and useful idea that:
1. Fits within one of the existing topics.
2. Cannot reasonably be covered by any existing subtopic.
3. Adds meaningful value to the interviewer.
Be concise and avoid redundancy; the agenda must remain clean, non-overlapping, and interpretable.

<context>
You are currently in an interview about: {interview_description}.
</context>
</exploration_planner_persona>

This is the portrait of the user:
<user_portrait>
{user_portrait}
</user_portrait>
"""

BRAINSTORM_EMERGENT_SUBTOPIC_TOPICS_AND_SUBTOPICS = """
Here is the topics and subtopics that you should consider when deciding to add new subtopics:
<topics_list>
{topics_list}
</topics_list>
"""

BRAINSTORM_EMERGENT_SUBTOPIC_ADDITIONAL_CONTEXT = """
<additional_input_context>

Here is the summary of the last meeting:
<last_meeting_summary>
{last_meeting_summary}
</last_meeting_summary>

Here are most recent conversation for additional context:
<recent_conversation>
{previous_events}
</recent_conversation>

</additional_input_context>
"""

BRAINSTORM_EMERGENT_SUBTOPIC_TOOL = """
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

BRAINSTORM_EMERGENT_SUBTOPIC_INSTRUCTIONS = """
<instructions>
## Process
1. Read the topics and subtopics in `topics_list`.
2. Read the user's recent conversation carefully. Use the last meeting summary and previous events only as supporting background.
3. Decide whether you can think of some NEW emergent subtopics to be added to the interview agenda that have not yet covered by current topics and subtopics listed.
4. Add exactly one emergent subtopic—the strongest candidate—or none.

## Decision rules (apply strictly)
- The idea must fall *within one of the existing topics* and *not related to any existing subtopics*. If it does not clearly map to a parent topic, do NOT add it.
- The idea must be *novel*: the idea of emergence topic is RARE, so if it can reasonably be addressed within any existing subtopic (even loosely), do NOT add it.
- The idea must enable *new probing that goes beyond deepening existing subtopics*, i.e., it should open up a qualitatively different line of inquiry that could surface emergent insights not reachable by further questioning within current subtopics.
- If multiple candidate ideas appear, select **only the strongest single candidate**.
- If no candidate satisfies all rules, do not add any new subtopic.

## What counts as an emergence
An emergent insight is a type of information that:

- Cannot be obtained by asking more detailed or follow-up questions within any existing subtopic.
- Reveals a new dimension, pattern, tradeoff, or mental model that reframes how existing subtopics are understood.
- Changes how future interview questions would be prioritized, sequenced, or interpreted.
- Surfaces higher-order understanding (e.g., cross-cutting constraints, implicit decision criteria, failure modes, or latent strategies).

NOT emergent insights:
- Additional examples, edge cases, or elaborations of existing subtopics.
- Narrow refinements or sub-steps of an existing subtopic.
- Clarifications that improve depth but not scope.
- Rephrasings of existing concepts using different wording.

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

BRAINSTORM_EMERGENT_SUBTOPIC_OUTPUT_FORMAT = """
<output_format>

<thinking>
Step-by-step reasoning (each step as a separate numbered line):
1. Identify candidate emergent idea(s) mentioned in the most recent conversation to be added as NOVEL subtopic to the current topics and subtopics list (explicitly list them or state "none").
2. Consider emergent insights which can be used as further ideas to probe more emergent insights. Come up with this emergent idea(s) in 5-10 words, maximum 1 sentence.
3. For the selected candidate, review ALL listed topic along with their associated subtopics, and identify the topic ID under which this novel emergent subtopic best fits.
4. Explain, in one short sentence, why this candidate is NOVEL and cannot be reasonably grouped under any existing subtopic, especially since 'emergence' is rare.
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

# =============================================================================
# GENERATE STRATEGIC QUESTIONS PROMPT COMPONENTS
# =============================================================================

GENERATE_STRATEGIC_QUESTIONS_PROMPT = """
{CONTEXT}

{TOPICS_AND_SUBTOPICS}

{ADDITIONAL_CONTEXT}

{TOOL_DESCRIPTIONS}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""

GENERATE_STRATEGIC_QUESTIONS_CONTEXT = """
<exploration_planner_persona>
You are a strategic question generator for semi-structured interviews.
Your role is to draft high-value interviewer questions that:
- Improve subtopic coverage
- Encourage depth and specificity
- Surface emergent insights when appropriate
- Follow natural conversational flow
</exploration_planner_persona>

<context>
You are currently in an interview about: {interview_description}.
</context>

<user_portrait>
This is the portrait of the user:
{user_portrait}
</user_portrait>
"""

GENERATE_STRATEGIC_QUESTIONS_TOPICS_AND_SUBTOPICS = """
Here is the topics and subtopics that you should consider when drafting new questions:
<topics_list>
{topics_list}
</topics_list>
"""

GENERATE_STRATEGIC_QUESTIONS_ADDITIONAL_CONTEXT = """
<additional_input_context>

Here is the summary of the last meeting:
<last_meeting_summary>
{last_meeting_summary}
</last_meeting_summary>

Here are most recent conversation for additional context:
<recent_conversation>
{previous_events}
</recent_conversation>

Predicted conversation trajectories (use as soft guidance, not strict plans):
<rollout_predictions>
{rollout_predictions}
</rollout_predictions>

</additional_input_context>
"""

GENERATE_STRATEGIC_QUESTIONS_TOOL = """
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

GENERATE_STRATEGIC_QUESTIONS_INSTRUCTIONS = """
<instructions>

## Strategic Question Generation

You are generating the NEXT interviewer questions in a semi-structured interview.
Your goal is to choose questions that maximize interview value given:
- what has already been discussed,
- what topics remain uncovered,
- and how the conversation is likely to evolve.

## Utility Function Framework

This interview is optimized using the utility function:
**U = α·Coverage - β·Cost + γ·Emergence**

Where:
- **Coverage (α = {alpha})**: Number of new subtopics covered (not yet marked as covered)
  - Highest weight - filling coverage gaps is the primary objective
  - Each newly covered subtopic adds +{alpha} to utility

- **Cost (β = {beta})**: Number of conversation turns needed
  - Moderate penalty - prefer efficient questions
  - Each additional turn subtracts -{beta} from utility

- **Emergence (γ = {gamma})**: Likelihood of eliciting novel/counter-intuitive insights (0-1 scale)
  - Bonus reward - encourages discovery of unexpected patterns
  - High emergence potential adds up to +{gamma} to utility

**Implication for Question Priority:**
- Questions targeting uncovered subtopics should have higher priority (α-weighted)
- Questions with high emergence potential deserve elevated priority (γ-weighted)
- Questions requiring many follow-ups should be lower priority unless justified by coverage/emergence gains

## How to Use the Provided Context (IMPORTANT)

### 1. Topics and Subtopics (`topics_list`)
- Treat the topics and subtopics as the **interview agenda**.
- Determine which subtopics are:
  - not yet covered,
  - partially covered,
  - or already sufficiently covered.
- **Prioritize questions that improve coverage**, especially for high-importance or underexplored subtopics.
- Avoid drafting questions that only repeat already-covered subtopics unless they add substantial new depth.

### 2. Recent Conversation (`recent_conversation`)
- Treat this as the **current state of the interview**.
- Your questions must follow naturally from what was most recently discussed.
- Use the user's phrasing, interests, and level of detail to shape question wording and depth.

### 3. Rollout Predictions (`rollout_predictions`)
- Rollout predictions show predicted conversation paths **ranked by utility score** (U = α·Coverage - β·Cost + γ·Emergence).
- The **highest-utility rollout (Rollout 1)** represents the most valuable predicted conversation path based on:
  - Expected new subtopic coverage
  - Emergence potential
  - Conversation efficiency

- Use rollouts to:
  - **Inform question priority**: Questions aligned with high-utility rollouts should receive higher priority
  - Anticipate which subtopics may naturally emerge next
  - Identify high-emergence opportunities flagged in rollout predictions
  - Avoid redundant or low-utility questions
  - Steer the conversation toward higher-value trajectories

- Treat rollout predictions as **soft guidance**, not rigid plans:
  - Do NOT copy rollout questions verbatim
  - DO use rollout insights to identify high-value coverage and emergence opportunities
  - Feel free to deviate if you identify better coverage gaps or emergence potential not captured in rollouts

## Strategic Objectives

Each question must addresses at least one sub-topic and should primarily serve ONE of the following goals:

1. **Fill Coverage Gaps**
   - Target a specific subtopic that is not yet fully covered.
   - Use questions that elicit concrete examples or detailed explanations.

2. **Explore Emergent Insights**
   - Follow up on counter-intuitive, uncommon, or surprising ideas already mentioned by the user.
   - Emergent insights must be grounded in PAST conversation, not speculation.

## Question Design Requirements

For EACH question:

- The question must be open-ended and conversational.
- Prefer questions that elicit specific experiences or examples (STAR-style when applicable).
- Avoid yes/no questions.
- Avoid introducing assumptions or facts not stated by the user.
- The question should feel like a natural next step in the interview.

## Required Metadata per Question

For each question, provide:
- **content**: The interviewer's question.
- **subtopic_id**: The primary subtopic this question addresses.
- **strategy_type**:
  - `"coverage_gap"` or
  - `"emergent_insight"`
- **priority** (1-10): Strategic importance of asking this question now, based on the utility function.

  Calculate priority by considering:

  1. **Coverage Impact** (α = {alpha}, highest weight):
     - Does this question target an uncovered subtopic?
     - Is the subtopic part of required topics or emergent high-value areas?
     - Higher priority for critical coverage gaps

  2. **Emergence Potential** (γ = {gamma}):
     - Could this question elicit counter-intuitive or unexpected insights?
     - Does it explore areas flagged for high emergence in rollout predictions?
     - Bonus priority for high emergence likelihood

  3. **Cost Efficiency** (β = {beta}, penalty):
     - Can this question efficiently cover its target without requiring many follow-ups?
     - Reduce priority if question requires extensive setup or context-building

  4. **Rollout Alignment**:
     - Review the utility scores of rollout predictions provided
     - Questions aligned with high-utility rollout paths (especially Rollout 1) should receive higher priority
     - Questions addressing subtopics/emergence from top-ranked rollouts deserve elevated priority

  **Priority Scale:**
  - **9-10**: Critical coverage gap + high emergence potential + efficient
  - **7-8**: Important coverage OR high emergence + moderate cost
  - **5-6**: Standard coverage question with moderate utility
  - **3-4**: Minor coverage improvement or high-cost questions
  - **1-2**: Low-utility questions (avoid unless no better options)

- **reasoning**: Why this question is strategically valuable given coverage state, utility function considerations, and context.

</instructions>
"""

GENERATE_STRATEGIC_QUESTIONS_OUTPUT_FORMAT = """
<output_format>

Produce exactly ONE tool call using the following XML structure:

<tool_calls>
  <suggest_strategic_questions>
    <questions>
      [
        {{
          "content": "Open-ended interviewer question",
          "subtopic_id": "1.1",
          "strategy_type": "coverage_gap",
          "priority": 9,
          "reasoning": "Why this question is strategically valuable"
        }},
        {{
          "content": "Another question",
          "subtopic_id": "2.3",
          "strategy_type": "emergent_insight",
          "priority": 7,
          "reasoning": "Explores unexpected angle"
        }}
      ]
    </questions>
  </suggest_strategic_questions>
</tool_calls>

Rules:
- Produce NO text outside the tool call
- Exactly {max_questions} questions
- strategy_type must be either "coverage_gap" or "emergent_insight"
- priority must be an integer between 1 and 10

</output_format>
"""


# =============================================================================
# DRAFT ROLLOUTS PROMPT COMPONENTS (New Single-Call Workflow)
# =============================================================================

DRAFT_ROLLOUTS_PROMPT = """
{CONTEXT}

{SESSION_STATE}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""

DRAFT_ROLLOUTS_CONTEXT = """
<exploration_planner_persona>
You are a strategic interview planning agent focused on maximizing topical coverage and emergent informational signal for an interview about {interview_description}. 

Given the current interview context and active subtopics, your task is to generate {num_rollouts} alternative interview plans. Each plan is a predicted conversation trajectory of {num_horizon} sequential interviewer-led questions followed by realistic candidate answers.

Each trajectory should:
- Intentionally explore different subtopic combinations, depths, or transitions.
- Encourage both subtopic emergence and conceptual emergence. Conceptual emergence refers to surfacing ideas, methods, or use cases that are uncommon, non-canonical, or unexpected within a subtopic, while remaining grounded in the candidate's real experience.
- Maintain internal coherence, with later turns conditioned on earlier answers.
- Represent a distinct interviewing strategy rather than surface-level variation.

These trajectories will be evaluated by a separate system to select the most effective plan, which will then guide the interviewer model.
</exploration_planner_persona>

<context>
Right now, you are observing a conversation between the interviewer and the user in an interview about {interview_description}.
</context>
"""

DRAFT_ROLLOUTS_SESSION_STATE = """
<user_portrait>
This is the portrait of the user:
{user_portrait}
</user_portrait>

Here are the topics and subtopics to review:
<topics_list>
{topics_list}
</topics_list>

Here are most recent conversation:
<recent_conversation>
{previous_events}
</recent_conversation>
"""

DRAFT_ROLLOUTS_INSTRUCTIONS = """
<instructions>
Generate {num_rollouts} diverse interview conversation rollout strategies, each consisting of {num_horizon} sequential Q&A turns.

For EACH rollout, reason step-by-step. For EACH turn, specify:
1. Question: The interviewer's next question.
2. Predicted Response: A brief, realistic summary of how the candidate would likely respond (not verbatim).
3. Subtopics Covered: List subtopic ID(s) addressed in this turn. Only newly introduced subtopics count toward coverage improvement.
4. Emergence Potential: A score between 0 and 1 indicating the likelihood of eliciting emergent signal, including:
   - Subtopic emergence (new relevant subtopics), and/or
   - Conceptual emergence (non-canonical, unconventional, or unexpected ideas within an existing subtopic, grounded in the candidate's experience).
5. Strategic Rationale: Why this turn is strategically useful given prior turns and the rollout's overall objective.

Diversity Requirements:
- Rollout 1: Prioritize filling the most critical or under-covered subtopics.
- Rollout 2: Prioritize high emergence potential, even at the cost of breadth.
- Rollout 3: Prioritize natural conversational continuity from the most recent interview turns.
- Additional rollouts: Use clearly distinct strategies (e.g., depth-first validation, cross-subtopic integration, or stress-testing assumptions).

Guidelines:
- Model how the conversation would realistically unfold, not an idealized or scripted interview.
- Condition later turns on earlier predicted responses.
- Avoid superficial rephrasing; each rollout should reflect a genuinely different interview strategy.
- Do not invent novelty—emergent insights should arise plausibly from the candidate's background, constraints, or prior answers.
</instructions>
"""

DRAFT_ROLLOUTS_OUTPUT_FORMAT = """
<output_format>

## Requirements

- Return a single valid JSON object and nothing else.
- The top-level object must contain exactly one key: "rollouts".
- Generate exactly {num_rollouts} rollout objects.
- Each rollout must contain exactly {num_horizon} predicted turns.
- rollout_id must be unique (e.g., "rollout_1", "rollout_2", ...).
- turn_number must start at 1 and increase sequentially within each rollout.
- subtopics_covered must list subtopic IDs provided in <topics_list>.
- Do not include subtopics already covered earlier in the rollout.
- emergence_potential must be a float between 0 and 1 representing the likelihood of eliciting emergent signal (subtopic or conceptual).
- strategic_rationale should be concise and specific to the current turn.

Return a JSON object with the following structure:

{{
  "rollouts": [
    {{
      "rollout_id": "rollout_1",
      "strategy_description": "Brief description of this rollout's overall interview strategy",
      "predicted_turns": [
        {{
          "turn_number": 1,
          "question": "Interviewer question text",
          "predicted_response": "Brief, realistic summary of the candidate's likely response",
          "subtopics_covered": ["1.1", "2.3"],
          "emergence_potential": 0.3,
          "strategic_rationale": "Why this turn is strategically useful given prior turns and the rollout strategy"
        }}
      ]
    }}
  ]
}}

Output JSON only. Do not include explanations, comments, or additional text.

</output_format>
"""


# =============================================================================
# JUDGE COVERAGE PROMPT COMPONENTS (Coverage Validation)
# =============================================================================

JUDGE_COVERAGE_PROMPT = """
{CONTEXT}

{ROLLOUT_DATA}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""

JUDGE_COVERAGE_CONTEXT = """
<coverage_judge_persona>
You are a coverage evaluation agent for interview rollouts.

You will be given a predicted conversation between an interviewer and a candidate. Your task is to assess each subtopic mentioned in the conversation and determine whether it has achieved full coverage.

For each subtopic, consider:
- The appropriate evaluation method (STAR framework for experience-based subtopics, or general descriptive evaluation for conceptual or knowledge-based subtopics).
- Whether the candidate's responses provide sufficient depth and completeness to mark the subtopic as covered.

Your role is to identify coverage, including incremental contributions from repeated or extended discussion.
</coverage_judge_persona>
"""

JUDGE_COVERAGE_ROLLOUT_DATA = """
Here are the topics and subtopics to review:
<topics_list>
{topics_list}
</topics_list>

Here is the predicted rollout conversation:
<predicted_rollout_conversation>
{rollout_data}
</predicted_rollout_conversation>
"""

JUDGE_COVERAGE_INSTRUCTIONS = """
<instructions>

Your task is to evaluate coverage for each subtopic based on the predicted rollout conversation.

Step-by-step for each predicted turn in the rollout:

1. Identify which subtopics (if any) are actually covered by this Q&A exchange.
2. Determine the appropriate evaluation method for each subtopic:
   - STAR framework (Situation, Task, Action, Result) for behavioral/experiential subtopics.
   - General descriptive evaluation for conceptual, factual, or reflective subtopics.
3. Assess whether the predicted response provides sufficient detail and depth to mark the subtopic as covered:
   - For STAR subtopics: all four elements (Situation, Task, Action, Result) must be meaningfully present.
   - For Descriptive subtopics: responses must include comprehensive factual or reflective detail with specificity.
4. Consider previous notes or coverage of subtopics in earlier turns.
5. Be strict: only mark subtopics as covered if the exchange truly meets coverage criteria.

Do NOT mark subtopics as covered if:
- Only surface-level mentions occur.
- STAR elements are incomplete.
- Responses are vague, generic, or lacking depth.

Considerations:
- Depth and specificity of the predicted response.
- Presence of STAR elements for applicable subtopics.
- Incremental contribution to coverage from repeated or extended discussion.

</instructions>
"""

JUDGE_COVERAGE_OUTPUT_FORMAT = """
<output_format>

## Output Instruction

- Return a JSON array listing only the subtopics that are now considered fully covered based on the predicted rollout conversation.
- Each entry must include:
  - `"subtopic_id"`: the ID of the subtopic.
  - `"coverage_rationale"`: a concise explanation of why the subtopic is considered covered (e.g., STAR elements or descriptive depth).
- Include only subtopics that meet the coverage criteria as outlined in the instructions.
- Output JSON only. Do not include explanations, comments, or any text outside the JSON array.

JSON Format Example:

[
  {{
    "subtopic_id": "1.1",
    "coverage_rationale": "Rationale why the subtopic is now covered."
  }},
  {{
    "subtopic_id": "2.3",
    "coverage_rationale": "Rationale why the subtopic is now covered."
  }}
]
</output_format>
"""

# =============================================================================
# IDENTIFY EMERGENT INSIGHTS PROMPT COMPONENTS
# =============================================================================

IDENTIFY_EMERGENT_INSIGHTS_PROMPT = """
{CONTEXT}

{TOPICS_AND_SUBTOPICS}

{ADDITIONAL_CONTEXT}

{TOOL_DESCRIPTIONS}

{INSTRUCTIONS}

{OUTPUT_FORMAT}
"""

IDENTIFY_EMERGENT_INSIGHTS_CONTEXT = """
<agenda_manager_persona>
You are a agenda manager assisting the interviewer during a live interview.
You observe the interaction between the interviewer and the user and update the session agenda accordingly.

You are responsible for detecting **emergent insights**:
- Novel or counter-intuitive findings
- Unexpected patterns or behaviors
- Observations that contradict or go beyond conventional wisdom

This analysis is based **only on the most recent question-answer exchange**.
You may choose **not** to identify or add emergent insights if none are present.
</agenda_manager_persona>

<context>
Right now, you are in an interview session with the interviewer and the user about: {interview_description}.
You have access to the agenda containing topics, subtopics, and observed emergent insights so far.
</context>

<user_portrait>
This is the portrait of the user:
{user_portrait}
</user_portrait>
"""

IDENTIFY_EMERGENT_INSIGHTS_TOPICS_AND_SUBTOPICS = """
Here is the topics and subtopics, along with the observed emergent insights so far that you should consider when deciding to add new emergent insights:
<topics_list>
{topics_list}
</topics_list>
"""

IDENTIFY_EMERGENT_INSIGHTS_ADDITIONAL_CONTEXT = """
Here is last meeting summary that might be helpful:
<last_meeting_summary>
{last_meeting_summary}
</last_meeting_summary>

Here is the stream of previous conversation for context:
<recent_conversation>
{previous_events}
</recent_conversation>

Reminder:
- The external tag of each event indicates the role of the sender of the event.
- Focus ONLY on processing the content within the current Q&A exchange above.
- Previous messages are shown only for context, not for reprocessing.
"""

IDENTIFY_EMERGENT_INSIGHTS_INSTRUCTIONS = """
<instructions>

Your task is to analyze the **current interviewer-user Q&A exchange** and determine whether it contains any **emergent insights**.

An emergent insight is:
- Counter-intuitive or unexpected within the interview topic scope
- Contradicts or challenges common or conventional beliefs
- Reveals a novel pattern or behavior not captured by existing subtopics
- **Not already covered by previously observed emergent insights**
- Grounded in the user's stated experience (not speculation or hypotheticals)

Emergent insights are **uncommon**. Do NOT force them.

## Novelty Scoring (use strictly)

Use an **integer novelty score from 1 to 5**:

- **5**: Highly counter-intuitive; challenges core assumptions or norms
- **4**: Clearly unexpected; reveals a strong new perspective
- **3**: Moderately novel; interesting deviation from standard practice
- **1-2**: Mild or unsurprising; NOT emergent (do not report)

## Analysis Procedure

1. Analyze the current Q&A exchange.
2. Review relevant topics, subtopics, and previously observed emergent insights.
3. Determine whether any **new and distinct** emergent insight is present.
4. For each valid emergent insight:
   - Identify the most relevant **subtopic_id**
   - Write a concise **description**
   - Assign a **novelty_score (1-5)**
   - Write a concise **evidence** of this emergence
   - State the **conventional belief** the insight contradicts

</instructions>
"""

IDENTIFY_EMERGENT_INSIGHTS_TOOL = """
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

IDENTIFY_EMERGENT_INSIGHTS_OUTPUT_FORMAT = """
<output_format>

<thinking>
Think step by step by analyzing:
- The current Q&A exchange
- Prior conversation context (for grounding only)
- Existing topics and subtopics
- Previously observed emergent insights

Decision rules:
- If **one or more** emergent insights are detected, call the `identify_emergent_insights` tool.
- If **no** emergent insights are detected, produce **NO tool call**.
- Do NOT output explanatory text outside the tool call.
</thinking>

<!-- If and only if emergent insights are detected, produce exactly ONE tool call -->

<tool_calls>
  <identify_emergent_insights>
    <emergent_insights>
      <insight>
        <subtopic_id>2.2</subtopic_id>
        <description>Concise description of the emergent insight</description>
        <novelty_score>4</novelty_score>
        <evidence>Relevant excerpt or paraphrase from the current Q&A</evidence>
        <conventional_belief>What is normally assumed instead</conventional_belief>
      </insight>
    </emergent_insights>
  </identify_emergent_insights>
</tool_calls>

<!-- If no emergent insights are detected, produce NO tool_calls section -->

</output_format>
"""
