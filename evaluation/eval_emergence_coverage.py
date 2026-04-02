import argparse
import json
import logging
import time
import os
import re
import ast
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
import numpy as np

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Constants
OPENAI_RETRIES = 3
MAX_TOKENS = 8192
OPENAI_CLIENT = None

# Global model client (will be initialized based on config)
MODEL_CLIENT = None
MODEL_CONFIG = None
TOKENIZER = None

# Deduplication config
DEDUP_MODEL = None
DEDUP_SIMILARITY_THRESHOLD = 0.8

JUDGE_PROMPT = """You are a agenda manager who assists an interviewer. You observe the dialogue between the interviewer and the candidate, and your role is to determine investigate each subtopic and its notes to determine whether the subtopic has achieved full coverage or not.

Your objectives:
1. Identify whether each subtopic should be evaluated using the STAR (Situation, Task, Action, Result) framework or a general descriptive evaluation.
2. Determine whether each subtopic is fully covered.
3. Return a final list of covered subtopics with NO duplicates (including semantically duplicated subtopics).

### Process

#### Step 1: Deduplicate Subtopics (MANDATORY)
- Subtopics may appear multiple times in the input.
- Treat subtopics with the same or semantically equivalent description as ONE subtopic.
- Create an internal list of UNIQUE subtopics.
- All evaluation must be performed only on this deduplicated list.

#### Step 2: Determine Subtopic Nature
For each UNIQUE subtopic, infer whether it is:
- **STAR-appropriate** → describes a specific event, project, or experience involving actions, challenges, or outcomes.
- **Descriptive** → focuses on background, motivation, interest, reasoning, or conceptual understanding rather than a specific event.

#### Step 3: Evaluate Completeness
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

### Subtopics
<subtopics>
{emergent_subtopics}
</subtopics>

### Interview Notes
<interview_notes>
{all_notes}
</interview_notes>

### Output Format (STRICT JSON ONLY)
Return a LIST of covered subtopics in the following format:

[
  {{
    "subtopic_covered": "Unique subtopic name",
    "rationale": "Why this subtopic is covered based on the interview notes"
  }}
]

If no subtopic is covered, return:

[]

### Your Response
"""

class EvaluationConfig:
    """Configuration for evaluation run"""
    def __init__(self, mode: str, base_path: str, sample_users_path: str,
                 ground_truth_path: str, max_workers: int = 16):
        self.mode = mode
        self.base_path = base_path
        self.sample_users_path = sample_users_path
        self.ground_truth_path = ground_truth_path
        self.max_workers = max_workers

        # Validate mode
        if mode not in ['sparkme', 'storysage', 'llmroleplay', 'freeform']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sparkme', 'storysage', or 'llmroleplay', 'freeform'")


def initialize_model(config_path: Optional[str] = None):
    """Initialize model client based on configuration"""
    global MODEL_CLIENT, MODEL_CONFIG, TOKENIZER

    if config_path is None:
        # Default to OpenAI
        MODEL_CLIENT = OPENAI_CLIENT
        MODEL_CONFIG = {
            "provider_name": "openai",
            "model_name": "gpt-4.1-nano",
            "generation_args": {
                "temperature": 0,
                "max_tokens": MAX_TOKENS
            }
        }
        logging.info("Using default OpenAI client")
        return

    # Load config from file
    with open(config_path, 'r') as f:
        MODEL_CONFIG = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "openai":
        MODEL_CLIENT = OPENAI_CLIENT
        logging.info(f"Using OpenAI client with model {MODEL_CONFIG.get('model_name', 'gpt-4.1-nano')}")
    elif provider == "local":
        # Initialize vLLM client
        try:
            from vllm import LLM

            model_name = MODEL_CONFIG["model_name"]
            model_args = MODEL_CONFIG.get("model_args", {})

            logging.info(f"Initializing vLLM with model {model_name}")
            MODEL_CLIENT = LLM(model=model_name, **model_args)

            # Initialize tokenizer for chat template
            logging.info(f"Loading tokenizer for {model_name}")
            TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            logging.info("vLLM client and tokenizer initialized successfully")
        except ImportError:
            logging.error("vLLM not installed. Please install with: pip install vllm")
            raise
        except Exception as e:
            logging.error(f"Error initializing vLLM: {e}")
            raise
    else:
        raise ValueError(f"Unknown provider: {provider}")


def initialize_dedup_model(model_name: str = "Qwen/Qwen3-Embedding-8B"):
    """Initialize vLLM model for embedding-based deduplication"""
    global DEDUP_MODEL
    from vllm import LLM
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # DEDUP_MODEL = LLM(model=model_name, task="embed", enforce_eager=True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    logging.info(f"Loaded vLLM embedding model: {model_name}")


def batch_embed_texts(texts: List[str]) -> np.ndarray:
    """
    Batch embed texts using vLLM.

    Args:
        texts: List of text strings to embed

    Returns:
        Normalized embeddings as numpy array of shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])

    outputs = DEDUP_MODEL.embed(texts)

    # Extract embeddings and normalize
    embeddings = np.array([output.outputs.embedding for output in outputs])

    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms

    return embeddings


def deduplicate_subtopics_with_embeddings(
    subtopics: List[Dict],
    embeddings: np.ndarray,
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    gt_embeddings: np.ndarray = None
) -> List[Dict]:
    """
    Deduplicate subtopics using pre-computed embeddings.
    Also filters out emergent subtopics that are similar to ground truth subtopics.

    Args:
        subtopics: List of dicts with 'emergent_subtopic' and 'topic' keys
        embeddings: Pre-computed normalized embeddings for each subtopic
        threshold: Cosine similarity threshold (>= threshold = duplicate)
        gt_embeddings: Optional ground truth subtopic embeddings to filter against

    Returns:
        Deduplicated list of subtopics (with GT-similar ones removed)
    """
    if not subtopics or len(subtopics) == 0:
        return subtopics

    # First, filter out emergent subtopics similar to ground truth
    if gt_embeddings is not None and len(gt_embeddings) > 0:
        valid_indices = []
        for i in range(len(embeddings)):
            # Check similarity against all ground truth subtopics
            similarities = np.dot(gt_embeddings, embeddings[i])
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0
            if max_similarity < threshold:
                valid_indices.append(i)

        # Update subtopics and embeddings to only include non-GT-similar ones
        subtopics = [subtopics[i] for i in valid_indices]
        embeddings = embeddings[valid_indices] if len(valid_indices) > 0 else np.array([])

    if len(subtopics) <= 1:
        return subtopics

    # Greedy deduplication among remaining emergent subtopics
    keep_indices = []
    for i in range(len(embeddings)):
        is_duplicate = False
        for kept_idx in keep_indices:
            similarity = np.dot(embeddings[i], embeddings[kept_idx])
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)

    return [subtopics[i] for i in keep_indices]


def safe_parse_json(text: str):
    """Parse JSON from various formats including code blocks"""
    text = text.strip()
    if not text:
        return None

    # Try to find ```json ... ``` fenced block first
    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if codeblock_match:
        candidate = codeblock_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try parsing entire text as JSON directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (last resort)
    try:
        parsed_dict = ast.literal_eval(text)
        if isinstance(parsed_dict, dict):
            return parsed_dict
    except Exception:
        pass

    return None


def request_openai_completion(msg: List[Dict]) -> str:
    """Request completion from OpenAI with retries"""
    global MODEL_CLIENT, MODEL_CONFIG

    model_name = MODEL_CONFIG.get("model_name", "gpt-4.1-nano")
    gen_args = MODEL_CONFIG.get("generation_args", {})

    for attempt in range(OPENAI_RETRIES):
        try:
            response = MODEL_CLIENT.chat.completions.create(
                model=model_name,
                messages=msg,
                **gen_args
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Error calling OpenAI API (attempt {attempt + 1}/{OPENAI_RETRIES}): {e}")
            time.sleep(61)

    logging.error(f"Could not resolve error after {OPENAI_RETRIES} attempts")
    return None


def build_ground_truth_index(ground_truth: List[Dict]) -> Dict[str, Dict]:
    """Build an index of ground truth subtopics by ID"""
    topic_list = []
    subtopic_list = []

    for topic_idx in range(len(ground_truth)):
        topic_list.append(ground_truth[topic_idx]['topic'])
        for subtopic in ground_truth[topic_idx].get('subtopics', []):
            subtopic_description = subtopic['subtopic_description']
            subtopic_list.append(subtopic_description)

    return topic_list, subtopic_list


def extract_all_notes_interviewer(session_agenda: Dict) -> str:
    """Extract ALL notes from interviewer session agenda format"""
    all_notes_parts = []

    for core_topic_id, core_topic in session_agenda["interview_topic_manager"]["core_topic_dict"].items():
        topic_desc = core_topic.get("description", "")

        # Process required subtopics
        for subto_id, subto in core_topic.get('required_subtopics', {}).items():
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")
            final_note = ""

            if notes:
                notes_str = "\n -".join(notes)
                final_note += f"**Topic: {topic_desc}**\n**Subtopic: {subto_desc}**\n{notes_str}"
                
            if final_summary:
                final_note += f"**Topic: {topic_desc}**\n**Subtopic: {subto_desc}**\nSummary: {final_summary}"
                
            if len(final_note) > 1:
                all_notes_parts.append(final_note)

        # Process emergent subtopics
        for subto_id, subto in core_topic.get('emergent_subtopics', {}).items():
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")
            final_note = ""

            if notes:
                notes_str = "\n -".join(notes)
                final_note += f"**Topic: {topic_desc}**\n**Subtopic: {subto_desc}**\n{notes_str}"
                
            if final_summary:
                final_note += f"**Topic: {topic_desc}**\n**Subtopic: {subto_desc}**\nSummary: {final_summary}"
                
            if len(final_note) > 1:
                all_notes_parts.append(final_note)

    return "\n\n".join(all_notes_parts)


def extract_all_notes_storysage(memory_bank: dict, snapshot_idx: int) -> str:
    """Extract notes from storysage format for a specific turn only"""
    all_notes_parts = []

    for mem in memory_bank.get("memories", []):
        turn = mem.get("metadata", {}).get("turn", -1)
        # Only get notes from this specific turn
        if turn == snapshot_idx:
            notes = mem.get("text", "").strip()
            if notes:
                all_notes_parts.append(f"- {notes}")

    return "\n".join(all_notes_parts)


def get_covered_subtopics_interviewer(session_agenda: Dict, gt_index: Dict) -> List[str]:
    """Get list of subtopic IDs that have any notes in interviewer format"""
    covered_subtopics = []

    for core_topic_id, core_topic in session_agenda["interview_topic_manager"]["core_topic_dict"].items():
        for subto_id, subto in core_topic.get('required_subtopics', {}).items():
            if subto_id in gt_index and (len(subto.get("notes", [])) > 0 or len(subto.get("final_summary", "")) > 0):
                covered_subtopics.append(subto_id)

    return covered_subtopics


def get_covered_subtopics_storysage(topics_filled: List[Dict], gt_index: Dict) -> List[str]:
    """Get list of subtopic IDs that have any notes in storysage format"""
    covered_subtopics = []

    for topic in topics_filled:
        for subtopic in topic.get("subtopics", []):
            subto_id = subtopic.get('subtopic_id')
            if subto_id in gt_index and len(subtopic.get("notes", [])) > 0:
                covered_subtopics.append(subto_id)

    return covered_subtopics


def extract_all_notes_llmroleplay(history: List[Dict], snapshot_idx: int) -> str:
    """Extract notes from llmroleplay baseline format for a specific turn only (interview_log.jsonl)

    Only returns the notes from the exact turn specified by snapshot_idx.
    """
    if snapshot_idx >= len(history):
        return ""

    item = history[snapshot_idx]
    notes = item.get("notes", None)

    if notes is not None and len(notes.strip()) > 0:
        topic_idx = item.get("topic_index", 0)
        question_idx = item.get("question_index", 0)
        subtopic_id = f"{topic_idx + 1}.{question_idx + 1}"
        return f"**Subtopic {subtopic_id}**\n{notes}"

    return ""


def get_covered_subtopics_llmroleplay(history: List[Dict], max_turn: int, gt_index: Dict) -> List[str]:
    """Get list of subtopic IDs that have any notes in llmroleplay format

    Notes: Only considers subtopics that have notes in the final aggregated state,
    since notes get updated over multiple turns.
    """
    # Dictionary to track latest notes for each subtopic
    subtopic_has_notes = {}

    for item in history:
        if item["turn_number"] > max_turn:
            break

        notes = item.get("notes", None)
        topic_idx = item.get("topic_index", 0)
        question_idx = item.get("question_index", 0)
        subtopic_id = f"{topic_idx + 1}.{question_idx + 1}"

        # Update latest state for this subtopic
        if notes is not None and len(notes.strip()) > 0:
            subtopic_has_notes[subtopic_id] = True
        elif subtopic_id in subtopic_has_notes:
            # If notes were cleared/emptied, update that too
            subtopic_has_notes[subtopic_id] = False

    # Return only subtopics that have notes in final state and exist in ground truth
    covered_subtopics = [sid for sid, has_notes in subtopic_has_notes.items()
                         if has_notes and sid in gt_index]

    return covered_subtopics

def extract_all_notes_freeform(history: List[Dict], max_turn: int) -> str:
    """Extract ALL notes from freeform baseline format (interview_log.jsonl)

    Notes: Only keeps the LATEST version of notes for each subtopic,
    since notes get aggregated and updated over multiple turns.
    """
    # Dictionary to store latest notes for each subtopic
    subtopic_notes = []

    for item in history:
        if item["turn_number"] > max_turn:
            break

        notes = item.get("notes", None)
        if notes is not None and len(notes.strip()) > 0:
            subtopic_notes.append(notes)

    # Format all collected notes
    return "\n- ".join(subtopic_notes)

def filter_valid_subtopics(ground_truth: List[Dict], emergent_subtopic_list: List[Dict]) -> List[Dict]:
    """
    Filter emergent subtopics to only include valid ones (topic matches, not in ground truth).

    Args:
        ground_truth: Ground truth data
        emergent_subtopic_list: List of emergent subtopics

    Returns:
        List of valid subtopics
    """
    gt_topic_list, gt_subtopic_list = build_ground_truth_index(ground_truth)

    valid_subtopics = []
    for emergent_subtopic in emergent_subtopic_list:
        if 'topic' not in emergent_subtopic or 'emergent_subtopic' not in emergent_subtopic:
            continue
        if emergent_subtopic['topic'] in gt_topic_list and emergent_subtopic['emergent_subtopic'] not in gt_subtopic_list:
            valid_subtopics.append(emergent_subtopic)

    return valid_subtopics


def load_fixed_emergent_pool(
    user_id: str,
    ground_truth: List[Dict],
    config: 'EvaluationConfig',
    max_snapshot: int,
    embedding_index: Dict[str, np.ndarray]
) -> List[Dict]:
    """
    Load ALL emergent subtopics from steps 1 to max_snapshot and deduplicate ONCE.
    Returns a FIXED reference pool.

    Args:
        user_id: User identifier
        ground_truth: Ground truth topics for filtering
        config: Evaluation configuration
        max_snapshot: Maximum snapshot index (T_max)
        embedding_index: Pre-computed embeddings for deduplication

    Returns:
        Deduplicated list of emergent subtopics with 'detected_at_step' metadata
    """
    emerg_dir = f"{config.base_path}/{user_id}/evaluations_emergence"
    all_subtopics = []

    # Collect ALL emergent subtopics from all steps
    for i in range(1, max_snapshot + 1):
        emerg_path = f"{emerg_dir}/snap_eval_{i}.json"
        if os.path.exists(emerg_path):
            with open(emerg_path, 'r') as f:
                emerg_data = json.load(f)
                if (emerg_data and
                    not emerg_data.get('error', True) and
                    'emergent_subtopics' in emerg_data and
                    isinstance(emerg_data['emergent_subtopics'], list)):
                    for subtopic in emerg_data['emergent_subtopics']:
                        # Add metadata: which step detected this subtopic
                        subtopic_copy = subtopic.copy()
                        subtopic_copy['detected_at_step'] = i
                        all_subtopics.append(subtopic_copy)

    if not all_subtopics:
        return []

    # Filter valid subtopics (topic matches GT, not already in GT)
    valid_subtopics = filter_valid_subtopics(ground_truth, all_subtopics)
    if not valid_subtopics:
        return []

    # Deduplicate using embeddings
    descriptions = [s['emergent_subtopic'] for s in valid_subtopics]
    embeddings_list = [embedding_index[d] for d in descriptions if d in embedding_index]

    if not embeddings_list:
        return valid_subtopics

    embeddings = np.array(embeddings_list)

    # Get GT embeddings for filtering against ground truth
    _, gt_subtopic_list = build_ground_truth_index(ground_truth)
    gt_embeddings_list = [embedding_index[d] for d in gt_subtopic_list if d in embedding_index]
    gt_embeddings = np.array(gt_embeddings_list) if gt_embeddings_list else None

    deduped = deduplicate_subtopics_with_embeddings(
        valid_subtopics,
        embeddings,
        threshold=DEDUP_SIMILARITY_THRESHOLD,
        gt_embeddings=gt_embeddings
    )

    return deduped


def create_evaluation_messages(
    ground_truth: List[Dict],
    session_data: Dict,
    mode: str,
    emergent_subtopic_list: List[Dict],
    snapshot_idx: int = None,
    embedding_index: Dict[str, np.ndarray] = None
) -> Tuple[List[List[Dict]], List[str], int]:
    """
    Create evaluation messages comparing ALL notes against ALL ground truth facts

    Args:
        ground_truth: Ground truth data
        session_data: Session agenda (interviewer), memory bank (storysage), or history list (llmroleplay/freeform)
        mode: 'sparkme', 'storysage', 'llmroleplay', or 'freeform'
        snapshot_idx: For llmroleplay/freeform mode, the turn number to evaluate up to

    Returns:
        Tuple of (list of messages, list of subtopic IDs, number of emergent subtopics)
    """
    # Extract ALL notes from the session
    if mode == 'sparkme':
        all_notes = extract_all_notes_interviewer(session_data)
    elif mode == 'storysage':
        all_notes = extract_all_notes_storysage(session_data, snapshot_idx)
    elif mode == 'llmroleplay':
        all_notes = extract_all_notes_llmroleplay(session_data, snapshot_idx)
    elif mode == 'freeform':
        all_notes = extract_all_notes_freeform(session_data, snapshot_idx)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # If no notes, return empty
    if not all_notes or len(all_notes.strip()) == 0:
        return [], [], 0

    # Filter valid subtopics (basic string-match filtering)
    valid_subtopics = filter_valid_subtopics(ground_truth, emergent_subtopic_list)

    if not valid_subtopics:
        return [], [], 0

    # Deduplicate using pre-computed embeddings if available
    if embedding_index is not None and len(valid_subtopics) >= 1:
        # Get embeddings for valid emergent subtopics from the index
        descriptions = [s['emergent_subtopic'] for s in valid_subtopics]
        embeddings = np.array([embedding_index[desc] for desc in descriptions if desc in embedding_index])

        # Get embeddings for ground truth subtopics
        _, gt_subtopic_list = build_ground_truth_index(ground_truth)
        gt_embeddings = np.array([embedding_index[desc] for desc in gt_subtopic_list if desc in embedding_index])

        final_emergent_subtopics_detected = deduplicate_subtopics_with_embeddings(
            valid_subtopics,
            embeddings,
            threshold=DEDUP_SIMILARITY_THRESHOLD,
            gt_embeddings=gt_embeddings if len(gt_embeddings) > 0 else None
        )
        logging.info(f"Final emergent subtopics: {len(final_emergent_subtopics_detected)}")
    else:
        # No embedding index - skip dedup
        final_emergent_subtopics_detected = valid_subtopics

    if len(final_emergent_subtopics_detected) == 0:
        return [], [], 0

    emerg_formatted = ""
    for subtopic_detected in final_emergent_subtopics_detected:
        emerg_formatted += f"- {subtopic_detected['emergent_subtopic']} (Topic: {subtopic_detected['topic']})\n"

    # Create a SINGLE evaluation message comparing ALL notes to ALL ground truth
    msg = [{
        "role": "user",
        "content": JUDGE_PROMPT.format(
            emergent_subtopics=emerg_formatted,
            all_notes=all_notes
        )
    }]

    # Return single message with a placeholder ID and count of emergent subtopics
    return [msg], ['all_subtopics'], len(final_emergent_subtopics_detected)


def evaluate_emergence_openai(msg: List[Dict], num_emergent_subtopics: int) -> Dict:
    """Evaluate emergence by comparing all notes against all ground truth"""
    response = request_openai_completion(msg)

    if response is None:
        return {
            'list_subtopic_covered': [],
            'error': True
        }

    parsed_response = safe_parse_json(response)

    if parsed_response is None:
        return {
            'list_subtopic_covered': parsed_response,
            'error': True
        }

    # Check for hallucination: if more covered subtopics than emergent subtopics provided
    if len(parsed_response) > num_emergent_subtopics:
        logging.warning(f"Hallucination detected: {len(parsed_response)} covered > {num_emergent_subtopics} emergent")
        return {
            'list_subtopic_covered': None,
            'error': True
        }

    return {
        'list_subtopic_covered': parsed_response,
        'error': False
    }


def prepare_session_data_raw(user_id: str, snapshot_idx: int, ground_truth: List[Dict], config: EvaluationConfig, surgery: bool = False):
    """
    Prepare raw session data for emergence evaluation (without creating messages yet).
    This allows collecting all subtopics first for batch embedding.
    """
    # Construct paths based on mode
    eval_dir = f"{config.base_path}/{user_id}/evaluations_emergence_coverage"
    emerg_dir = f"{config.base_path}/{user_id}/evaluations_emergence"
    if config.mode == 'sparkme':
        session_path = f"{config.base_path}/{user_id}/execution_logs/session_0/session_agenda_snap_{snapshot_idx}.json"
    elif config.mode == 'storysage':
        session_path = f"{config.base_path}/{user_id}/memory_bank_content.json"
    elif config.mode == 'llmroleplay':
        session_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
    elif config.mode == 'freeform':
        session_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    save_eval_path = f"{eval_dir}/snap_eval_{snapshot_idx}.json"

    # Skip if session file doesn't exist
    if not os.path.exists(session_path):
        return None

    # Skip if already evaluated (and not empty)
    if not surgery and os.path.exists(save_eval_path):
        with open(save_eval_path, 'r') as f:
            existing_results = json.load(f)
            if len(existing_results) > 0:
                return None

    # Load emergence data
    emergent_subtopic_list = []
    emerg_path = f"{emerg_dir}/snap_eval_{snapshot_idx}.json"
    if os.path.exists(emerg_path):
        with open(emerg_path, 'r') as f:
            emerg_data = json.load(f)
            if emerg_data and not emerg_data['error'] and 'emergent_subtopics' in emerg_data and isinstance(emerg_data['emergent_subtopics'], list):
                emergent_subtopic_list.extend(emerg_data['emergent_subtopics'])

    if len(emergent_subtopic_list) == 0:
        return None

    # Filter valid subtopics
    valid_subtopics = filter_valid_subtopics(ground_truth, emergent_subtopic_list)
    if len(valid_subtopics) == 0:
        return None

    # Load session data
    if config.mode == 'llmroleplay' or config.mode == 'freeform':
        # For these modes, load the JSONL file as a list of history items
        history = []
        with open(session_path) as f:
            for line in f.readlines():
                history.append(json.loads(line))

        # Check if snapshot_idx is beyond the history length
        if snapshot_idx >= len(history) + 4:
            return None

        session_data = history
    else:
        # For interviewer and storysage modes, load JSON
        with open(session_path) as f:
            session_data = json.load(f)

    return {
        'user_id': user_id,
        'snapshot_idx': snapshot_idx,
        'ground_truth': ground_truth,
        'session_data': session_data,
        'emergent_subtopic_list': emergent_subtopic_list,
        'valid_subtopics': valid_subtopics,
        'save_path': save_eval_path,
        'eval_dir': eval_dir
    }


def finalize_session_data(raw_session_info: Dict, config: EvaluationConfig, embedding_index: Dict[str, np.ndarray] = None):
    """
    Finalize session data by creating evaluation messages with deduplication.
    Uses pre-computed embedding index for batch deduplication.
    """
    # Create evaluation messages with embedding-based deduplication
    messages, subtopic_ids, num_emergent_subtopics = create_evaluation_messages(
        raw_session_info['ground_truth'],
        raw_session_info['session_data'],
        config.mode,
        raw_session_info['emergent_subtopic_list'],
        raw_session_info['snapshot_idx'],
        embedding_index=embedding_index
    )

    if len(messages) == 0:
        return None

    return {
        'user_id': raw_session_info['user_id'],
        'snapshot_idx': raw_session_info['snapshot_idx'],
        'messages': messages,
        'subtopic_ids': subtopic_ids,
        'num_emergent_subtopics': num_emergent_subtopics,
        'save_path': raw_session_info['save_path'],
        'eval_dir': raw_session_info['eval_dir']
    }


def process_session_openai(user_id: str, snapshot_idx: int, ground_truth: List[Dict], config: EvaluationConfig, surgery: bool = False):
    """Process a single session snapshot with OpenAI (for threading)"""
    raw_session_info = prepare_session_data_raw(user_id, snapshot_idx, ground_truth, config, surgery=surgery)

    if raw_session_info is None:
        return

    # Finalize with embedding-based deduplication if index is provided
    session_info = finalize_session_data(raw_session_info, config, embedding_index=None)

    if session_info is None:
        return

    # Evaluate ALL notes against ALL ground truth in a single call
    if len(session_info['messages']) == 0:
        return

    msg = session_info['messages'][0]  # Now only one message
    num_emergent_subtopics = session_info['num_emergent_subtopics']
    result = evaluate_emergence_openai(msg, num_emergent_subtopics)

    # Save results as a single evaluation
    os.makedirs(session_info['eval_dir'], exist_ok=True)
    with open(session_info['save_path'], 'w') as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Evaluate interview sessions')
    parser.add_argument('--mode', type=str, required=True, choices=['sparkme', 'storysage', 'llmroleplay', 'freeform'],
                        help='Evaluation mode: interviewer, storysage, llmroleplay, or freeform')
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to logs directory')
    parser.add_argument('--sample-users-path', type=str,
                        default='analysis/sample_users_50.json',
                        help='Path to sample users JSON')
    parser.add_argument('--ground-truth-path', type=str,
                        default='data/sample_user_profiles',
                        help='Path to ground truth data directory')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to model config JSON file (optional, defaults to OpenAI)')
    parser.add_argument('--max-workers', type=int, default=16,
                        help='Maximum number of parallel workers')
    parser.add_argument('--num-users', type=int, default=30,
                        help='Number of users to process')
    parser.add_argument('--snapshot-start', type=int, default=1,
                        help='Starting snapshot index')
    parser.add_argument('--snapshot-end', type=int, default=80,
                        help='Ending snapshot index')
    parser.add_argument('--snapshot-step', type=int, default=1,
                        help='Snapshot step size')
    parser.add_argument('--surgery', action="store_true", help="Enable surgery")
    parser.add_argument('--dedup-model', type=str, default='Qwen/Qwen3-Embedding-8B',
                        help='Sentence-transformers model for deduplication')
    parser.add_argument('--dedup-threshold', type=float, default=DEDUP_SIMILARITY_THRESHOLD,
                        help='Cosine similarity threshold for deduplication (0.0-1.0)')

    args = parser.parse_args()

    # Initialize deduplication model
    initialize_dedup_model(args.dedup_model)

    # Initialize model
    initialize_model(args.model_config)

    # Create config
    config = EvaluationConfig(
        mode=args.mode,
        base_path=args.base_path,
        sample_users_path=args.sample_users_path,
        ground_truth_path=args.ground_truth_path,
        max_workers=args.max_workers
    )

    # Load sample users
    with open(config.sample_users_path, 'r') as f:
        sample_users = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "local":
        # vLLM: Batch ALL evaluations at once
        logging.info("Using vLLM - collecting all sessions for batch inference")

        # Step 1: Prepare all raw session data and collect unique subtopic texts (both emergent AND ground truth)
        all_raw_session_infos = []
        all_unique_subtopics = set()  # Emergent subtopics
        all_gt_subtopics = set()  # Ground truth subtopics

        for person_profile in tqdm(sample_users[:args.num_users], desc="Preparing sessions"):
            user_id = person_profile["User ID"]

            # Load ground truth
            gt_path = f"{config.ground_truth_path}/{user_id}/{user_id}_topics_filled.json"
            with open(gt_path) as f:
                ground_truth = json.load(f)

            # Collect ground truth subtopics for this user
            _, gt_subtopic_list = build_ground_truth_index(ground_truth)
            for gt_subtopic in gt_subtopic_list:
                all_gt_subtopics.add(gt_subtopic)

            # Prepare all snapshots for this user
            for snapshot_idx in range(args.snapshot_start, args.snapshot_end, args.snapshot_step):
                raw_session_info = prepare_session_data_raw(user_id, snapshot_idx, ground_truth, config, surgery=args.surgery)
                if raw_session_info is not None:
                    all_raw_session_infos.append(raw_session_info)
                    # Collect unique emergent subtopic texts for batch embedding
                    for subtopic in raw_session_info['valid_subtopics']:
                        all_unique_subtopics.add(subtopic['emergent_subtopic'])

        logging.info(f"Prepared {len(all_raw_session_infos)} raw sessions")
        logging.info(f"Found {len(all_unique_subtopics)} unique emergent subtopic texts")
        logging.info(f"Found {len(all_gt_subtopics)} unique ground truth subtopic texts")

        # Step 2: Finalize session data with deduplication
        all_session_infos = []
        for raw_session_info in tqdm(all_raw_session_infos, desc="Finalizing sessions with dedup"):
            session_info = finalize_session_data(raw_session_info, config, embedding_index=None)
            if session_info is not None:
                all_session_infos.append(session_info)

        logging.info(f"Finalized {len(all_session_infos)} sessions for evaluation")

        # Step 3: Collect ALL messages and metadata (now one message per session)
        all_messages = []
        all_metadata = []  # Track which session each message belongs to

        for session_info in all_session_infos:
            # Now each session has only ONE message (all notes vs all ground truth)
            if len(session_info['messages']) > 0:
                all_messages.append(session_info['messages'][0])
                all_metadata.append({
                    'session_info': session_info
                })

        logging.info(f"Total sessions to evaluate: {len(all_messages)}")

        # Step 4: Batch inference ALL at once
        from vllm import SamplingParams

        prompts = []
        for msg in tqdm(all_messages, desc="Applying chat templates"):
            prompt = TOKENIZER.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        gen_args = MODEL_CONFIG.get("generation_args", {})
        sampling_params = SamplingParams(**gen_args)

        logging.info("Running batch inference...")
        outputs = MODEL_CLIENT.generate(prompts, sampling_params)

        # Step 5: Parse results and compile
        logging.info("Parsing results...")
        session_results = {}

        for metadata, output in tqdm(zip(all_metadata, outputs), total=len(outputs), desc="Processing outputs"):
            session_info = metadata['session_info']
            session_key = (session_info['user_id'], session_info['snapshot_idx'])
            num_emergent_subtopics = session_info['num_emergent_subtopics']

            generated_text = output.outputs[0].text.strip()
            parsed_response = safe_parse_json(generated_text)

            # Check for hallucination: if more covered subtopics than emergent subtopics provided
            if parsed_response is not None and len(parsed_response) > num_emergent_subtopics:
                logging.warning(f"Hallucination detected for {session_key}: {len(parsed_response)} covered > {num_emergent_subtopics} emergent")
                result = {
                    "list_subtopic_covered": None,
                    'error': True
                }
            elif parsed_response is None:
                result = {
                    "list_subtopic_covered": None,
                    'error': True
                }
            else:
                result = {
                    'list_subtopic_covered': parsed_response,
                    'error': False
                }

            session_results[session_key] = {
                'result': result,
                'save_path': session_info['save_path'],
                'eval_dir': session_info['eval_dir']
            }

        # Step 6: Write all results to files
        logging.info(f"Writing results to {len(session_results)} files...")
        for session_key, session_data in tqdm(session_results.items(), desc="Writing results"):
            os.makedirs(session_data['eval_dir'], exist_ok=True)
            with open(session_data['save_path'], 'w') as f:
                json.dump(session_data['result'], f, indent=2)

        logging.info("vLLM batch evaluation complete!")

    else:
        # OpenAI: Use threading for parallel API calls
        logging.info("Using OpenAI - processing sessions in parallel with threading")

        # Step 1: Collect all raw session data and unique subtopic texts (both emergent AND ground truth)
        logging.info("Collecting all sessions for batch embedding...")
        all_raw_session_infos = []
        all_unique_subtopics = set()  # Emergent subtopics
        all_gt_subtopics = set()  # Ground truth subtopics
        user_ground_truths = {}

        for person_profile in tqdm(sample_users[:args.num_users], desc="Preparing sessions"):
            user_id = person_profile["User ID"]

            # Load ground truth
            gt_path = f"{config.ground_truth_path}/{user_id}/{user_id}_topics_filled.json"
            with open(gt_path) as f:
                ground_truth = json.load(f)
            user_ground_truths[user_id] = ground_truth

            # Collect ground truth subtopics for this user
            _, gt_subtopic_list = build_ground_truth_index(ground_truth)
            for gt_subtopic in gt_subtopic_list:
                all_gt_subtopics.add(gt_subtopic)

            # Prepare all snapshots for this user
            for snapshot_idx in range(args.snapshot_start, args.snapshot_end, args.snapshot_step):
                raw_session_info = prepare_session_data_raw(user_id, snapshot_idx, ground_truth, config, surgery=args.surgery)
                if raw_session_info is not None:
                    all_raw_session_infos.append(raw_session_info)
                    # Collect unique emergent subtopic texts for batch embedding
                    for subtopic in raw_session_info['valid_subtopics']:
                        all_unique_subtopics.add(subtopic['emergent_subtopic'])

        logging.info(f"Found {len(all_unique_subtopics)} unique emergent subtopic texts")
        logging.info(f"Found {len(all_gt_subtopics)} unique ground truth subtopic texts")

        # Step 2: Process sessions with threading (using pre-computed embedding index)
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []

            for raw_session_info in all_raw_session_infos:
                user_id = raw_session_info['user_id']
                snapshot_idx = raw_session_info['snapshot_idx']
                ground_truth = user_ground_truths[user_id]

                futures.append(
                    executor.submit(process_session_openai, user_id, snapshot_idx, ground_truth, config, surgery=args.surgery)
                )

            # Wait for all tasks to complete
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating sessions ({config.mode} mode)"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing session: {e}")


if __name__ == "__main__":
    main()
