from flask import Flask, render_template, request, session, redirect, url_for, flash
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-change-in-production')

# Config
# Default paths work for both local development (with .env overrides) and Cloud Run (with mounted bucket)
USERS_FILE = Path(os.getenv('USERS_FILE', '/app/data/data/users.json'))
INTERVIEW_DATA_ROOT = Path(os.getenv('INTERVIEW_DATA_ROOT', '/app/data'))
ANNOTATIONS_ROOT = Path(os.getenv('ANNOTATIONS_ROOT', '/app/data/annotations'))  # Save to GCS bucket
ANNOTATION_CONFIG_FILE = Path('annotation_config.json')

# Task 1: Coverage rubric (session-level)
COVERAGE_RUBRIC = [
    {
        "title": "Coverage",
        "description": "Considering all topics discussed in the interview, how comprehensively were the relevant aspects addressed overall?",
        "scale": {
            "1": "Major areas were not addressed at all",
            "2": "Only a few areas were touched on, with significant gaps",
            "3": "Key areas were addressed, but some important gaps remain",
            "4": "Most relevant areas were covered with minor omissions",
            "5": "All relevant areas were thoroughly and systematically covered"
        }
    },
    {
        "title": "Depth",
        "description": "How deeply did the interviewer explore your responses throughout the interview?",
        "scale": {
            "1": "No probing; responses were accepted at face value",
            "2": "Minimal probing; questions stayed mostly surface-level",
            "3": "Some probing with occasional follow-up questions",
            "4": "Consistent probing with meaningful follow-up questions",
            "5": "In-depth probing with sustained exploration, including nuances and trade-offs"
        }
    },
    {
        "title": "Correctness",
        "description": "How accurate and appropriate were the interviewer's summaries across all topics in the interview?",
        "scale": {
            "1": "Frequently inaccurate or misleading framing",
            "2": "Several inaccuracies or questionable assumptions",
            "3": "Generally accurate with minor issues",
            "4": "Accurate and well-framed throughout",
            "5": "Consistently precise, accurate, and contextually appropriate"
        }
    }
]

# Task 2: Emergent rubric (per-item)
EMERGENT_RUBRIC = [
    {
        "title": "Topical Relevance",
        "description": "How relevant were the emergent ideas or questions to the current topic and the participant's role?",
        "scale": {
            "1": "Irrelevant or distracting",
            "2": "Weakly relevant with limited connection",
            "3": "Somewhat relevant but not fully aligned",
            "4": "Mostly relevant and clearly connected",
            "5": "Highly relevant and directly aligned with the subtopic"
        }
    },
    {
        "title": "Topic-Level Emergence",
        "description": "How much did this discussion reveal novel or unexpected perspectives within this topic area? Did it go beyond obvious aspects to uncover new angles?",
        "scale": {
            "1": "Adds nothing new beyond the subtopic; not emergent",
            "2": "Adds minor new connections but largely redundant",
            "3": "Moderately emergent with some novel insight",
            "4": "Clearly emergent and adds meaningful new perspectives",
            "5": "Highly emergent, introducing substantial novel directions"
        }
    },
    {
        "title": "Surprise",
        "description": "How unexpected were the emergent ideas or questions relative to what had already been discussed?",
        "scale": {
            "1": "Entirely expected or routine",
            "2": "Slightly unexpected",
            "3": "Moderately unexpected",
            "4": "Highly unexpected",
            "5": "Strongly surprising while remaining relevant"
        }
    },
    {
        "title": "Analytical Value",
        "description": "How much does this emergent insight contribute to understanding AI workforce issues across the industry? What's the overall utility for research?",
        "scale": {
            "1": "Distracting or uninformative",
            "2": "Adds little analytical insight",
            "3": "Some insight but limited evaluative usefulness",
            "4": "Adds meaningful analytical insight",
            "5": "Substantially deepens understanding or evaluative power"
        }
    }
]

# Task 3: Overall interaction quality rubric (session-level)
OVERALL_RUBRIC = [
    {
        "title": "Clarity",
        "description": "How clear and easy was it to understand the interviewer's questions?",
        "scale": {
            "1": "Very unclear",
            "2": "Often unclear",
            "3": "Mostly clear",
            "4": "Clear",
            "5": "Exceptionally clear"
        }
    },
    {
        "title": "Adaptiveness",
        "description": "How well did the interviewer adapt to the candidate's responses and level?",
        "scale": {
            "1": "No adaptation",
            "2": "Minimal adaptation",
            "3": "Some adaptation",
            "4": "Well adapted",
            "5": "Highly adaptive"
        }
    },
    {
        "title": "Comfort & Fairness",
        "description": "How comfortable, fair, and respectful did the interview feel?",
        "scale": {
            "1": "Uncomfortable or unfair",
            "2": "Somewhat uncomfortable",
            "3": "Neutral",
            "4": "Comfortable and fair",
            "5": "Very comfortable and fair"
        }
    },
    {
        "title": "Experience",
        "description": "Overall quality of the interview experience.",
        "scale": {
            "1": "Very poor",
            "2": "Poor",
            "3": "Acceptable",
            "4": "Good",
            "5": "Excellent"
        }
    }
]

# Utility functions
def load_users():
    """Load users from interview app's users.json"""
    with open(USERS_FILE) as f:
        return json.load(f)

def hash_password(password):
    """Hash password using SHA-256 (same as interview app)"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_annotation_config():
    """Load annotation configuration"""
    if ANNOTATION_CONFIG_FILE.exists():
        with open(ANNOTATION_CONFIG_FILE) as f:
            return json.load(f)
    return {"annotation_permissions": {}, "session_filter": {"only_session_0": True}}

def get_allowed_sessions_for_user(username, config):
    """Return list of user_ids this annotator can annotate"""
    allowed = set()
    permissions = config.get('annotation_permissions', {})

    # Self-annotation
    if username in permissions.get('self_annotation', {}).get('users', []):
        allowed.add(username)

    # Cross-annotation
    cross = permissions.get('cross_annotation', {}).get(username, {})
    allowed.update(cross.get('can_annotate', []))

    return list(allowed)

def get_tasks_for_session(annotator, target_user, config):
    """Determine which tasks this annotator should complete for this target user"""
    permissions = config.get('annotation_permissions', {})

    # Self-annotation: all tasks
    if annotator == target_user:
        return permissions.get('self_annotation', {}).get('tasks', ['subtopics', 'emergent', 'overall'])

    # Cross-annotation: emergent only
    cross = permissions.get('cross_annotation', {}).get(annotator, {})
    if target_user in cross.get('can_annotate', []):
        return cross.get('tasks', ['emergent'])

    return []

def get_available_sessions(current_user=None, config=None):
    """Scan interview data directory and return list of sessions"""
    sessions = []
    logs_dir = INTERVIEW_DATA_ROOT / 'final_logs'

    if not logs_dir.exists():
        return sessions

    for user_dir in logs_dir.iterdir():
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name

        exec_logs = user_dir / 'execution_logs'
        if not exec_logs.exists():
            continue

        for session_dir in exec_logs.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                try:
                    session_id = int(session_dir.name.split('_')[1])
                    sessions.append({
                        'user_id': user_id,
                        'session_id': session_id,
                        'session_key': f"{user_id}:{session_id}",
                        'path': str(session_dir)
                    })
                except (IndexError, ValueError):
                    continue

    # Filter to session_0 only
    if config and config.get('session_filter', {}).get('only_session_0'):
        sessions = [s for s in sessions if s['session_id'] == 0]

    # Filter based on user permissions
    if current_user and config:
        allowed_user_ids = get_allowed_sessions_for_user(current_user, config)
        sessions = [s for s in sessions if s['user_id'] in allowed_user_ids]

    return sorted(sessions, key=lambda x: (x['user_id'], x['session_id']))

def load_session_snapshot(user_id, session_id):
    """Load latest snapshot for a session"""
    session_dir = INTERVIEW_DATA_ROOT / 'final_logs' / user_id / 'execution_logs' / f'session_{session_id}'

    if not session_dir.exists():
        return None

    # Find latest snapshot
    snapshots = list(session_dir.glob('session_agenda_snap_*.json'))
    if not snapshots:
        return None

    latest = max(snapshots, key=lambda p: int(p.stem.split('_')[-1]))

    with open(latest) as f:
        data = json.load(f)

    # Extract what we need
    topic_manager = data.get('interview_topic_manager', {})
    core_topics = topic_manager.get('core_topic_dict', {})

    # Get subtopics and emergent items
    subtopics = []
    emergent_items = []

    for topic_id, topic in sorted(core_topics.items()):
        topic_desc = topic.get('description', f'Topic {topic_id}')

        # Required subtopics
        for subtopic_id, subtopic in sorted(topic.get('required_subtopics', {}).items()):
            notes = subtopic.get('notes', [])
            summary = subtopic.get('final_summary', '')

            # Only include if there's content (notes OR summary)
            if notes or summary:
                subtopics.append({
                    'id': subtopic_id,
                    'topic': topic_desc,
                    'description': subtopic['description'],
                    'notes': notes,
                    'final_summary': summary,
                    'is_covered': subtopic.get('is_covered', False)
                })

        # Emergent subtopics - deduplicate by description
        seen_descriptions = set()
        for subtopic_id, subtopic in sorted(topic.get('emergent_subtopics', {}).items()):
            notes = subtopic.get('notes', [])
            summary = subtopic.get('final_summary', '')
            description = subtopic['description']

            # Only include if there's content AND not a duplicate description
            if (notes or summary) and description not in seen_descriptions:
                seen_descriptions.add(description)
                emergent_items.append({
                    'id': subtopic_id,
                    'topic': topic_desc,
                    'description': description,
                    'notes': notes,
                    'final_summary': summary,
                    'is_covered': subtopic.get('is_covered', False)
                })

    # Get coverage stats for overview
    coverage_stats = topic_manager.get('coverage_stats', {})
    total_topics = len(core_topics)

    return {
        'user_id': user_id,
        'session_id': session_id,
        'subtopics': subtopics,
        'emergent_items': emergent_items,
        'total_topics': total_topics,
        'coverage_stats': coverage_stats
    }

def save_annotations_batch(annotator, session_key, annotations_list):
    """Save all annotations to JSON file in a single atomic write"""
    # Create directory structure: /app/data/annotations/{annotator}/
    annotation_dir = ANNOTATIONS_ROOT / annotator
    annotation_dir.mkdir(parents=True, exist_ok=True)

    # Create filename: session_key with ':' replaced by '_'
    json_file = annotation_dir / f"{session_key.replace(':', '_')}.json"

    # Load existing annotations for this session
    if json_file.exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = {
            'annotator': annotator,
            'session_key': session_key,
            'timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'annotations': {
                'coverage': {},
                'emergent': {},
                'overall': {}
            }
        }

    # Update timestamp
    current_time = datetime.now().isoformat()
    data['last_updated'] = current_time

    # Process all annotations in batch
    for item_type, item_id, dimension, rating, comment in annotations_list:
        if item_type == 'coverage':
            if dimension == 'comment':
                data['annotations']['coverage']['comment'] = {'comment': comment, 'timestamp': current_time}
            else:
                data['annotations']['coverage'][dimension] = {'rating': rating, 'timestamp': current_time}

        elif item_type == 'emergent':
            if item_id == 'session':  # Session-level comment
                data['annotations']['emergent']['comment'] = {'comment': comment, 'timestamp': current_time}
            else:  # Per-item rating
                if item_id not in data['annotations']['emergent']:
                    data['annotations']['emergent'][item_id] = {}
                data['annotations']['emergent'][item_id][dimension] = {'rating': rating, 'timestamp': current_time}

        elif item_type == 'overall':
            if dimension == 'comment':
                data['annotations']['overall']['comment'] = {'comment': comment, 'timestamp': current_time}
            else:
                data['annotations']['overall'][dimension] = {'rating': rating, 'timestamp': current_time}

    # Write back to JSON atomically (write to temp file, then rename)
    temp_file = json_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Atomic rename (on POSIX systems, rename is atomic)
    temp_file.replace(json_file)

def get_annotated_sessions(annotator):
    """Get list of sessions already annotated by this user"""
    annotation_dir = ANNOTATIONS_ROOT / annotator

    if not annotation_dir.exists():
        return set()

    annotated = set()
    # List all JSON files in the user's annotation directory
    for json_file in annotation_dir.glob('*.json'):
        # Convert filename back to session_key (replace last '_' with ':')
        # Since filenames are always {user_id}_{session_num}.json, replace last underscore
        stem = json_file.stem
        # Use rsplit to split from right, ensuring we only replace the delimiter underscore
        if '_' in stem:
            parts = stem.rsplit('_', 1)  # Split from right, max 1 split
            session_key = ':'.join(parts)  # Rejoin with ':'
        else:
            session_key = stem
        annotated.add(session_key)

    return annotated

def load_previous_annotations(annotator, session_key):
    """Load previous annotations for this annotator and session from JSON"""
    json_file = ANNOTATIONS_ROOT / annotator / f"{session_key.replace(':', '_')}.json"

    if not json_file.exists():
        return {}

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # Corrupted JSON file - log warning and return empty dict
        # This allows user to re-annotate and overwrite corrupted file
        print(f"WARNING: Corrupted JSON file for {annotator}/{session_key}: {e}")
        print(f"File path: {json_file}")
        print(f"User can re-annotate to overwrite corrupted file")
        return {}
    except Exception as e:
        # Catch any other errors (permission, I/O, etc.)
        print(f"ERROR: Failed to load annotations for {annotator}/{session_key}: {e}")
        return {}

    previous = {}

    # Parse coverage annotations
    for dimension, value in data['annotations'].get('coverage', {}).items():
        if dimension == 'comment':
            key = f"coverage:session:comment"
            previous[key] = {'comment': value.get('comment', ''), 'timestamp': value.get('timestamp', '')}
        else:
            key = f"coverage:session:{dimension}"
            previous[key] = {'rating': str(value.get('rating', '')), 'timestamp': value.get('timestamp', '')}

    # Parse emergent annotations
    for item_id, dimensions in data['annotations'].get('emergent', {}).items():
        if item_id == 'comment':
            key = f"emergent:session:comment"
            previous[key] = {'comment': dimensions.get('comment', ''), 'timestamp': dimensions.get('timestamp', '')}
        else:
            for dimension, value in dimensions.items():
                key = f"emergent:{item_id}:{dimension}"
                previous[key] = {'rating': str(value.get('rating', '')), 'timestamp': value.get('timestamp', '')}

    # Parse overall annotations
    for dimension, value in data['annotations'].get('overall', {}).items():
        if dimension == 'comment':
            key = f"overall:session:comment"
            previous[key] = {'comment': value.get('comment', ''), 'timestamp': value.get('timestamp', '')}
        else:
            key = f"overall:session:{dimension}"
            previous[key] = {'rating': str(value.get('rating', '')), 'timestamp': value.get('timestamp', '')}

    return previous

# Routes
@app.route('/')
def index():
    """Redirect to login or sessions"""
    if 'username' in session:
        return redirect(url_for('sessions_list'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please provide both username and password.', 'error')
            return render_template('login.html')

        users = load_users()

        # Find user by username
        user_id = None
        for uid, user_data in users.items():
            if user_data['username'] == username:
                user_id = uid
                break

        if not user_id:
            flash('Invalid username or password.', 'error')
            return render_template('login.html')

        # Verify password
        password_hash = hash_password(password)
        if users[user_id]['password'] != password_hash:
            flash('Invalid username or password.', 'error')
            return render_template('login.html')

        # Login successful
        session['username'] = username
        session['user_id'] = user_id
        flash(f'Welcome, {username}!', 'success')
        return redirect(url_for('sessions_list'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout handler"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/sessions')
def sessions_list():
    """List all available sessions"""
    if 'username' not in session:
        return redirect(url_for('login'))

    config = load_annotation_config()
    sessions = get_available_sessions(current_user=session['user_id'], config=config)
    annotated = get_annotated_sessions(session['user_id'])

    # Mark which sessions are already annotated
    for s in sessions:
        s['annotated'] = s['session_key'] in annotated

    return render_template('sessions.html', sessions=sessions)

@app.route('/annotate/<session_key>')
def annotate(session_key):
    """Annotation page for a session"""
    if 'username' not in session:
        return redirect(url_for('login'))

    config = load_annotation_config()

    # Parse session key
    try:
        user_id, session_id = session_key.split(':')
        session_id = int(session_id)
    except (ValueError, AttributeError):
        flash('Invalid session key.', 'error')
        return redirect(url_for('sessions_list'))

    # Check permissions
    tasks_to_show = get_tasks_for_session(session['user_id'], user_id, config)
    if not tasks_to_show:
        flash('You do not have permission to annotate this session.', 'error')
        return redirect(url_for('sessions_list'))

    # Load session data
    session_data = load_session_snapshot(user_id, session_id)
    if not session_data:
        flash(f'Session {session_key} not found.', 'error')
        return redirect(url_for('sessions_list'))

    # Load previous annotations if they exist
    previous_annotations = load_previous_annotations(session['user_id'], session_key)

    return render_template('annotate.html',
                         session_data=session_data,
                         session_key=session_key,
                         tasks_to_show=tasks_to_show,
                         is_self_annotation=(session['user_id'] == user_id),
                         coverage_rubric=COVERAGE_RUBRIC,
                         emergent_rubric=EMERGENT_RUBRIC,
                         overall_rubric=OVERALL_RUBRIC,
                         previous_annotations=previous_annotations)

@app.route('/submit', methods=['POST'])
def submit():
    """Handle annotation form submission"""
    if 'username' not in session:
        return redirect(url_for('login'))

    session_key = request.form.get('session_key')
    if not session_key:
        flash('No session key provided.', 'error')
        return redirect(url_for('sessions_list'))

    annotator = session['user_id']

    # Collect all annotations first, then save in one batch
    annotations_list = []

    for key, value in request.form.items():
        if not value or key == 'session_key':
            continue

        try:
            # Handle Task 1 comments (session-level)
            if key == 'task1_comments':
                annotations_list.append(('coverage', 'session', 'comment', '', value))

            # Handle Task 1 session-level ratings: task1_coverage, task1_depth, task1_correctness
            elif key.startswith('task1_'):
                dimension = key.replace('task1_', '')  # Extract: coverage, depth, correctness
                annotations_list.append(('coverage', 'session', dimension, value, ''))

            # Handle Task 2 comments (session-level)
            elif key == 'task2_comments':
                annotations_list.append(('emergent', 'session', 'comment', '', value))

            # Handle Task 2 per-item ratings: task2_{item_id}_accuracy, task2_{item_id}_plausibility, etc.
            elif key.startswith('task2_'):
                parts = key.split('_', 2)  # Split into ['task2', 'item_id', 'dimension']
                item_id = parts[1]
                dimension = parts[2]  # accuracy, plausibility, surprise, value
                annotations_list.append(('emergent', item_id, dimension, value, ''))

            # Handle Task 3 comments (session-level)
            elif key == 'task3_comments' or key == 'overall_comments':
                annotations_list.append(('overall', 'session', 'comment', '', value))

            # Handle Task 3 overall ratings: overall_clarity, overall_adaptiveness, etc.
            elif key.startswith('overall_'):
                dimension = key.replace('overall_', '')
                annotations_list.append(('overall', 'session', dimension, value, ''))

        except Exception as e:
            print(f"Error parsing {key}: {e}")
            continue

    # Save all annotations in a single atomic write
    try:
        save_annotations_batch(annotator, session_key, annotations_list)
        flash(f'Annotations for session {session_key} saved successfully!', 'success')
    except Exception as e:
        print(f"Error saving annotations: {e}")
        flash(f'Error saving annotations: {e}', 'error')

    return redirect(url_for('sessions_list'))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
