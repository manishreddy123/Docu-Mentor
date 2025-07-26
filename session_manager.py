import os
import pickle
from core.config_manager import ConfigManager


def save_sessions(session_groups):
    """Persist all session groups and their chat history to disk."""
    try:
        with open(ConfigManager.SESSION_FILE, "wb") as f:
            pickle.dump(session_groups, f)
    except Exception as e:
        print(f"⚠️ Failed to save sessions: {str(e)}")


import time
from threading import Timer

SESSION_FILE = "session_store.pkl"
_save_timer = None
_pending_sessions = None

def _delayed_save():
    """Internal function to perform the actual save operation."""
    global _pending_sessions
    if _pending_sessions is not None:
        with open(SESSION_FILE, "wb") as f:
            pickle.dump(_pending_sessions, f)
        _pending_sessions = None

def save_sessions(session_groups, delay=2.0):
    """Persist all session groups and their chat history to disk with debouncing."""
    global _save_timer, _pending_sessions
    
    _pending_sessions = session_groups
    
    if _save_timer is not None:
        _save_timer.cancel()
    
    _save_timer = Timer(delay, _delayed_save)
    _save_timer.start()


def load_sessions():
    """Load previously saved session groups from disk."""
    try:
        if os.path.exists(ConfigManager.SESSION_FILE):
            with open(ConfigManager.SESSION_FILE, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load sessions: {str(e)}")
    return {}


def cleanup_old_sessions(session_groups, max_age_days=30):
    """Remove sessions older than max_age_days."""
    import time
    from datetime import datetime, timedelta

    try:
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cleaned_sessions = {}
        
        for session_id, session_data in session_groups.items():
            session_time = session_data.get('timestamp', datetime.now())
            if isinstance(session_time, str):
                continue  # Skip malformed timestamps
            if session_time > cutoff_time:
                cleaned_sessions[session_id] = session_data

        removed_count = len(session_groups) - len(cleaned_sessions)
        if removed_count > 0:
            print(f"✅ Cleaned up {removed_count} old sessions")

        return cleaned_sessions
    except Exception as e:
        print(f"⚠️ Session cleanup failed: {str(e)}")
        return session_groups


def validate_and_cleanup_sessions(session_groups):
    """Remove sessions that reference missing document files."""
    import os
    
    valid_sessions = {}
    cleaned_count = 0
    
    for session_id, session_data in session_groups.items():
        session_dir = f"data/{session_id}"
        files = session_data.get('files', [])
        
        if os.path.exists(session_dir):
            missing_files = []
            for filename in files:
                file_path = os.path.join(session_dir, filename)
                if not os.path.exists(file_path):
                    missing_files.append(filename)
            
            if not missing_files:  # All files exist
                valid_sessions[session_id] = session_data
            else:
                cleaned_count += 1
                print(f"⚠️ Removed session {session_id} - missing files: {missing_files}")
        else:
            cleaned_count += 1
            print(f"⚠️ Removed session {session_id} - directory missing")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned up {cleaned_count} sessions with missing files")
    
    return valid_sessions
