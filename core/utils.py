import os
from collections import defaultdict

def extract_page_chunks(chunks):
    """Extract page-specific chunks from document chunks."""
    page_map = defaultdict(list)
    for chunk in chunks:
        if "p." in chunk.get("source", ""):
            try:
                page = int(chunk["source"].split("p.")[-1])
                content = chunk.get("content", "").strip()
                if content:
                    page_map[page].append(content)
            except (ValueError, IndexError):
                continue
    return dict(page_map)

def ensure_session_directory(session_id):
    """Ensure session-specific directory exists."""
    data_dir = os.path.join("data", session_id)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def safe_execute(func, fallback=None, error_msg="Operation failed"):
    """Safely execute a function with error handling."""
    try:
        return func()
    except Exception as e:
        print(f"⚠️ {error_msg}: {str(e)}")
        return fallback
