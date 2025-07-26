# streaming_response.py
import time

def stream_response(text, delay=0.03):
    """Yields words one-by-one to simulate streaming."""
    for word in text.split():
        yield word + " "
        time.sleep(delay)
