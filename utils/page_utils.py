from collections import defaultdict

def extract_page_chunks(chunks):
    page_map = defaultdict(list)
    for chunk in chunks:
        if not isinstance(chunk, dict):
            print(f"⚠️ Skipping non-dict chunk: {type(chunk)} - {str(chunk)[:100]}")
            continue
        if "p." in chunk.get("source", ""):
            try:
                page = int(chunk["source"].split("p.")[-1])
                content = chunk.get("content", "").strip()
                if content:
                    page_map[page].append(content)
            except (ValueError, IndexError):
                continue
    return dict(page_map)
