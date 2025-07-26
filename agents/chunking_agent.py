from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.utils import safe_execute
from typing import List

# Optional: placeholder for semantic and markdown-specific splitters
class MarkdownTextSplitter:
    def split_text(self, text):
        return text.split("\n\n")

class CSVChunkSplitter:
    def split_text(self, text):
        return text.split("\n\n")  # Placeholder; implement column-based chunking later

class ChunkingAgent:
    def __init__(self):
        self.default_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    def chunk(self, text: str, format: str = "txt", query: str | None = None) -> List[str]:
        if format == "md" or format == "markdown":
            splitter = MarkdownTextSplitter()
        elif format == "csv":
            splitter = CSVChunkSplitter()
        else:
            splitter = self.default_splitter

        # Optional: Modify splitter dynamically based on query (coming later)
        return splitter.split_text(text)
