class PromptFormatterAgent:
    def __init__(self):
        self.default_task = "qa"  # options: qa, compare, summarize, extract

    def format_metadata(self, chunk):
        return f"""**File**: {chunk.get("source", "unknown")}
**Type**: {chunk.get("type", "txt")}
**Rank Score**: {round(chunk.get("score", 1.0), 2)}"""

    def format_chunk(self, chunk):
        meta = self.format_metadata(chunk)
        return f"""{meta}
---
{chunk['content'].strip()}
"""

    def format(self, query, chunks, task_type=None, memory_context=None):
        if not chunks:
            return "No relevant documents found to answer the query."

        task = task_type or self.default_task

        # Sort chunks by rank score descending
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 1.0), reverse=True)

        formatted_chunks = "\n\n".join([self.format_chunk(c) for c in sorted_chunks])

        instruction_block = self.build_instruction_prompt(task, query)

        # Add memory context if provided
        memory_section = ""
        if memory_context:
            memory_section = f"\n\n📋 **Relevant Chat History:**\n{memory_context}\n"

        return f"""{instruction_block}{memory_section}

{formatted_chunks}
"""

    def build_instruction_prompt(self, task, query):
        if task == "qa":
            return f"""You are a document QA agent. Answer the question below using the provided context. Cite relevant sources if possible.\n\nQuestion: {query}\n"""
        elif task == "compare":
            return f"""Compare and contrast the insights across different documents related to:\n\nTopic: {query}\n"""
        elif task == "summarize":
            return f"""Summarize the main insights from the following document segments, grouped by source if possible.\n\nQuery: {query}\n"""
        elif task == "table":
            return f"""Extract the following key metrics and return in a Markdown table:\n\nQuery: {query}\n"""
        elif task == "extract":
            return f"""Extract all facts related to:\n\nQuery: {query}\n"""
        else:
            return f"""Answer the following query using the documents below:\n\nQuery: {query}\n"""
