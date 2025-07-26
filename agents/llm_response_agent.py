import openai
from core.mcp import create_mcp_message
from tenacity import retry, wait_exponential, stop_after_attempt
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, AVAILABLE_MODELS, TEMPERATURE


class LLMResponseAgent:
    def __init__(self):
        self.name = "LLMResponseAgent"
        self.models = AVAILABLE_MODELS
        self.client = openai.OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )

    def build_prompt(self, query, chunks, memory_context="", format_type="markdown", cot=True):
        context = "\n\n".join([
            f"### Source: {c.get('source', 'unknown')}\n{c['content']}" for c in chunks
        ])
        if memory_context:
            context = f"📋 Relevant Chat History:\n{memory_context}\n\n{context}"
        instruction = "Answer the following question using the provided context below."
        if cot:
            instruction += " Think step by step and explain your reasoning clearly."
        if format_type == "table":
            instruction += " Present your answer in a Markdown table if possible."
        elif format_type == "json":
            instruction += " Return the answer in JSON format."
        elif format_type == "list":
            instruction += " Return the answer as a bullet-point list."

        return f"""{instruction}

📚 Context:
{context}

❓ Question: {query}

💡 Answer:"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_llm(self, prompt, model):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content

    def guardrails_check(self, response: str) -> bool:
        red_flags = ["I'm not sure", "I cannot find", "no information", "not available", "hallucination"]
        return any(flag.lower() in response.lower() for flag in red_flags)

    def handle(self, message):
        try:
            payload = message.get("payload", {})
            query = payload.get("query", "")
            chunks = payload.get("retrieved_context", [])

            # Debug logging
            print(f"🔍 LLMResponseAgent: Received {len(chunks)} chunks for query: '{query}'")
            if chunks:
                print(f"📄 Sample chunk: {chunks[0].get('content', '')[:100]}...")
            else:
                print("⚠️ LLMResponseAgent: No chunks found!")

            if not query:
                return "⚠️ No query provided."

            memory_context = ""
            
            # Build enriched prompt
            prompt = self.build_prompt(query, chunks, memory_context=memory_context)

            # Try models in fallback order
            for model in self.models:
                if not model:
                    continue

                try:
                    response = self.call_llm(prompt, model)
                    if response and not self.guardrails_check(response):
                        return response
                except Exception as e:
                    print(f"⚠️ Model {model} failed: {str(e)}")
                    continue

            return "⚠️ Sorry, all LLMs failed or did not produce a confident response."
        except Exception as e:
            print(f"⚠️ LLMResponseAgent.handle failed: {str(e)}")
            return "⚠️ An error occurred while processing your request."

