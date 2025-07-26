from sentence_transformers import CrossEncoder
from sklearn.preprocessing import minmax_scale
import openai
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL, TEMPERATURE


class RerankerAgent:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.client = openai.OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )

    def normalize_scores(self, scores):
        return minmax_scale(scores)  # scale to [0, 1]

    def crossencoder_rerank(self, query, docs, top_k=10):
        if not docs:
            return []

        try:
            pairs = [(query, doc.get("content", "")) for doc in docs if doc.get("content")]
            if not pairs:
                return []

            scores = self.model.predict(pairs)
            normalized_scores = self.normalize_scores(scores)

            for i, doc in enumerate(docs):
                if i < len(normalized_scores):
                    doc["score"] = normalized_scores[i]
                else:
                    doc["score"] = 0.0

            return sorted(docs, key=lambda d: d.get("score", 0), reverse=True)[:top_k]
        except Exception as e:
            print(f"⚠️ CrossEncoder reranking failed: {str(e)}")
            return docs[:top_k]

    def llm_rerank_react(self, query, docs, top_k=5):
        if not docs:
            return []

        try:
            bullets = "\n".join([f"{i+1}. {doc.get('content', '')[:300]}" for i, doc in enumerate(docs)])
            react_prompt = f"""
You are a helpful assistant selecting the most relevant text chunks for the question.

Question: {query}

Chunks:
{bullets}

Think step-by-step and list the most relevant chunks by number (e.g., 1, 3, 4).

Answer:
"""

            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": react_prompt}],
                temperature=TEMPERATURE
            )
            content = response.choices[0].message.content
            selected_indices = [int(num.strip()) - 1 for num in content.split() if num.strip().isdigit()]
            selected = [docs[i] for i in selected_indices if 0 <= i < len(docs)]
            return selected[:top_k] if selected else docs[:top_k]
        except Exception as e:
            print(f"⚠️ LLM reranking failed: {str(e)}")
            return docs[:top_k]

    def rerank(self, query, docs, method="hybrid", top_k=5):
        if method == "crossencoder":
            return self.crossencoder_rerank(query, docs, top_k)
        elif method == "react":
            top_docs = self.crossencoder_rerank(query, docs, top_k=10)
            return self.llm_rerank_react(query, top_docs, top_k)
        elif method == "hybrid":
            top_docs = self.crossencoder_rerank(query, docs, top_k=20)
            return self.llm_rerank_react(query, top_docs, top_k)
        else:
            return docs[:top_k]
