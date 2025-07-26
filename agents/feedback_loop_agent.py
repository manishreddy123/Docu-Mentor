import openai
import json
import os
from datetime import datetime
from agents.query_rewrite_agent import QueryRewriteAgent
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL, TEMPERATURE


class FeedbackLoopAgent:
    def __init__(self, log_dir="logs", query_agent=None):
        self.query_agent = query_agent or QueryRewriteAgent()
        try:
            if OPENROUTER_API_KEY:
                self.client = openai.OpenAI(
                    api_key=OPENROUTER_API_KEY,
                    base_url=OPENROUTER_BASE_URL
                )
            else:
                self.client = None
                print("Warning: OpenAI client not initialized due to missing API key. Feedback loop functionality will be disabled.")
        except Exception as e:
            self.client = None
            print(f"Warning: Failed to initialize OpenAI client: {str(e)}. Feedback loop functionality will be disabled.")
        self.feedback_csv_path = os.path.join(log_dir, "feedback.csv")
        self.label_jsonl_path = os.path.join(log_dir, "label_dataset.jsonl")
        os.makedirs(log_dir, exist_ok=True)

    def log_feedback(self, trace_id, query, answer, user_rating):
        # CSV log for quick viewing
        with open(self.feedback_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{trace_id},{query},{answer},{user_rating}\n")

        # JSONL log for training
        entry = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "user_rating": user_rating
        }
        with open(self.label_jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def grade_with_llm(self, query, answer):
        if not self.client:
            print("⚠️ LLM grading skipped: OpenAI client not available")
            return 3  # fallback score
            
        grading_prompt = f"""
You are an expert evaluator. Grade the assistant's answer on a scale of 1 to 5, based on relevance, accuracy, and completeness.

Query: {query}

Answer: {answer}

Score (just the number from 1-5):
"""

        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": grading_prompt}],
            temperature=0
        )
        score_text = response.choices[0].message.content.strip()
        try:
            score = int(score_text)
        except:
            score = 3  # fallback
        return score

    def auto_correct(self, query, answer, regenerate_callback):
        score = self.grade_with_llm(query, answer)
        if score <= 2:
            print("🛠️ Auto-correcting low-quality answer...")
            revised_query = self.query_agent.rewrite(query + " (be more specific)")
            return regenerate_callback(revised_query)
        return answer
