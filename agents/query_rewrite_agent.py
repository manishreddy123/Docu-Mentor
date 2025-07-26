"""rewrite the classify intent program 
much better to get the intent much better
and try to add more classifications in the
program to get more different type of classifications
"""

import openai
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL, TEMPERATURE

class QueryRewriteAgent:
    def __init__(self):
        try:
            if OPENROUTER_API_KEY:
                self.client = openai.OpenAI(
                    api_key=OPENROUTER_API_KEY,
                    base_url=OPENROUTER_BASE_URL
                )
            else:
                self.client = None
                print("Warning: OpenAI client not initialized due to missing API key. Query rewriting will be disabled.")
        except Exception as e:
            self.client = None
            print(f"Warning: Failed to initialize OpenAI client: {str(e)}. Query rewriting will be disabled.")
        self.few_shot_examples = [
            {
                "input": "Show me the results.",
                "output": "What are the financial metrics for Q4 2023 in the uploaded reports?"
            },
            {
                "input": "How was it last year?",
                "output": "What was the revenue in FY2022 according to the annual summary PDF?"
            },
            {
                "input": "Compare performance.",
                "output": "How did customer acquisition cost (CAC) change between Q1 and Q2?"
            }
        ]

    def classify_intent(self, query: str) -> str:
        if any(keyword in query.lower() for keyword in ["how many", "total", "count", "amount"]):
            return "statistical"
        elif any(keyword in query.lower() for keyword in ["why", "cause", "reason"]):
            return "causal"
        elif any(keyword in query.lower() for keyword in ["compare", "difference", "trend"]):
            return "comparative"
        else:
            return "factual"

    def extract_doc_type_preferences(self, query: str) -> list:
        """Extract document type preferences from query for retrieval filtering"""
        preferences = []
        doc_type_keywords = {
            "pdf": ["pdf", "document", "report", "paper"],
            "csv": ["csv", "spreadsheet", "data", "excel"],
            "txt": ["text", "note", "readme"],
            "markdown": ["markdown", "md", "documentation", "readme"],
            "docx": ["word", "docx", "document"],
            "pptx": ["powerpoint", "ppt", "presentation", "slide"]
        }
        
        query_lower = query.lower()
        for doc_type, keywords in doc_type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                preferences.append(doc_type)
        
        return preferences

    def rewrite(self, query: str) -> str:
        if not self.client:
            print("⚠️ Query rewriting skipped: OpenAI client not available")
            return query
            
        try:
            # Get personalized context based on current query
            history = ""
            examples = "\n".join([
                f"Q: {ex['input']}\nRewritten: {ex['output']}" for ex in self.few_shot_examples
            ])
            intent = self.classify_intent(query)
            
            # Extract document type preferences from query
            doc_type_preferences = self.extract_doc_type_preferences(query)

            prompt = f"""
You are a retrieval expert. Given the user question and personalized chat history, rewrite the question to optimize retrieval from uploaded documents.

Examples:
{examples}

Intent: {intent}
Document Type Preferences: {', '.join(doc_type_preferences) if doc_type_preferences else 'None specified'}

Personalized History (relevant to this query):
{history}

Current Query: {query}

Rewritten:
"""

            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Query rewriting failed: {str(e)}")
            return query
