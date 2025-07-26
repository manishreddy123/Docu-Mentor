from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from agents.query_rewrite_agent import QueryRewriteAgent
from agents.prompt_formatter_agent import PromptFormatterAgent

class AgentManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.ingestion_agent = IngestionAgent()
            self.retrieval_agent = RetrievalAgent()
            self.llm_agent = LLMResponseAgent()
            self.query_agent = QueryRewriteAgent()
            self.formatter_agent = PromptFormatterAgent()
            self._initialized = True

    def get_agents(self):
        return {
            'ingestion': self.ingestion_agent,
            'retrieval': self.retrieval_agent,
            'llm': self.llm_agent,
            'query': self.query_agent,
            'formatter': self.formatter_agent
        }
