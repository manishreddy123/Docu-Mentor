"""
Centralized configuration for OpenRouter API and models.
All configuration values are loaded from environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/openai/gpt-4")
MODELS = {
    "gpt4": os.getenv("MODEL_1"),
    "claude3": os.getenv("MODEL_2"),
    "grok": os.getenv("MODEL_3")
}

# API Configuration
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))

# Validate required configuration (optional for development)
if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY environment variable not set. LLM functionality will be limited.")
    OPENROUTER_API_KEY = "dev-mode"

# Available models list for fallback
AVAILABLE_MODELS = [
    MODELS["gpt4"],
    MODELS["claude3"],
    MODELS["grok"]
]
