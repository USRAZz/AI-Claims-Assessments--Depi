"""
Shared Gemini configuration for all pipeline agents.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from agents import (
    AgentOutputSchema,
    ModelSettings,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    set_default_openai_api,
    set_default_openai_client,
)

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini using its OpenAI-compatible API
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_default_openai_client(client)
set_default_openai_api("chat_completions")

# Disable OpenAI tracing when using Gemini
set_tracing_disabled(True)

# Gemini model
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MODEL_SETTINGS = ModelSettings(temperature=0)


def output_schema(model_class):
    return AgentOutputSchema(
        model_class,
        strict_json_schema=False,
    )