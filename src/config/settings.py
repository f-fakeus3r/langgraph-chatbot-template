import os

from dotenv import load_dotenv

# Every environment variable the app reads is listed here, so there's a single
# place to look. Copy .env.example to .env and fill it in.

load_dotenv()

# FastEmbed downloads its models from Hugging Face.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Supabase Postgres: stores the conversation memory (LangGraph checkpoints)
SUPABASE_POSTGRES_URI = os.getenv("SUPABASE_POSTGRES_URI")
POSTGRES_POOL_MIN_SIZE = int(os.getenv("POSTGRES_POOL_MIN_SIZE", "1"))
POSTGRES_POOL_MAX_SIZE = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10"))

# OpenRouter LLM Provider -> You may change this to your own provider of choice.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")  # or "your-default-model"
# Guardrails and routing should use smaller models to save tokens
LLM_GUARDRAIL_MODEL = os.getenv("LLM_GUARDRAIL_MODEL") or LLM_MODEL
LLM_ROUTER_MODEL = os.getenv("LLM_ROUTER_MODEL") or LLM_MODEL

# Qdrant -> vector database used by the RAG tool
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION") or "example_collection"

LOG_LEVEL = os.getenv("LOG_LEVEL") or "INFO"


def check_required_settings() -> None:
    required = {
        "SUPABASE_POSTGRES_URI": SUPABASE_POSTGRES_URI,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
