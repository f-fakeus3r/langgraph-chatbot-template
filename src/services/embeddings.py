from functools import cache

from openrouter import OpenRouter

from src.config.rag_config import EMBEDDING_MODEL
from src.config.settings import OPENROUTER_API_KEY

"""
Embeddings for the RAG tool, through OpenRouter's embeddings API (same API key
as the chat models). Change these functions for your own provider. Documents
and queries must be embedded with the same model, see src/config/rag_config.py.
"""


@cache
def _client() -> OpenRouter:
    return OpenRouter(api_key=OPENROUTER_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _client().embeddings.generate(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
