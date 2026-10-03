import logging

from langchain_core.tools import tool

from src.utils.fastembed_reranker import FastEmbedReranker
from src.utils.qdrant_helper import search

logger = logging.getLogger(__name__)

_reranker = FastEmbedReranker()


def _format_results(reranked: list[tuple[str, float]]) -> str:
    # Change how the retrieved chunks are shown to the LLM,
    return "\n\n---\n\n".join(text for text, _score in reranked)


@tool
def qdrant_vector_search(query: str) -> str:
    """Search the knowledge base for information relevant to the query.
    Replace this docstring with a description of your documents, so the LLM
    knows when to use this tool."""
    try:
        candidates = search(query)
        reranked = _reranker.rerank(query, candidates)
    except Exception:
        logger.exception("qdrant_vector_search failed")
        return "The knowledge base is unavailable right now."

    if not reranked:
        return "No relevant information found."
    return _format_results(reranked)
