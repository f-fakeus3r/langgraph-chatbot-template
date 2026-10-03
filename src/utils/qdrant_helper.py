from functools import lru_cache

from qdrant_client import QdrantClient

from src.config.rag_config import (
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    RETRIEVAL_TOP_K,
    TEXT_PAYLOAD_KEY,
    VECTOR_NAME,
)
from src.services.embeddings import embed_query


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL is not set. Add it to your .env file.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def search(query: str, limit: int = RETRIEVAL_TOP_K) -> list[str]:
    """Return the text of the most similar chunks in the collection.
    The query is embedded with the same model as the documents."""
    points = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=embed_query(query),
        using=VECTOR_NAME,
        limit=limit,
        with_payload=True,
    ).points

    return [p.payload[TEXT_PAYLOAD_KEY] for p in points
            if p.payload and TEXT_PAYLOAD_KEY in p.payload]
