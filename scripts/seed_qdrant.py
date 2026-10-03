"""
This script was created with Claude Code.

Seed the Qdrant collection with a few sample documents, so the RAG tool has
something to retrieve. Replace SAMPLE_DOCUMENTS with your own data, or use
your own ingestion pipeline. It uses the settings in src/config/rag_config.py,
so the documents are stored exactly the way qdrant_helper.search reads them:
embedded with EMBEDDING_MODEL (through OpenRouter) into EMBEDDING_DIMENSIONS-size vectors.
If the collection already exists with a different vector size, the script stops
before writing anything.

Usage, from the project root:
    uv run python -m scripts.seed_qdrant             # create the collection if missing, add the documents
    uv run python -m scripts.seed_qdrant --recreate  # delete the collection first (erases its data!)
"""
import argparse
import uuid

from qdrant_client import models

from src.config.rag_config import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    TEXT_PAYLOAD_KEY,
    VECTOR_NAME,
)
from src.services.embeddings import embed_texts
from src.utils.qdrant_helper import get_qdrant_client

# PLACEHOLDER: sample documents, replace them with your own
SAMPLE_DOCUMENTS = [
    (
        "This assistant is built with LangGraph. Every message goes through an input "
        "guardrail, then a classifier picks the agent that answers, and an output "
        "guardrail reviews the answer before it is sent to the user."
    ),
    (
        "Conversation memory is stored in Supabase Postgres. Each session id is a "
        "separate conversation, so the assistant remembers the previous messages of "
        "the same session."
    ),
    (
        "The guardrails block unsafe requests and unsafe answers. When a message or "
        "an answer is blocked, the assistant replies with a short, safe message instead."
    ),
    (
        "Answers about the documents come from a knowledge base stored in Qdrant. The "
        "most relevant passages are retrieved and reranked before the assistant answers."
    ),
    (
        "Developers can extend the assistant with new tools, new agents, or new "
        "subgraphs. Each agent keeps its own conversation history and system prompt."
    ),
    (
        "The language model is provided through OpenRouter, and it can be replaced by "
        "any chat model supported by LangChain."
    ),
]


def _vector(embedding: list[float]) -> list[float] | dict[str, list[float]]:
    return {VECTOR_NAME: embedding} if VECTOR_NAME else embedding


def _collection_size(client) -> int | None:
    """Vector size of the existing collection, or None if it has no vector named VECTOR_NAME."""
    vectors = client.get_collection(COLLECTION_NAME).config.params.vectors
    if isinstance(vectors, dict):
        # Named vectors; the default unnamed vector is listed under ""
        params = vectors.get(VECTOR_NAME or "")
        return params.size if params else None
    return vectors.size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the Qdrant collection with sample documents.")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete the collection before seeding. Erases all of its data.")
    args = parser.parse_args()

    embeddings = embed_texts(SAMPLE_DOCUMENTS)
    if len(embeddings[0]) != EMBEDDING_DIMENSIONS:
        raise SystemExit(
            f"{EMBEDDING_MODEL} returned {len(embeddings[0])}-dim vectors, but EMBEDDING_DIMENSIONS "
            f"is {EMBEDDING_DIMENSIONS}. Fix EMBEDDING_DIMENSIONS in src/config/rag_config.py.")

    client = get_qdrant_client()

    if args.recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted collection '{COLLECTION_NAME}'")

    if not client.collection_exists(COLLECTION_NAME):
        vector_params = models.VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=models.Distance.COSINE,
        )
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                VECTOR_NAME: vector_params} if VECTOR_NAME else vector_params,
        )
        print(f"Created collection '{COLLECTION_NAME}'")

    collection_size = _collection_size(client)
    if collection_size != EMBEDDING_DIMENSIONS:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' stores {collection_size}-dim vectors, but {EMBEDDING_MODEL} "
            f"makes {EMBEDDING_DIMENSIONS}-dim ones. Run with --recreate (erases the collection's data), "
            "or change EMBEDDING_MODEL / VECTOR_NAME in src/config/rag_config.py.")

    points = [
        models.PointStruct(
            # Same text -> same id, so running the script twice doesn't duplicate points
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, text)),
            vector=_vector(embedding),
            payload={TEXT_PAYLOAD_KEY: text},
        )
        for text, embedding in zip(SAMPLE_DOCUMENTS, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} documents into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
