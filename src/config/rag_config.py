from src.config import settings

# RAG settings. The RAG tool retrieves from a Qdrant collection .
# You can populate it with test data using the provided script (scripts/seed_qdrant.py loads a few sample documents to get started).
# The values below must match how that collection was built.


# Connection, set these in your .env file
QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
COLLECTION_NAME = settings.QDRANT_COLLECTION

# Embedding model, called through OpenRouter (src/services/embeddings.py).
# Must be the same model used to embed your documents, otherwise the query
# vectors won't match. Models: https://openrouter.ai/models?output_modalities=embeddings
EMBEDDING_MODEL = "openai/text-embedding-3-small"
# Vector size of EMBEDDING_MODEL. The collection must be created with this size.
EMBEDDING_DIMENSIONS = 1536

# Name of the vector in your collection. Use None if the
# collection has a single unnamed vector.
VECTOR_NAME: str | None = None

# Payload field that holds the chunk text.
# Common values: "text", "document", LangChain uses "page_content".
TEXT_PAYLOAD_KEY = "text"

# Reranker model and how many chunks to fetch / keep
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_TOP_K = 12
RERANK_TOP_N = 4
