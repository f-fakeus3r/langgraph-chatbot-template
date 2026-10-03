from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.config.rag_config import RERANK_TOP_N, RERANKER_MODEL


class FastEmbedReranker:
    """Reranks documents with a fastembed (ONNX) cross-encoder model."""

    def __init__(self, model: str = RERANKER_MODEL, top_n: int = RERANK_TOP_N) -> None:
        self.model = model
        self.top_n = top_n
        self._cross_encoder: TextCrossEncoder | None = None

    def _get_encoder(self) -> TextCrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = TextCrossEncoder(model_name=self.model)
        return self._cross_encoder

    def rerank(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        """Return the top_n (document, score) pairs, best first."""
        if not documents:
            return []

        scores = self._get_encoder().rerank(query, documents)
        ranked = sorted(zip(documents, map(float, scores)),
                        key=lambda pair: pair[1], reverse=True)
        return ranked[: self.top_n]
