"""Second retrieval stage: cross-encoder reranking.

Stage 1 (hybrid) is cheap and optimizes recall: ~30 candidates with the answer somewhere inside.
Stage 2 (rerank) is expensive and optimizes precision: it reads query and chunk *together*
and reorders, keeping only the best top_k.
"""

from dataclasses import replace

from ragkit.providers.base import RerankProvider
from ragkit.retrieval.hybrid import Hit


def rerank(question: str, hits: list[Hit], reranker: RerankProvider, top_k: int = 5) -> list[Hit]:
    order = reranker.rerank(question, [h.text for h in hits], top_n=top_k)
    return [
        replace(hits[idx], score=score, ranks={**hits[idx].ranks, "rerank": pos})
        for pos, (idx, score) in enumerate(order, start=1)
    ]
