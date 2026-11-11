"""Hybrid retrieval: dense (meaning) + sparse (BM25 keywords), fused with Reciprocal Rank Fusion.

RRF ignores raw scores (cosine and BM25 live on incomparable scales) and uses only rank:

    rrf(doc) = sum over lists containing doc of  weight / (k + rank)

Documents ranked well by both searches rise; a document only one search found still enters.
k=60 comes from the original paper (Cormack et al., 2009) and is the industry default.
"""

from dataclasses import dataclass, field

from ragkit.providers.base import EmbeddingProvider
from ragkit.retrieval.query_hints import QueryHinter
from ragkit.retrieval.sparse import BM25Encoder

RRF_K = 60


def rrf(
    rankings: dict[str, list[str]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Returns (id, score) sorted best first."""
    scores: dict[str, float] = {}
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for pos, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + pos)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class Hit:
    chunk_id: str
    score: float
    metadata: dict
    ranks: dict[str, int] = field(default_factory=dict)  # rank in each ranked list

    @property
    def text(self) -> str:
        return self.metadata.get("text", "")


def _and(base: dict | None, extra: dict) -> dict:
    return {"$and": [base, extra]} if base else extra


class HybridRetriever:
    """All collaborators are injected, so tests use fakes and production wires real services."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        bm25: BM25Encoder,
        dense_index,
        sparse_index,
        namespace: str = "",
        hinter: QueryHinter | None = None,
    ):
        self.embedder = embedder
        self.bm25 = bm25
        self.dense = dense_index
        self.sparse = sparse_index
        self.namespace = namespace
        self.hinter = hinter

    def dense_search(self, qv: list[float], n: int, flt: dict | None = None):
        return self.dense.query(
            vector=qv, top_k=n, include_metadata=True, filter=flt, namespace=self.namespace
        ).matches

    def sparse_search(self, sv: dict, n: int, flt: dict | None = None):
        if not sv["indices"]:  # no known term: BM25 has nothing to say
            return []
        return self.sparse.query(
            sparse_vector=sv, top_k=n, include_metadata=True, filter=flt, namespace=self.namespace
        ).matches

    def search(
        self,
        question: str,
        top_k: int = 10,
        candidates: int = 30,
        filter: dict | None = None,
        weights: dict[str, float] | None = None,
        use_hints: bool = True,
    ) -> list[Hit]:
        qv = self.embedder.embed_query(question)  # one embedding, reused by every search
        sv = self.bm25.encode_query(question)
        results = {
            "dense": self.dense_search(qv, candidates, filter),
            "sparse": self.sparse_search(sv, candidates, filter),
        }
        if use_hints and self.hinter:
            for i, hint in enumerate(self.hinter.hints(question)):
                flt = _and(filter, hint)
                results[f"dense_hint{i}"] = self.dense_search(qv, candidates, flt)
                results[f"sparse_hint{i}"] = self.sparse_search(sv, candidates, flt)

        metadata = {m.id: m.metadata or {} for ms in results.values() for m in ms}
        rankings = {name: [m.id for m in ms] for name, ms in results.items()}
        hits = []
        for chunk_id, score in rrf(rankings, weights=weights)[:top_k]:
            ranks = {n: ids.index(chunk_id) + 1 for n, ids in rankings.items() if chunk_id in ids}
            hits.append(Hit(chunk_id, score, metadata[chunk_id], ranks))
        return hits
