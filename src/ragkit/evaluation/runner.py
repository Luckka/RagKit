"""Compare retrieval configurations on a golden set: which stage actually earns its cost?"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from ragkit.evaluation.metrics import first_hit_rank, summarize
from ragkit.providers.base import RerankProvider
from ragkit.retrieval.hybrid import Hit, HybridRetriever
from ragkit.retrieval.rerank import rerank


def as_rows(hits: list[Hit]) -> list[dict]:
    return [{"chunk_id": h.chunk_id, **h.metadata} for h in hits]


class CachedReranker:
    """Disk cache + pacing: rerank quotas can be a few requests per minute on new accounts,
    and an evaluation must be re-runnable without paying for the same calls twice."""

    def __init__(self, inner: RerankProvider, cache_path: Path, min_interval_s: float = 0.0):
        self.inner = inner
        self.path = cache_path
        self.min_interval_s = min_interval_s
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._last = 0.0

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        # Keyed by the candidate *set*, not its order: approximate nearest-neighbour search can
        # return the same candidates in a slightly different order between runs.
        digests = [hashlib.md5(d.encode()).hexdigest() for d in documents]
        key = hashlib.md5(f"{query}|{top_n}|{'|'.join(sorted(digests))}".encode()).hexdigest()
        if key not in self.cache:
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            order = self.inner.rerank(query, documents, top_n)
            self.cache[key] = [(digests[i], score) for i, score in order]
            self._last = time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache))
        position = {d: i for i, d in enumerate(digests)}
        return [(position[d], score) for d, score in self.cache[key]]


def configs(
    r: HybridRetriever, reranker: RerankProvider | None
) -> dict[str, Callable[[str], list[Hit]]]:
    def dense(q: str) -> list[Hit]:
        ms = r.dense_search(r.embedder.embed_query(q), 10)
        return [Hit(m.id, m.score, m.metadata or {}) for m in ms]

    def sparse(q: str) -> list[Hit]:
        ms = r.sparse_search(r.bm25.encode_query(q), 10)
        return [Hit(m.id, m.score, m.metadata or {}) for m in ms]

    out: dict[str, Callable[[str], list[Hit]]] = {
        "dense": dense,
        "sparse": sparse,
        "hybrid": lambda q: r.search(q, top_k=10, use_hints=False),
    }
    if r.hinter:
        out["hybrid+hints"] = lambda q: r.search(q, top_k=10)
    if reranker:

        def dense_rerank(q: str) -> list[Hit]:
            ms = r.dense_search(r.embedder.embed_query(q), 30)
            return rerank(q, [Hit(m.id, m.score, m.metadata or {}) for m in ms], reranker, 10)

        out["dense+rerank"] = dense_rerank
        out["hybrid+rerank"] = lambda q: rerank(q, r.search(q, top_k=30), reranker, 10)
    return out


def evaluate(
    golden: list[dict],
    runs: dict[str, Callable[[str], list[Hit]]],
    match_fields: tuple[str, ...] = ("source", "section"),
) -> dict[str, dict]:
    """match_fields=("source",) judges relevance by page overlap only, which keeps a golden set
    valid across different chunking strategies (chunk ids and sections change, pages do not)."""
    report = {}
    for name, fn in runs.items():
        ranks = [
            first_hit_rank(as_rows(fn(g["question"])), g, match_fields=match_fields) for g in golden
        ]
        report[name] = summarize(ranks) | {"ranks": ranks}
    return report


def to_markdown(report: dict[str, dict], n: int) -> str:
    lines = [
        f"| config (n={n}) | hit@1 | hit@5 | hit@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, m in report.items():
        lines.append(
            f"| {name} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | {m['hit@10']:.2f} "
            f"| {m['mrr@10']:.2f} |"
        )
    return "\n".join(lines)
