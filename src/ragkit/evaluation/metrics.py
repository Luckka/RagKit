"""Retrieval metrics as pure functions.

- hit@k: share of questions whose relevant chunk appears in the top k (recall@k with one
  relevant chunk per question).
- MRR@k: mean of 1/rank of the first relevant chunk (1st = 1.0, 2nd = 0.5, outside k = 0).
"""


def is_relevant(
    retrieved: dict, expected: dict, match_fields: tuple[str, ...] = ("source", "section")
) -> bool:
    """Relevant = the source chunk itself, or another chunk from the same place covering the
    same pages (e.g. the other half of a table that was split across chunks)."""
    if retrieved["chunk_id"] == expected["source_chunk_id"]:
        return True
    same_place = all(retrieved.get(f) == expected.get(f) for f in match_fields)
    overlap = retrieved.get("page_start", 0) <= expected.get("page_end", 0) and retrieved.get(
        "page_end", 0
    ) >= expected.get("page_start", 0)
    return same_place and overlap


def first_hit_rank(results: list[dict], expected: dict, **kw) -> int | None:
    for pos, r in enumerate(results, start=1):
        if is_relevant(r, expected, **kw):
            return pos
    return None


def summarize(ranks: list[int | None], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    n = len(ranks) or 1
    out = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks}
    out["mrr@10"] = sum(1 / r for r in ranks if r and r <= 10) / n
    return out
