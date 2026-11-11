"""Retrieval metrics as pure functions.

- hit@k: share of questions whose relevant chunk appears in the top k (recall@k with one
  relevant chunk per question).
- MRR@k: mean of 1/rank of the first relevant chunk (1st = 1.0, 2nd = 0.5, outside k = 0).
"""

from ragkit.ingestion.dedup import parse_location


def is_relevant(
    retrieved: dict, expected: dict, match_fields: tuple[str, ...] = ("source", "section")
) -> bool:
    """Relevant = the source chunk itself, or another chunk from the same place covering the
    same pages (e.g. the other half of a table that was split across chunks)."""
    if retrieved["chunk_id"] == expected["source_chunk_id"]:
        return True
    same_place = all(
        expected.get(f) is not None and retrieved.get(f) == expected.get(f) for f in match_fields
    )
    if same_place and _overlap(
        retrieved.get("page_start", 0), retrieved.get("page_end", 0), expected
    ):
        return True
    # A near-identical copy elsewhere is the same answer. Locations carry only source and
    # pages, so they count only when matching on those fields.
    if set(match_fields) <= {"source"}:
        for loc in retrieved.get("also_in") or []:
            source, start, end = parse_location(loc)
            if source == expected.get("source") and _overlap(start, end, expected):
                return True
    return False


def _overlap(start: int, end: int, expected: dict) -> bool:
    return start <= expected.get("page_end", 0) and end >= expected.get("page_start", 0)


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
