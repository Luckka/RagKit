"""Near-duplicate detection across documents.

Document sets often embed each other (an overview that copies whole chapters of a
detailed guide). Duplicates waste context slots and, in evaluation, a correct passage
retrieved from the "other" copy would be scored as a miss.

Every chunk with near-identical twins gets their locations in `also_in`
("source#p<start>-<end>"). Optionally only the first copy, in document order, is kept.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace

from ragkit.ingestion.chunk import Chunk

WORD = re.compile(r"\w+")


def location(source: str, page_start: int, page_end: int) -> str:
    return f"{source}#p{page_start}-{page_end}"


def parse_location(loc: str) -> tuple[str, int, int]:
    source, _, pages = loc.rpartition("#p")
    start, _, end = pages.partition("-")
    return source, int(start), int(end)


def shingles(text: str, n: int = 5) -> set[str]:
    # Skip the breadcrumb line: twins often sit under different heading paths
    body = text.split("\n", 1)[1] if "\n" in text else text
    words = WORD.findall(body.lower())
    return {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def find_twins(chunks: list[Chunk], threshold: float, max_df: int = 50) -> list[list[int]]:
    """For each chunk, the indexes of chunks whose shingle Jaccard is >= threshold.

    An inverted index limits comparisons to chunks that share at least one shingle;
    shingles that appear in more than max_df chunks (boilerplate) are ignored.
    """
    sets = [shingles(c.text) for c in chunks]
    postings: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for g in s:
            postings[g].append(i)
    twins: list[list[int]] = []
    for i, s in enumerate(sets):
        candidates = {j for g in s if len(postings[g]) <= max_df for j in postings[g] if j != i}
        twins.append(
            sorted(j for j in candidates if len(s & sets[j]) / len(s | sets[j]) >= threshold)
        )
    return twins


def dedup(chunks: list[Chunk], threshold: float, drop: bool = False) -> list[Chunk]:
    twins = find_twins(chunks, threshold)
    out: list[Chunk] = []
    removed: set[int] = set()
    for i, c in enumerate(chunks):
        if drop and i in removed:
            continue
        group = twins[i]
        if drop:
            removed.update(j for j in group if j > i)
        if not group:
            out.append(c)
            continue
        also_in = [
            location(chunks[j].source, chunks[j].page_start, chunks[j].page_end) for j in group
        ]
        out.append(replace(c, extra={**c.extra, "also_in": also_in}))
    return out
