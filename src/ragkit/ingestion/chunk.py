"""Structure-aware chunking.

Rules:
1. A chunk never mixes sections: elements are grouped by (source, heading path, extra metadata).
2. Size cap: embedding models truncate silently (Cohere Embed v3 at 512 tokens), so bodies are
   capped well below that (default 1600 chars) instead of losing text without an error.
3. Tables are never cut mid-row; a large table is split by rows, repeating its header.
4. Every chunk starts with a breadcrumb (document > headings) so the embedding knows its context.
5. A short trailing paragraph is repeated in the next chunk (overlap) for continuity.
6. Tiny sections are merged into the previous chunk of the same parent heading.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import groupby

from ragkit.ingestion.elements import Element


@dataclass
class Chunk:
    chunk_id: str
    text: str  # breadcrumb + body: this is what gets embedded
    source: str
    section: str
    page_start: int
    page_end: int
    has_table: bool
    extra: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        """Flat metadata for the vector store (no nested objects, no nulls)."""
        base = {
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "has_table": self.has_table,
        }
        return base | {k: v for k, v in self.extra.items() if v is not None}


def compact_table(md: str) -> str:
    """Docling pads columns with spaces for alignment; for embeddings that is wasted tokens."""
    rows = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= {"-", ":"} for c in cells):
            cells = ["---"] * len(cells)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def split_table(md: str, max_chars: int) -> list[str]:
    """Split a large Markdown table by rows, repeating header and separator in every part."""
    lines = md.splitlines()
    header, rows = lines[:2], lines[2:]
    parts, current = [], []
    for row in rows:
        if current and len("\n".join(header + current + [row])) > max_chars:
            parts.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(header + current))
    return parts


class HeadingChunker:
    def __init__(self, max_chars: int = 1600, min_chars: int = 400, overlap_max: int = 300):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_max = overlap_max

    @staticmethod
    def breadcrumb(e: Element) -> str:
        return " > ".join([e.source, *e.headings])

    @staticmethod
    def _key(e: Element) -> tuple:
        return (e.source, tuple(e.headings), tuple(sorted(e.extra.items())))

    def _pieces(self, group: list[Element]) -> list[tuple[str, int, bool]]:
        out = []
        for e in group:
            if e.kind == "heading":
                continue  # already in the breadcrumb
            if e.kind == "table":
                table = compact_table(e.text)
                parts = (
                    split_table(table, self.max_chars) if len(table) > self.max_chars else [table]
                )
                out += [(t, e.page, True) for t in parts]
            else:
                out.append((f"- {e.text}" if e.kind == "list_item" else e.text, e.page, False))
        return out

    def _chunk_group(self, group: list[Element]) -> list[Chunk]:
        first = group[0]
        crumb = self.breadcrumb(first)
        chunks: list[Chunk] = []
        buf: list[tuple[str, int, bool]] = []

        def flush():
            body = "\n\n".join(t for t, _, _ in buf)
            digest = hashlib.sha1(f"{crumb}|{len(chunks)}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=digest,
                    text=f"{crumb}\n\n{body}",
                    source=first.source,
                    section=" > ".join(first.headings),
                    page_start=min(p for _, p, _ in buf),
                    page_end=max(p for _, p, _ in buf),
                    has_table=any(t for _, _, t in buf),
                    extra=dict(first.extra),
                )
            )

        carried = False  # True when the buffer only holds overlap inherited from the last chunk
        for piece in self._pieces(group):
            if buf and sum(len(t) for t, _, _ in buf) + len(piece[0]) > self.max_chars:
                if not carried:
                    flush()
                last = buf[-1]
                keep = not last[2] and len(last[0]) <= self.overlap_max
                keep = keep and len(last[0]) + len(piece[0]) <= self.max_chars
                buf, carried = ([last], True) if keep else ([], False)
            buf.append(piece)
            carried = False
        if buf and not carried:
            flush()
        return chunks

    def chunk(self, elements: list[Element]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for _, grp in groupby(elements, key=self._key):  # consecutive groups keep reading order
            grp = list(grp)
            if not self._pieces(grp):
                continue
            new = self._chunk_group(grp)
            prev = chunks[-1] if chunks else None
            parent = grp[0].headings[:-1]
            if (
                len(new) == 1
                and prev
                and prev.source == new[0].source
                and prev.extra == new[0].extra
                and prev.section.split(" > ")[: len(parent)] == parent
                and len(new[0].text) < self.min_chars
                and len(prev.text) + len(new[0].text) <= self.max_chars
            ):
                body = new[0].text.split("\n\n", 1)[1]
                prev.text += f"\n\n### {grp[0].headings[-1] if grp[0].headings else ''}\n{body}"
                prev.page_end = max(prev.page_end, new[0].page_end)
                prev.has_table |= new[0].has_table
                continue
            chunks += new
        return chunks
