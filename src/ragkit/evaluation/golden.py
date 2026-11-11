"""Synthetic golden set generation (a draft for human review).

Strategy, the same idea as RAGAS testset generation:
1. Sample chunks stratified across documents and sections.
2. Ask an LLM for one question only that chunk answers, plus the answer.
3. A human (or a check against the source of truth) reviews before the set is trusted.

Bias warning: synthetic questions tend to copy the chunk's wording, which flatters BM25.
The prompt demands paraphrase, the way a real user would ask.
"""

from __future__ import annotations

import json
import random
import re

from ragkit.ingestion.chunk import Chunk
from ragkit.providers.base import LLMProvider

SYSTEM = """You write evaluation questions for a retrieval-augmented generation system.
Reply ONLY with valid JSON, no extra text."""

PROMPT = """Document excerpt:
<excerpt>
{text}
</excerpt>

Write ONE question that a practitioner would ask and that THIS excerpt answers.
Rules:
- Write the question and the answer in this language: {language}.
- Paraphrase: do not copy sentences from the excerpt; use the words a user would use.
- Be specific enough to have an objective answer in the excerpt.
- The answer has 1 to 3 sentences, based only on the excerpt.

Format: {{"question": "...", "answer": "..."}}"""

LANGUAGES = {"en": "English", "pt": "Brazilian Portuguese"}


def pick_chunks(chunks: list[Chunk], n: int, seed: int, min_chars: int = 600) -> list[Chunk]:
    """Round-robin over documents, preferring sections not used yet."""
    rng = random.Random(seed)
    by_source: dict[str, list[Chunk]] = {}
    for c in chunks:
        if len(c.text) >= min_chars:
            by_source.setdefault(c.source, []).append(c)
    for pool in by_source.values():
        rng.shuffle(pool)
    picked: list[Chunk] = []
    used_sections: set[tuple[str, str]] = set()
    sources = sorted(by_source)
    while len(picked) < n and any(by_source.values()):
        for s in sources:
            pool = by_source[s]
            if not pool or len(picked) >= n:
                continue
            fresh = [c for c in pool if (s, c.section) not in used_sections]
            c = (fresh or pool)[0]
            pool.remove(c)
            used_sections.add((s, c.section))
            picked.append(c)
    return picked


def parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0) if match else raw)


def generate(chunks: list[Chunk], llm: LLMProvider, n: int, seed: int, language: str) -> list[dict]:
    rows = []
    for i, c in enumerate(pick_chunks(chunks, n, seed), start=1):
        prompt = PROMPT.format(text=c.text, language=LANGUAGES.get(language, language))
        qa = parse_json(llm.generate(SYSTEM, prompt))
        rows.append(
            {
                "id": f"q{i:02d}",
                "question": qa["question"],
                "expected_answer": qa["answer"],
                "source_chunk_id": c.chunk_id,
                "source": c.source,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "reviewed": False,
            }
        )
    return rows
