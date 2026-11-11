import pytest
from fastapi.testclient import TestClient

from ragkit.api.main import app
from ragkit.evaluation.metrics import first_hit_rank, is_relevant, summarize
from ragkit.retrieval.hybrid import Hit
from ragkit.retrieval.rerank import rerank


class FakeReranker:
    """Score = share of query words present in the document."""

    def rerank(self, query, documents, top_n):
        words = set(query.split())
        scored = [(i, len(words & set(d.split())) / len(words)) for i, d in enumerate(documents)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]


def test_rerank_reorders_and_keeps_original_ranks():
    hits = [
        Hit("a", 0.0, {"text": "unrelated"}, {"dense": 1}),
        Hit("b", 0.0, {"text": "s3 bucket encryption"}, {"dense": 2}),
        Hit("c", 0.0, {"text": "encryption"}, {"dense": 3}),
    ]
    out = rerank("s3 bucket encryption", hits, FakeReranker(), top_k=2)
    assert [h.chunk_id for h in out] == ["b", "c"]
    assert out[0].ranks == {"dense": 2, "rerank": 1} and out[0].score == 1.0
    assert rerank("x", [], FakeReranker()) == []


EXPECTED = {
    "source_chunk_id": "c1",
    "source": "guide.pdf",
    "section": "Backups",
    "page_start": 10,
    "page_end": 11,
}


def chunk(cid, source="guide.pdf", section="Backups", p0=10, p1=10):
    return {"chunk_id": cid, "source": source, "section": section, "page_start": p0, "page_end": p1}


def test_relevance_by_id_or_same_section_and_pages():
    assert is_relevant(chunk("c1", source="x"), EXPECTED)
    assert is_relevant(chunk("c2", p0=11, p1=11), EXPECTED)
    assert not is_relevant(chunk("c3", section="Logging"), EXPECTED)
    assert not is_relevant(chunk("c4", p0=20, p1=21), EXPECTED)
    assert first_hit_rank([chunk("x", section="Other"), chunk("c1")], EXPECTED) == 2


def test_summarize():
    s = summarize([1, 2, None, 6])
    assert (s["hit@1"], s["hit@5"], s["hit@10"]) == (0.25, 0.5, 0.75)
    assert s["mrr@10"] == pytest.approx((1 + 0.5 + 1 / 6) / 4)


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
