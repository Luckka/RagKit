from types import SimpleNamespace

import pytest

from ragkit.retrieval.hybrid import HybridRetriever, rrf
from ragkit.retrieval.query_hints import RegexHinter
from ragkit.retrieval.sparse import BM25Encoder


def test_rrf_rewards_consensus():
    # B is 2nd and 1st; A is 1st and 3rd -> B wins by ranking well in both
    fused = rrf({"dense": ["A", "B", "D"], "sparse": ["B", "C", "A"]})
    assert [i for i, _ in fused] == ["B", "A", "C", "D"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_handles_empty_list_and_weights():
    assert [i for i, _ in rrf({"dense": ["A", "B"], "sparse": []})] == ["A", "B"]
    rankings = {"dense": ["A", "B"], "sparse": ["B", "A"]}
    assert rrf(rankings, weights={"dense": 2.0})[0][0] == "A"
    assert rrf(rankings, weights={"sparse": 2.0})[0][0] == "B"


class FakeIndex:
    """Test double for a Pinecone index: returns fixed ids, records filters and namespaces."""

    def __init__(self, ids, filtered_ids=None):
        self.ids, self.filtered_ids = ids, filtered_ids or []
        self.calls = []

    def query(self, top_k, filter=None, namespace="", **_):
        self.calls.append((filter, namespace))
        ids = self.filtered_ids if filter else self.ids
        ms = [SimpleNamespace(id=i, score=0.0, metadata={"text": i}) for i in ids[:top_k]]
        return SimpleNamespace(matches=ms)


class FakeEmbedder:
    dimension = 3

    def embed_query(self, text):
        return [0.0, 0.0, 1.0]


def test_hybrid_fuses_reports_ranks_and_uses_namespace():
    bm25 = BM25Encoder().fit(["mfa login", "password"])
    dense, sparse = FakeIndex(["A", "B", "D"]), FakeIndex(["B", "C", "A"])
    r = HybridRetriever(FakeEmbedder(), bm25, dense, sparse, namespace="docs")
    hits = r.search("mfa", top_k=3)
    assert [h.chunk_id for h in hits] == ["B", "A", "C"]
    assert hits[0].ranks == {"dense": 2, "sparse": 1}
    assert dense.calls == [(None, "docs")]


def test_sparse_skipped_when_no_known_terms():
    bm25 = BM25Encoder().fit(["mfa login"])
    r = HybridRetriever(FakeEmbedder(), bm25, FakeIndex(["A"]), FakeIndex(["X"]))
    assert [h.chunk_id for h in r.search("blockchain")] == ["A"]


def test_hint_adds_soft_filtered_lists():
    hinter = RegexHinter("pillar", r"(?i)\b(REL|SEC)-?(\d{1,2})\b", {"REL-02"}, "{0}-{1:0>2}")
    bm25 = BM25Encoder().fit(["rel-02 backup strategy", "other text"])
    dense = FakeIndex(["X", "Y", "M"], filtered_ids=["M"])
    sparse = FakeIndex(["Y", "X"], filtered_ids=["M"])
    r = HybridRetriever(FakeEmbedder(), bm25, dense, sparse, hinter=hinter)
    hits = r.search("backup strategy for rel2", top_k=3)
    assert hits[0].chunk_id == "M"  # was 3rd in dense, rises with the hint
    assert ({"pillar": {"$in": ["REL-02"]}}, "") in dense.calls
    assert {h.chunk_id for h in hits} == {"M", "X", "Y"}  # soft filter excludes nothing
    no_hint = r.search("backup strategy for rel2", top_k=3, use_hints=False)
    assert no_hint[0].chunk_id != "M"


def test_regex_hinter_normalizes_and_validates():
    h = RegexHinter("code", r"\b([A-Za-z]{2,3})[-\s]?(\d{1,2})\b", {"SEC-01"}, "{0}-{1:0>2}")
    assert h.values("compare sec1, SEC-01 and xyz-9") == ["SEC-01"]
    assert h.hints("nothing here") == []
