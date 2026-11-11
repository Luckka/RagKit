from fastapi.testclient import TestClient

from ragkit.api.main import app, get_answerer_factory
from ragkit.generation.answer import (
    NOT_FOUND,
    REFUSAL,
    Answerer,
    build_prompt,
    collapse_twins,
    parse_citations,
)
from ragkit.retrieval.hybrid import Hit


def hit(i: int, text: str = "", score: float = 1.0) -> Hit:
    meta = {"text": text or f"passage {i}", "source": f"doc{i}.pdf", "section": f"S{i}"}
    return Hit(f"c{i}", score, {**meta, "page_start": i, "page_end": i + 1})


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, question, top_k=10):
        self.calls.append(top_k)
        return self.hits[:top_k]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, system, prompt):
        self.prompts.append((system, prompt))
        return self.reply


class FakeReranker:
    def rerank(self, query, documents, top_n):
        # reverse order, descending scores
        idx = list(range(len(documents)))[::-1][:top_n]
        return [(i, 1.0 - pos * 0.3) for pos, i in enumerate(idx)]


def test_prompt_numbers_passages_with_source_and_pages():
    prompt = build_prompt("q?", [hit(1, "alpha"), hit(2, "beta")])
    assert "[1] doc1.pdf, p. 1-2\nalpha" in prompt
    assert "[2] doc2.pdf, p. 2-3\nbeta" in prompt
    assert prompt.endswith("Question: q?")


def test_parse_citations_dedups_and_drops_out_of_range():
    hits = [hit(1), hit(2)]
    cites = parse_citations("A [2]. B [1][2]. C [7].", hits)
    assert [c.ref for c in cites] == [2, 1]
    assert cites[0].source == "doc2.pdf" and cites[0].page_end == 3


def test_ask_returns_cited_sources_only():
    llm = FakeLLM("Encrypt it [2].")
    result = Answerer(FakeRetriever([hit(1), hit(2), hit(3)]), llm, top_k=3).ask("q?")
    assert result.grounded
    assert [c.chunk_id for c in result.citations] == ["c2"]


def test_not_found_marker_becomes_refusal():
    result = Answerer(FakeRetriever([hit(1)]), FakeLLM(NOT_FOUND)).ask("q?")
    assert result.answer == REFUSAL and not result.grounded


def test_uncited_answer_is_refused():
    result = Answerer(FakeRetriever([hit(1)]), FakeLLM("Sounds right.")).ask("q?")
    assert result.answer == REFUSAL and not result.grounded


def test_no_context_skips_the_llm():
    llm = FakeLLM("x")
    result = Answerer(FakeRetriever([]), llm).ask("q?")
    assert result.answer == REFUSAL and llm.prompts == []


def test_rerank_path_and_min_score_filter():
    retriever = FakeRetriever([hit(i) for i in range(1, 6)])
    answerer = Answerer(retriever, FakeLLM("ok [1]"), FakeReranker(), top_k=3, min_score=0.5)
    ctx = answerer.contexts("q?")
    assert retriever.calls == [30]
    assert [h.chunk_id for h in ctx] == ["c5", "c4"]  # third scored 0.4, dropped


def test_collapse_twins_keeps_higher_ranked_copy():
    a = hit(1)
    a.metadata["also_in"] = ["doc2.pdf#p2-3"]
    assert [h.chunk_id for h in collapse_twins([a, hit(2), hit(3)])] == ["c1", "c3"]


def test_ask_endpoint_uses_injected_answerer():
    answerer = Answerer(FakeRetriever([hit(1), hit(2)]), FakeLLM("Yes [1]."))
    app.dependency_overrides[get_answerer_factory] = lambda: lambda name: answerer
    try:
        body = TestClient(app).post("/ask", json={"question": "is it?", "corpus": "demo"}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["grounded"] is True
    assert body["sources"] == [
        {"ref": 1, "source": "doc1.pdf", "section": "S1", "page_start": 1, "page_end": 2}
    ]


def test_ask_endpoint_rejects_bad_corpus_name():
    resp = TestClient(app).post("/ask", json={"question": "is it?", "corpus": "../etc"})
    assert resp.status_code == 422
