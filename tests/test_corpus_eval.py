import json

from ragkit.corpus import Corpus
from ragkit.evaluation.golden import parse_json, pick_chunks
from ragkit.evaluation.runner import CachedReranker, evaluate, to_markdown
from ragkit.ingestion.chunk import Chunk
from ragkit.retrieval.hybrid import Hit

YAML = """
name: demo
language: en
documents:
  - file: a.pdf
    url: https://example.com/a.pdf
chunking: {max_chars: 1200}
hints:
  - field: pillar
    pattern: '(?i)\\b(SEC|REL)\\b'
golden: {size: 3, seed: 7}
"""


def test_corpus_yaml_and_hinter(tmp_path, monkeypatch):
    (tmp_path / "corpora" / "demo").mkdir(parents=True)
    (tmp_path / "corpora" / "demo" / "corpus.yaml").write_text(YAML)
    monkeypatch.chdir(tmp_path)
    c = Corpus.load("demo")
    assert (c.max_chars, c.min_chars, c.golden_size) == (1200, 400, 3)
    assert c.documents[0].url.endswith("a.pdf")
    assert c.build_hinter() is None  # no indexed values yet
    c.processed_dir.mkdir(parents=True)
    c.hint_values_path.write_text(json.dumps({"pillar": ["SEC"]}))
    assert c.build_hinter().hints("what does sec say?") == [{"pillar": {"$in": ["SEC"]}}]


def chunk(i, source, section):
    return Chunk(f"c{i}", "x" * 700, source, section, 1, 1, False)


def test_pick_chunks_spreads_over_sources_and_sections():
    chunks = [chunk(i, f"doc{i % 2}.pdf", f"s{i % 3}") for i in range(12)]
    picked = pick_chunks(chunks, n=4, seed=1)
    assert len(picked) == 4
    assert {c.source for c in picked} == {"doc0.pdf", "doc1.pdf"}
    assert len({(c.source, c.section) for c in picked}) == 4


def test_parse_json_tolerates_wrapping_text():
    assert parse_json('Sure! {"question": "q", "answer": "a"} ok') == {
        "question": "q",
        "answer": "a",
    }


def test_evaluate_and_markdown():
    golden = [
        {"question": "q1", "source_chunk_id": "a"},
        {"question": "q2", "source_chunk_id": "z"},
    ]
    runs = {"fake": lambda q: [Hit("b", 1, {}), Hit("a", 1, {})]}
    report = evaluate(golden, runs)
    assert report["fake"]["ranks"] == [2, None]
    assert report["fake"]["hit@5"] == 0.5
    assert "| fake | 0.00 | 0.50 | 0.50 | 0.25 |" in to_markdown(report, 2)


def test_cached_reranker_calls_once(tmp_path):
    calls = []

    class Inner:
        def rerank(self, query, documents, top_n):
            calls.append(query)
            return [(1, 0.9), (0, 0.1)]

    r = CachedReranker(Inner(), tmp_path / "cache.json")
    assert r.rerank("q", ["a", "b"], 2) == [(1, 0.9), (0, 0.1)]
    again = CachedReranker(Inner(), tmp_path / "cache.json")
    assert again.rerank("q", ["a", "b"], 2) == [(1, 0.9), (0, 0.1)]
    assert calls == ["q"]


def test_cached_reranker_ignores_candidate_order(tmp_path):
    class Inner:
        def rerank(self, query, documents, top_n):
            return [(documents.index("b"), 0.9)]

    r = CachedReranker(Inner(), tmp_path / "cache.json")
    assert r.rerank("q", ["a", "b"], 1) == [(1, 0.9)]
    r.inner = None  # a second call must come from the cache
    assert r.rerank("q", ["b", "a"], 1) == [(0, 0.9)]
