"""Command line: `ragkit <command> <corpus>`.

ragkit download aws-well-architected   fetch the documents listed in corpus.yaml
ragkit ingest   aws-well-architected   parse with Docling and chunk
ragkit index    aws-well-architected   embed + BM25 and upsert to Pinecone (one namespace)
ragkit search   aws-well-architected "how do I protect data at rest?"
ragkit ask      aws-well-architected "how do I protect data at rest?"
ragkit golden   aws-well-architected   draft a synthetic golden set for review
ragkit eval     aws-well-architected   compare retrieval configurations
"""

from __future__ import annotations

import argparse
import json

from ragkit import pipeline
from ragkit.corpus import Corpus


def cmd_search(corpus: Corpus, question: str, rerank: bool) -> None:
    r = pipeline.retriever(corpus)
    hits = r.search(question, top_k=30 if rerank else 5)
    if rerank:
        from ragkit.providers import get_reranker
        from ragkit.retrieval.rerank import rerank as do_rerank

        hits = do_rerank(question, hits, get_reranker(), top_k=5)
    for pos, h in enumerate(hits, start=1):
        meta = h.metadata
        print(f"{pos}. [{h.score:.4f}] {meta.get('source')} p.{meta.get('page_start')} {h.ranks}")
        print(f"   {meta.get('section')}")


def cmd_ask(corpus: Corpus, question: str) -> None:
    from ragkit.generation.answer import Answerer
    from ragkit.providers import get_llm, get_reranker

    result = Answerer(pipeline.retriever(corpus), get_llm(), get_reranker()).ask(question)
    print(result.answer)
    if result.citations:
        print("\nSources:")
    for c in result.citations:
        pages = f"p. {c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "")
        print(f"  [{c.ref}] {c.source}, {pages}, {c.section}")


def cmd_golden(corpus: Corpus) -> None:
    from ragkit.evaluation.golden import generate
    from ragkit.providers import get_llm

    rows = generate(
        pipeline.load_chunks(corpus),
        get_llm(),
        corpus.golden_size,
        corpus.golden_seed,
        corpus.language,
    )
    corpus.eval_dir.mkdir(parents=True, exist_ok=True)
    out = corpus.eval_dir / "golden_draft.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} questions -> {out} (draft: review before trusting it)")


def cmd_eval(corpus: Corpus, rerank_interval: float, no_rerank: bool, match: str, tag: str) -> None:
    from ragkit.evaluation.runner import CachedReranker, configs, evaluate, to_markdown

    path = corpus.eval_dir / "golden.jsonl"
    if not path.exists():
        path = corpus.eval_dir / "golden_draft.jsonl"
    golden = [json.loads(line) for line in path.open(encoding="utf-8")]
    reranker = None
    if not no_rerank:
        from ragkit.providers import get_reranker

        cache = corpus.eval_dir / "rerank_cache.json"
        reranker = CachedReranker(get_reranker(), cache, rerank_interval)
    fields = tuple(match.split(","))
    report = evaluate(golden, configs(pipeline.retriever(corpus), reranker), fields)
    table = to_markdown(report, len(golden))
    suffix = f"_{tag}" if tag else ""
    (corpus.eval_dir / f"results{suffix}.md").write_text(table + "\n")
    (corpus.eval_dir / f"results{suffix}.json").write_text(json.dumps(report, indent=2))
    print(table)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ragkit", description="RagKit corpus pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("download", "ingest", "index", "golden"):
        sub.add_parser(name).add_argument("corpus")
    s = sub.add_parser("search")
    s.add_argument("corpus")
    s.add_argument("question")
    s.add_argument("--rerank", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("corpus")
    a.add_argument("question")
    e = sub.add_parser("eval")
    e.add_argument("corpus")
    e.add_argument("--rerank-interval", type=float, default=0.0, help="seconds between calls")
    e.add_argument("--no-rerank", action="store_true")
    e.add_argument("--match", default="source,section", help="metadata fields that must match")
    e.add_argument("--tag", default="", help="suffix for the results files")
    args = p.parse_args(argv)

    corpus = Corpus.load(args.corpus)
    match args.command:
        case "download":
            pipeline.download(corpus)
        case "ingest":
            pipeline.ingest(corpus)
        case "index":
            pipeline.index(corpus)
        case "search":
            cmd_search(corpus, args.question, args.rerank)
        case "ask":
            cmd_ask(corpus, args.question)
        case "golden":
            cmd_golden(corpus)
        case "eval":
            cmd_eval(corpus, args.rerank_interval, args.no_rerank, args.match, args.tag)


if __name__ == "__main__":
    main()
