"""End-to-end steps for one corpus: download, ingest, index and build a retriever."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict

from ragkit.corpus import Corpus
from ragkit.ingestion.chunk import Chunk, HeadingChunker
from ragkit.providers import get_embeddings
from ragkit.retrieval.hybrid import HybridRetriever
from ragkit.retrieval.sparse import BM25Encoder

BATCH = 96


def download(corpus: Corpus) -> None:
    corpus.raw_dir.mkdir(parents=True, exist_ok=True)
    for d in corpus.documents:
        target = corpus.raw_dir / d.file
        if target.exists() or not d.url:
            print(f"skip {d.file}")
            continue
        req = urllib.request.Request(d.url, headers={"User-Agent": "ragkit"})
        with urllib.request.urlopen(req, timeout=120) as r:
            target.write_bytes(r.read())
        print(f"downloaded {d.file} ({target.stat().st_size // 1024} KB)")


def ingest(corpus: Corpus) -> list[Chunk]:
    from ragkit.ingestion.elements import apply_tags, extract
    from ragkit.ingestion.parse import converter, parse

    conv = converter()
    profile = corpus.load_profile()
    chunker = HeadingChunker(max_chars=corpus.max_chars, min_chars=corpus.min_chars)
    chunks: list[Chunk] = []
    for d in corpus.documents:
        t0 = time.time()
        doc = parse(corpus.raw_dir / d.file, corpus.processed_dir / "docling", conv)
        elements = extract(doc, d.file, corpus.heading_rules, corpus.default_heading_level)
        elements = apply_tags(elements, corpus.tags)
        if profile:
            elements = profile.process(elements)
        new = chunker.chunk(elements)
        chunks += new
        print(f"{d.file}: {doc.num_pages()} pages, {len(new)} chunks ({time.time() - t0:.0f}s)")

    corpus.processed_dir.mkdir(parents=True, exist_ok=True)
    with corpus.chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    sizes = sorted(len(c.text) for c in chunks)
    print(f"{len(chunks)} chunks | median {sizes[len(sizes) // 2]} | max {sizes[-1]} chars")
    return chunks


def load_chunks(corpus: Corpus) -> list[Chunk]:
    with corpus.chunks_path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def index(corpus: Corpus) -> None:
    from ragkit.retrieval import store

    chunks = load_chunks(corpus)
    texts = [c.text for c in chunks]
    bm25 = BM25Encoder(language=corpus.language).fit(texts)
    bm25.save(corpus.bm25_path)
    known = {
        h.field: sorted({str(c.extra[h.field]) for c in chunks if c.extra.get(h.field)})
        for h in corpus.hints
    }
    corpus.hint_values_path.write_text(json.dumps(known))

    embedder = get_embeddings()
    t0 = time.time()
    dense = embedder.embed_documents(texts)
    print(f"{len(chunks)} embeddings (dim={len(dense[0])}) in {time.time() - t0:.1f}s")

    store.ensure_indexes(dimension=embedder.dimension)
    dense_idx, sparse_idx = store.dense_index(), store.sparse_index()
    ns = corpus.name
    for i in range(0, len(chunks), BATCH):
        part = chunks[i : i + BATCH]
        dense_idx.upsert(
            namespace=ns,
            vectors=[
                {"id": c.chunk_id, "values": v, "metadata": c.metadata()}
                for c, v in zip(part, dense[i : i + BATCH], strict=True)
            ],
        )
        sparse_idx.upsert(
            namespace=ns,
            vectors=[
                {
                    "id": c.chunk_id,
                    "sparse_values": bm25.encode_document(c.text),
                    "metadata": c.metadata(),
                }
                for c in part
            ],
        )
    print(f"upserted {len(chunks)} vectors into namespace '{ns}' of both indexes")


def retriever(corpus: Corpus) -> HybridRetriever:
    from ragkit.retrieval import store

    return HybridRetriever(
        embedder=get_embeddings(),
        bm25=BM25Encoder.load(corpus.bm25_path),
        dense_index=store.dense_index(),
        sparse_index=store.sparse_index(),
        namespace=corpus.name,
        hinter=corpus.build_hinter(),
    )
