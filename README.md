# RagKit

**Production-grade, provider-agnostic Retrieval-Augmented Generation toolkit in Python, built around measured retrieval quality.**

[![CI](https://github.com/Luckka/RagKit/actions/workflows/ci.yml/badge.svg)](https://github.com/Luckka/RagKit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![AWS Bedrock](https://img.shields.io/badge/AWS_Bedrock-Claude_%7C_Cohere-232F3E?logo=amazonwebservices&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-supported-412991?logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/vector_DB-Pinecone-000000)
![License](https://img.shields.io/badge/license-MIT-green)

RagKit turns any document collection (PDF, DOCX, PPTX, HTML, Markdown) into a searchable knowledge base, and answers the question most RAG demos skip: **which retrieval strategy actually works for this corpus, and by how much?**

It combines structure-aware ingestion, hybrid search, cross-encoder reranking and a built-in evaluation harness behind small, swappable interfaces, so the same pipeline runs on AWS Bedrock or OpenAI and on any corpus described by a YAML file.

---

## Highlights

| Capability | What it does | Why it matters |
| --- | --- | --- |
| **Structure-aware ingestion** | Docling layout parsing, heading breadcrumbs, tables kept intact or split by rows with the header repeated, page ranges on every chunk | Chunks that keep their context retrieve better and can be cited by page |
| **Hybrid retrieval** | Dense embeddings + BM25 sparse vectors in Pinecone, fused with Reciprocal Rank Fusion (k=60) | Semantic recall plus exact-term matching for codes, acronyms and product names |
| **Two-stage retrieval** | Cohere Rerank 3.5 cross-encoder over the top 30 hybrid candidates | Stage 1 optimizes recall, stage 2 optimizes precision at the top |
| **Query hints** | Pluggable `QueryHinter` detects identifiers in a question and adds filtered result lists to the fusion | A soft metadata filter: matching chunks rise, nothing is excluded |
| **Built-in evaluation** | Synthetic golden set generation, hit@1/5/10 and MRR@10 across six configurations, cached and rate-limit aware | Every design decision is backed by a number, not a vibe |
| **Provider-agnostic** | `LLMProvider`, `EmbeddingProvider` and `RerankProvider` Protocols with Bedrock and OpenAI implementations | Swap models or vendors through configuration |
| **Corpus-as-config** | `corpora/<name>/corpus.yaml` declares documents, language, chunking, hints and an optional cleanup profile | A new knowledge base is a YAML file, not a fork |
| **Multi-tenant storage** | One Pinecone index pair, one namespace per corpus | Many corpora on the free tier, clean isolation |

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        D[Documents<br/>PDF · DOCX · HTML] --> P[Docling parser]
        P --> E[Elements with<br/>heading path + page]
        E --> PR[Optional corpus Profile]
        PR --> C[Structure-aware chunker]
    end
    subgraph Indexing
        C --> EMB[Dense embeddings<br/>Cohere v3 / OpenAI]
        C --> BM[BM25 sparse vectors]
        EMB --> PD[(Pinecone dense<br/>namespace = corpus)]
        BM --> PS[(Pinecone sparse<br/>namespace = corpus)]
    end
    subgraph Retrieval
        Q[Question] --> H[Query hints]
        Q --> PD
        Q --> PS
        H --> PD
        H --> PS
        PD --> F[RRF fusion]
        PS --> F
        F --> R[Cross-encoder rerank<br/>Cohere Rerank 3.5]
        R --> K[Top-k chunks<br/>with source + page]
    end
    K --> EV[Evaluation<br/>hit@k · MRR]
```

## Design decisions

- **Separate dense and sparse indexes, fused in code.** Cosine and BM25 scores live on incomparable scales. RRF uses only ranks, so it needs no score normalization or weight tuning to be a strong baseline (Cormack et al., 2009).
- **Chunk size is capped by the embedding model, not by taste.** Cohere Embed v3 silently truncates at 512 tokens. Chunks are capped at about 1,600 characters including the breadcrumb, so no text is ever dropped without an error.
- **Tables are never cut mid-row.** Large tables are split by rows and every part repeats the header, so each chunk is still a readable table.
- **Reranking is justified by measurement.** On a private 416-page technical document (Portuguese, 25-question golden set), hybrid search won on recall (hit@10 0.92) but not at the top (hit@1 0.52). Adding the cross-encoder raised hit@1 to 0.72 and MRR from 0.64 to 0.80.
- **Hints are soft filters.** A hard metadata filter turns one wrong guess into a guaranteed miss. Fusing a filtered list into RRF boosts matches without excluding anything else.
- **Evaluation is cheap to rerun.** Rerank calls are cached on disk and paced, because new cloud accounts can have rerank quotas of a few requests per minute.
- **Synthetic questions must paraphrase.** Questions copied from the chunk flatter BM25. The generator demands user-style wording, and drafts are reviewed before they are trusted.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Luckka/RagKit.git && cd RagKit
uv sync
cp .env.example .env   # set PINECONE_API_KEY and AWS credentials (or OPENAI_API_KEY)
```

Run the public demo corpus, the [AWS Well-Architected Framework](https://docs.aws.amazon.com/wellarchitected/latest/framework/welcome.html) and its Security pillar (about 1,240 pages):

```bash
uv run ragkit download aws-well-architected   # fetches the official PDFs (not stored in git)
uv run ragkit ingest   aws-well-architected   # Docling parse + chunking
uv run ragkit index    aws-well-architected   # embeddings + BM25 into Pinecone
uv run ragkit search   aws-well-architected "How should I protect data at rest?" --rerank
uv run ragkit golden   aws-well-architected   # draft a synthetic golden set for review
uv run ragkit eval     aws-well-architected   # compare all retrieval configurations
```

## Bring your own corpus

```yaml
# corpora/my-docs/corpus.yaml
name: my-docs
language: en            # en or pt, drives BM25 stopwords and golden set language
documents:
  - file: handbook.pdf  # placed in data/my-docs/raw/ (or give a url to download it)
chunking:
  max_chars: 1600
  min_chars: 400
hints:                  # optional: identifiers worth boosting when a question mentions them
  - field: ticket
    pattern: '(?i)\b(OPS)-?(\d{1,4})\b'
    template: '{0}-{1}'
profile: my_package.profiles:HandbookProfile   # optional document-specific cleanup
```

A `Profile` is a single method, `process(elements) -> elements`, for rules that only make sense for one document type, such as running headers or numbering schemes. The pipeline itself never mentions a specific corpus.

## Evaluation

`ragkit eval` compares six configurations on the golden set and writes `results.md` and `results.json`:

| Config | Stage 1 | Stage 2 |
| --- | --- | --- |
| `dense` | embeddings only | none |
| `sparse` | BM25 only | none |
| `hybrid` | dense + BM25 with RRF | none |
| `hybrid+hints` | hybrid + filtered lists from query hints | none |
| `dense+rerank` | embeddings, top 30 | cross-encoder |
| `hybrid+rerank` | hybrid + hints, top 30 | cross-encoder |

Metrics: **hit@k**, the share of questions whose relevant chunk is in the top k, and **MRR@10**, the mean reciprocal rank of the first relevant chunk. A retrieved chunk counts as relevant if it is the source chunk or covers the same section and pages, for example the other half of a split table.

Public results on the AWS Well-Architected corpus will be published here after the first full run.

## Project layout

```text
src/ragkit/
  providers/    LLM, embedding and rerank contracts + Bedrock and OpenAI implementations
  ingestion/    Docling parsing, element extraction, structure-aware chunking
  retrieval/    tokenizer, BM25 encoder, Pinecone store, hybrid retriever, query hints, rerank
  evaluation/   metrics, golden set generation, configuration comparison
  api/          FastAPI service
  corpus.py     corpus.yaml loader and per-corpus paths
  pipeline.py   download, ingest, index and retriever wiring
  cli.py        `ragkit` command line
corpora/        corpus configurations (documents are downloaded, never committed)
tests/          offline unit tests, all external services faked
```

## Engineering practices

- Typed Python with Protocol-based dependency injection; every external service is replaceable by a fake.
- Offline test suite (no network, no keys) running in GitHub Actions with Ruff lint and format checks.
- A CI guard fails the build if documents, datasets or `.env` files are ever tracked by git.
- Semantic commit history, so each capability can be reviewed as one change.

## Roadmap

- [x] Structure-aware ingestion for any Docling-supported format
- [x] Hybrid retrieval with RRF, query hints and namespaces
- [x] Cross-encoder reranking and retrieval evaluation harness
- [ ] `/ask` endpoint: grounded answers with page-level citations
- [ ] Generation evals with RAGAS (faithfulness, answer relevancy, context precision)
- [ ] Guardrails: prompt-injection screening, PII redaction, refusal when context is missing
- [ ] Docker image and AWS deployment
- [ ] Azure OpenAI and pgvector providers

## License

MIT, see [`pyproject.toml`](pyproject.toml).
