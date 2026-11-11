"""Corpus-as-config: everything specific to a document collection lives in corpus.yaml.

    corpora/<name>/corpus.yaml   versioned: where documents come from and how to treat them
    data/<name>/raw/             downloaded documents (never committed)
    data/<name>/processed/       Docling cache, chunks, BM25 vocabulary, hint values
    data/<name>/eval/            golden set and evaluation results

The pipeline code never mentions a specific corpus. A new collection is a new YAML file,
plus, only if needed, a Profile class for document-specific cleanup.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ragkit.ingestion.elements import HeadingRule, TagRule
from ragkit.retrieval.query_hints import QueryHinter, RegexHinter

CORPORA_DIR = Path("corpora")
DATA_DIR = Path("data")


@dataclass
class Document:
    file: str
    url: str | None = None


@dataclass
class HintConfig:
    field: str
    pattern: str
    template: str = "{0}"


@dataclass
class Corpus:
    name: str
    description: str = ""
    language: str = "en"
    documents: list[Document] = field(default_factory=list)
    max_chars: int = 1600
    min_chars: int = 400
    profile: str | None = None  # "package.module:ClassName"
    hints: list[HintConfig] = field(default_factory=list)
    heading_rules: list[HeadingRule] = field(default_factory=list)
    default_heading_level: int | None = None
    tags: list[TagRule] = field(default_factory=list)
    golden_size: int = 25
    golden_seed: int = 42
    dedup_threshold: float | None = None  # shingle Jaccard; None disables duplicate detection
    dedup_drop: bool = False  # keep only the first copy, in document order

    @classmethod
    def load(cls, name: str, root: Path = CORPORA_DIR) -> Corpus:
        raw = yaml.safe_load((root / name / "corpus.yaml").read_text(encoding="utf-8"))
        chunking = raw.pop("chunking", {}) or {}
        golden = raw.pop("golden", {}) or {}
        headings = raw.pop("headings", {}) or {}
        dedup = raw.pop("dedup", {}) or {}
        return cls(
            name=raw["name"],
            description=raw.get("description", ""),
            language=raw.get("language", "en"),
            documents=[Document(**d) for d in raw.get("documents", [])],
            max_chars=chunking.get("max_chars", 1600),
            min_chars=chunking.get("min_chars", 400),
            profile=raw.get("profile"),
            hints=[HintConfig(**h) for h in raw.get("hints", []) or []],
            heading_rules=[HeadingRule(**r) for r in headings.get("rules", []) or []],
            default_heading_level=headings.get("default_level"),
            tags=[TagRule(**t) for t in raw.get("tags", []) or []],
            golden_size=golden.get("size", 25),
            golden_seed=golden.get("seed", 42),
            dedup_threshold=dedup.get("threshold"),
            dedup_drop=dedup.get("drop", False),
        )

    # --- paths -------------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return DATA_DIR / self.name

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.processed_dir / "bm25.json"

    @property
    def hint_values_path(self) -> Path:
        return self.processed_dir / "hint_values.json"

    # --- plug-ins ----------------------------------------------------------------------
    def load_profile(self):
        if not self.profile:
            return None
        module, _, cls_name = self.profile.partition(":")
        return getattr(importlib.import_module(module), cls_name)()

    def build_hinter(self) -> QueryHinter | None:
        if not self.hints or not self.hint_values_path.exists():
            return None
        known = json.loads(self.hint_values_path.read_text())
        hinters = [
            RegexHinter(h.field, h.pattern, set(known.get(h.field, [])), h.template)
            for h in self.hints
        ]
        return MultiHinter(hinters)


class MultiHinter:
    def __init__(self, hinters: list[QueryHinter]):
        self.hinters = hinters

    def hints(self, question: str) -> list[dict]:
        return [h for hinter in self.hinters for h in hinter.hints(question)]
