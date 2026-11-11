"""BM25 as sparse vectors, so a vector database can compute keyword relevance.

The BM25 score of a document for a query factors into a dot product of two sparse vectors:

- document side: saturated, length-normalized term frequency
      tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
- query side: inverse document frequency of each query term
      log((N - df + 0.5) / (df + 0.5) + 1)

So a sparse index with the dotproduct metric returns BM25 rankings natively. Corpus
statistics (the "fit") are saved to JSON and reused to encode queries, like a trained model.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from ragkit.retrieval.text import tokenize


def token_id(token: str) -> int:
    """Stable uint32 index for a token (sparse indices must be 32-bit integers)."""
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, language: str = "en"):
        self.k1 = k1
        self.b = b
        self.language = language
        self.n_docs = 0
        self.avgdl = 0.0
        self.df: dict[str, int] = {}

    def _tokens(self, text: str) -> list[str]:
        return tokenize(text, self.language)

    def fit(self, corpus: list[str]) -> "BM25Encoder":
        df: Counter[str] = Counter()
        total = 0
        for text in corpus:
            tokens = self._tokens(text)
            total += len(tokens)
            df.update(set(tokens))
        self.n_docs = len(corpus)
        self.avgdl = total / max(self.n_docs, 1)
        self.df = dict(df)
        return self

    def idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    @staticmethod
    def _to_sparse(weights: dict[str, float]) -> dict:
        merged: dict[int, float] = {}
        for tok, w in weights.items():  # rare hash collisions: weights are summed
            i = token_id(tok)
            merged[i] = merged.get(i, 0.0) + w
        items = sorted(merged.items())
        return {"indices": [i for i, _ in items], "values": [v for _, v in items]}

    def encode_document(self, text: str) -> dict:
        tf = Counter(self._tokens(text))
        dl = sum(tf.values())
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        return self._to_sparse({t: f * (self.k1 + 1) / (f + norm) for t, f in tf.items()})

    def encode_query(self, text: str) -> dict:
        # Terms never seen in the corpus cannot help ranking
        terms = {t for t in self._tokens(text) if t in self.df}
        return self._to_sparse({t: self.idf(t) for t in terms})

    def save(self, path: Path) -> None:
        data = {
            "k1": self.k1,
            "b": self.b,
            "language": self.language,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "df": self.df,
        }
        path.write_text(json.dumps(data, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "BM25Encoder":
        data = json.loads(path.read_text())
        enc = cls(k1=data["k1"], b=data["b"], language=data.get("language", "en"))
        enc.n_docs, enc.avgdl, enc.df = data["n_docs"], data["avgdl"], data["df"]
        return enc
