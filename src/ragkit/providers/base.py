"""Provider contracts. Any implementation that matches these Protocols plugs in."""

from typing import Protocol


class LLMProvider(Protocol):
    """Generates text from a system instruction and a user prompt."""

    def generate(self, system: str, prompt: str) -> str: ...


class EmbeddingProvider(Protocol):
    """Turns text into vectors. Documents and queries may be embedded differently."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RerankProvider(Protocol):
    """Cross-encoder that scores (query, document) pairs.

    Returns (index into documents, relevance score), most relevant first.
    """

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]: ...
