"""Factories that pick a provider implementation from settings."""

from ragkit.config import settings
from ragkit.providers.base import EmbeddingProvider, LLMProvider, RerankProvider

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "RerankProvider",
    "get_embeddings",
    "get_llm",
    "get_reranker",
]


def get_llm(provider: str | None = None) -> LLMProvider:
    if (provider or settings.llm_provider) == "openai":
        from ragkit.providers.openai_provider import OpenAILLM

        return OpenAILLM()
    from ragkit.providers.bedrock import BedrockLLM

    return BedrockLLM()


def get_embeddings(provider: str | None = None) -> EmbeddingProvider:
    if (provider or settings.embedding_provider) == "openai":
        from ragkit.providers.openai_provider import OpenAIEmbeddings

        return OpenAIEmbeddings()
    from ragkit.providers.bedrock import BedrockEmbeddings

    return BedrockEmbeddings()


def get_reranker() -> RerankProvider:
    # Cohere Rerank via Bedrock is the only implementation; OpenAI has no native reranker.
    from ragkit.providers.bedrock import BedrockReranker

    return BedrockReranker()
