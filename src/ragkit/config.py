"""Runtime settings, loaded from environment variables or a local .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Which provider serves each role: "bedrock" or "openai"
    llm_provider: str = "bedrock"
    embedding_provider: str = "bedrock"

    # AWS. An empty profile falls back to the default credential chain (env vars, SSO, IAM role).
    aws_region: str = "us-east-1"
    aws_profile: str | None = None

    # Models are configuration, never hardcoded
    bedrock_llm_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    bedrock_embedding_model: str = "cohere.embed-multilingual-v3"
    bedrock_rerank_model: str = "cohere.rerank-v3-5:0"
    openai_llm_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    openai_api_key: str = ""
    pinecone_api_key: str = ""

    # Pinecone serverless. One index pair per embedding provider; each corpus is a namespace.
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_index_prefix: str = "ragkit"

    @property
    def dense_index_name(self) -> str:
        # Dimension depends on the embedding model (Cohere 1024, OpenAI 1536)
        return f"{self.pinecone_index_prefix}-dense-{self.embedding_provider}"

    @property
    def sparse_index_name(self) -> str:
        return f"{self.pinecone_index_prefix}-sparse"


settings = Settings()
