"""AWS Bedrock providers: Claude (generation), Cohere Embed and Cohere Rerank."""

import json

import boto3
from botocore.config import Config

from ragkit.config import settings

COHERE_MAX_BATCH = 96  # Cohere Embed accepts at most 96 texts per call

# Adaptive retries back off automatically on ThrottlingException (low default Bedrock quotas).
RETRY = Config(retries={"mode": "adaptive", "max_attempts": 10})


def _client():
    session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
    return session.client("bedrock-runtime", config=RETRY)


class BedrockLLM:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or settings.bedrock_llm_model
        self.client = _client()

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 1024},
        )
        return resp["output"]["message"]["content"][0]["text"]


class BedrockEmbeddings:
    dimension = 1024

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or settings.bedrock_embedding_model
        self.client = _client()

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        body = json.dumps({"texts": texts, "input_type": input_type, "truncate": "END"})
        resp = self.client.invoke_model(modelId=self.model_id, body=body)
        return json.loads(resp["body"].read())["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), COHERE_MAX_BATCH):
            out.extend(self._embed(texts[i : i + COHERE_MAX_BATCH], "search_document"))
        return out

    def embed_query(self, text: str) -> list[float]:
        # Asymmetric embeddings: queries and documents use different input types
        return self._embed([text], "search_query")[0]


class BedrockReranker:
    """Cohere Rerank 3.5: reads query and document together and scores relevance."""

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or settings.bedrock_rerank_model
        self.client = _client()

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        body = json.dumps(
            {
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "api_version": 2,
            }
        )
        resp = self.client.invoke_model(modelId=self.model_id, body=body)
        return [
            (r["index"], r["relevance_score"]) for r in json.loads(resp["body"].read())["results"]
        ]
