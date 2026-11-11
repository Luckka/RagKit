"""OpenAI providers: GPT (generation) and text-embedding-3 (embeddings)."""

from openai import OpenAI

from ragkit.config import settings


class OpenAILLM:
    def __init__(self, model: str | None = None):
        self.model = model or settings.openai_llm_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


class OpenAIEmbeddings:
    dimension = 1536

    def __init__(self, model: str | None = None):
        self.model = model or settings.openai_embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
