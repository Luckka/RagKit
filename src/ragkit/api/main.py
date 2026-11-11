from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="RagKit", description="Hybrid RAG over any document collection")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    corpus: str


class Source(BaseModel):
    source: str
    section: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    # Grounded generation with citations lands in the next milestone
    return AskResponse(answer="not implemented yet", sources=[])
