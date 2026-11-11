from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragkit.generation.answer import Answerer

app = FastAPI(title="RagKit", description="Hybrid RAG over any document collection")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    corpus: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")


class Source(BaseModel):
    ref: int
    source: str
    section: str
    page_start: int
    page_end: int


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]


@lru_cache(maxsize=8)
def _answerer(corpus_name: str) -> Answerer:
    from ragkit import pipeline
    from ragkit.corpus import Corpus
    from ragkit.providers import get_llm, get_reranker

    corpus = Corpus.load(corpus_name)
    return Answerer(pipeline.retriever(corpus), get_llm(), get_reranker())


def get_answerer_factory() -> Callable[[str], Answerer]:
    return _answerer


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


AnswererFactory = Annotated[Callable[[str], Answerer], Depends(get_answerer_factory)]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, factory: AnswererFactory) -> AskResponse:
    try:
        answerer = factory(req.corpus)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown corpus '{req.corpus}'") from exc
    result = answerer.ask(req.question)
    return AskResponse(
        answer=result.answer,
        grounded=result.grounded,
        sources=[
            Source(
                ref=c.ref,
                source=c.source,
                section=c.section,
                page_start=c.page_start,
                page_end=c.page_end,
            )
            for c in result.citations
        ],
    )
