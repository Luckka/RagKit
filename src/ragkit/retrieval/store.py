"""Pinecone access: one dense (semantic) index and one sparse (BM25) index.

Separate indexes is Pinecone's recommended hybrid design: each searches its own way and
fusion (RRF) happens in our code. Every corpus lives in its own namespace, so many corpora
share one index pair (the free tier caps the number of indexes).
"""

import time

from pinecone import Pinecone, ServerlessSpec

from ragkit.config import settings


def client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_indexes(dimension: int) -> None:
    """Create both indexes if missing (idempotent, like a migration)."""
    pc = client()
    spec = ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region)
    if not pc.has_index(settings.dense_index_name):
        pc.create_index(
            name=settings.dense_index_name, dimension=dimension, metric="cosine", spec=spec
        )
    if not pc.has_index(settings.sparse_index_name):
        pc.create_index(
            name=settings.sparse_index_name, metric="dotproduct", vector_type="sparse", spec=spec
        )
    for name in (settings.dense_index_name, settings.sparse_index_name):
        while not pc.describe_index(name).status.ready:
            time.sleep(2)


def _index(name: str):
    pc = client()
    return pc.Index(host=pc.describe_index(name).host)


def dense_index():
    return _index(settings.dense_index_name)


def sparse_index():
    return _index(settings.sparse_index_name)
