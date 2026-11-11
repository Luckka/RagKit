"""Query understanding: detect metadata a question refers to, to boost matching chunks.

A QueryHinter turns a question into Pinecone metadata filters. The retriever runs extra
filtered searches for each hint and fuses them into RRF as additional ranked lists. That is
a *soft* filter: matching chunks rise, but nothing else is excluded.

Corpora plug in their own hinters. The built-in RegexHinter covers the common case of
identifiers that appear in questions (ticket ids, module codes, RFC numbers, service names).
"""

import re
from typing import Protocol


class QueryHinter(Protocol):
    def hints(self, question: str) -> list[dict]:
        """Return metadata filters (possibly empty) implied by the question."""
        ...


class RegexHinter:
    """Finds identifiers with a regex, normalizes them, keeps only values known in the corpus.

    Example: field="module", pattern=r"\\b([A-Za-z]{2,3})[-\\s]?(\\d{1,2})\\b",
    template="{0}-{1:0>2}" maps "sec1", "SEC 1" and "SEC-01" to "SEC-01".
    """

    def __init__(self, field: str, pattern: str, known: set[str], template: str = "{0}"):
        self.field = field
        self.pattern = re.compile(pattern)
        self.known = known
        self.template = template

    def values(self, question: str) -> list[str]:
        found: list[str] = []
        for m in self.pattern.finditer(question):
            groups = [g.upper() for g in m.groups()] or [m.group(0)]
            value = self.template.format(*groups)
            if value in self.known and value not in found:
                found.append(value)
        return found

    def hints(self, question: str) -> list[dict]:
        values = self.values(question)
        return [{self.field: {"$in": values}}] if values else []
