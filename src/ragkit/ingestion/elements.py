"""Flatten a DoclingDocument into clean, ordered elements that carry their heading path.

Generic rules that hold for most documents:
- page headers and footers are noise ("Page 3 of 120", confidentiality footers);
- every element remembers the headings above it, so chunks know where they came from;
- tables are exported as Markdown and never split here.

Document-specific rules (a numbering scheme, running titles, domain codes) belong in a
Profile, which post-processes these elements without forking the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TableItem, TextItem

NOISE_LABELS = {"page_header", "page_footer"}


@dataclass
class Element:
    kind: str  # heading | text | list_item | code | table | ...
    text: str
    page: int
    source: str
    headings: list[str] = field(default_factory=list)  # path of headings above this element
    extra: dict = field(default_factory=dict)  # profile-defined metadata (e.g. a product code)


class Profile(Protocol):
    """Corpus-specific post-processing over generic elements."""

    def process(self, elements: list[Element]) -> list[Element]: ...


def extract(doc: DoclingDocument, source: str) -> list[Element]:
    elements: list[Element] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    for item, _ in doc.iterate_items():  # true reading order
        if not getattr(item, "prov", None):
            continue
        page = item.prov[0].page_no
        path = [t for _, t in stack]
        if isinstance(item, TableItem):
            elements.append(Element("table", item.export_to_markdown(doc), page, source, path))
            continue
        if not isinstance(item, TextItem) or item.label.value in NOISE_LABELS:
            continue
        text = item.text.strip()
        if not text:
            continue
        if isinstance(item, SectionHeaderItem):
            level = getattr(item, "level", 1) or 1
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, text)]
            elements.append(Element("heading", text, page, source, [t for _, t in stack]))
            continue
        elements.append(Element(item.label.value, text, page, source, path))
    return elements
