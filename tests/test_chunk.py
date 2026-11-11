from ragkit.ingestion.chunk import HeadingChunker, compact_table, split_table
from ragkit.ingestion.elements import Element


def el(kind, text, page=1, headings=("Security",), extra=None):
    return Element(kind, text, page, "guide.pdf", list(headings), extra or {})


def test_compact_table_removes_padding():
    md = "| a    | b |\n|------|---|\n| 1    | 2 |"
    assert compact_table(md) == "| a | b |\n| --- | --- |\n| 1 | 2 |"


def test_split_table_repeats_header():
    md = "| h |\n| --- |\n" + "\n".join(f"| row{i} |" for i in range(50))
    parts = split_table(md, max_chars=120)
    assert len(parts) > 1
    assert all(p.startswith("| h |\n| --- |") for p in parts)


def test_chunks_never_mix_sections_and_carry_breadcrumb():
    els = [
        el("heading", "Security"),
        el("text", "Enable MFA." * 20),
        el("heading", "Reliability", headings=("Reliability",)),
        el("text", "Back up data." * 20, page=2, headings=("Reliability",)),
    ]
    chunks = HeadingChunker(min_chars=0).chunk(els)
    assert [c.section for c in chunks] == ["Security", "Reliability"]
    assert chunks[0].text.startswith("guide.pdf > Security\n\n")
    assert (chunks[1].page_start, chunks[1].page_end) == (2, 2)


def test_long_section_splits_under_cap_with_unique_ids():
    els = [el("text", f"Paragraph {i}. " + "x" * 300) for i in range(12)]
    chunks = HeadingChunker(max_chars=1000).chunk(els)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 + 50 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_extra_metadata_splits_groups_and_is_flattened():
    els = [
        el("text", "a" * 500, extra={"code": "X-1"}),
        el("text", "b" * 500, extra={"code": "X-2"}),
    ]
    chunks = HeadingChunker().chunk(els)
    assert [c.extra["code"] for c in chunks] == ["X-1", "X-2"]
    assert chunks[0].metadata()["code"] == "X-1"


def test_tiny_sibling_section_is_merged():
    els = [
        el("text", "Overview text " * 30, headings=("Pillar", "Overview")),
        el("text", "Short goal.", headings=("Pillar", "Goal")),
    ]
    chunks = HeadingChunker().chunk(els)
    assert len(chunks) == 1
    assert "### Goal\nShort goal." in chunks[0].text
