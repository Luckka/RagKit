from ragkit.evaluation.metrics import is_relevant
from ragkit.ingestion.chunk import Chunk
from ragkit.ingestion.dedup import dedup, find_twins, location, parse_location

SHARED = " ".join(f"word{i}" for i in range(60))


def chunk(cid, source, page, body):
    return Chunk(cid, f"Crumb {cid}\n{body}", source, "S", page, page, False, {})


def corpus():
    return [
        chunk("a", "overview.pdf", 10, SHARED),
        chunk("b", "other.pdf", 3, "completely different text about something else entirely"),
        chunk("c", "guide.pdf", 42, SHARED + " extra"),
    ]


def test_location_round_trip():
    assert parse_location(location("my#file.pdf", 4, 6)) == ("my#file.pdf", 4, 6)


def test_twins_ignore_breadcrumb_and_are_symmetric():
    assert find_twins(corpus(), 0.8) == [[2], [], [0]]


def test_annotate_keeps_all_and_records_locations():
    out = dedup(corpus(), 0.8)
    assert [c.chunk_id for c in out] == ["a", "b", "c"]
    assert out[0].extra["also_in"] == ["guide.pdf#p42-42"]
    assert "also_in" not in out[1].extra


def test_drop_keeps_first_copy_in_document_order():
    out = dedup(corpus(), 0.8, drop=True)
    assert [c.chunk_id for c in out] == ["a", "b"]
    assert out[0].extra["also_in"] == ["guide.pdf#p42-42"]


def test_relevance_accepts_twin_location_on_source_match():
    retrieved = {"chunk_id": "a", "source": "overview.pdf", "page_start": 10, "page_end": 10}
    retrieved["also_in"] = ["guide.pdf#p42-42"]
    expected = {"source_chunk_id": "c", "source": "guide.pdf", "page_start": 42, "page_end": 42}
    assert is_relevant(retrieved, expected, match_fields=("source",))
    assert not is_relevant(
        retrieved, {**expected, "section": "S"}, match_fields=("source", "section")
    )
