"""Document parsing with Docling: PDF, DOCX, PPTX, HTML and Markdown into one document model.

Docling keeps layout (reading order, page numbers, tables as structured grids), which is what
makes structure-aware chunking and page-level citations possible.
"""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument


def converter(ocr: bool = False) -> DocumentConverter:
    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def parse(path: Path, cache_dir: Path, conv: DocumentConverter | None = None) -> DoclingDocument:
    """Parse once and cache as Docling JSON (parsing is the slowest step of ingestion)."""
    cache = cache_dir / f"{path.stem}.docling.json"
    if cache.exists():
        return DoclingDocument.load_from_json(cache)
    doc = (conv or converter()).convert(path).document
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(cache)
    return doc
