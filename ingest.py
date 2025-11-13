# ingest.py
import os, pathlib
from typing import List
from docling_runner import run_docling_convert, chunk_markdown
from rag_store import upsert_chunks

def ingest_pdf_to_chroma(pdf_path: str, workspace_dir: str, do_ocr: bool = False, do_tables: bool = False) -> dict:
    """
    1) Docling → md/json (stored under workspace/structured/<stem>)
    2) Chunk md
    3) Upsert chunks to Chroma w/ metadata (doc_name, page, section, abs_path)
    
    Args:
        pdf_path: Path to PDF file
        workspace_dir: Workspace directory
        do_ocr: Enable OCR (slower, for scanned PDFs)
        do_tables: Enable table structure extraction (slower)
    """
    pdf = pathlib.Path(pdf_path)
    struct_dir = pathlib.Path(workspace_dir) / "structured" / pdf.stem
    struct_dir.mkdir(parents=True, exist_ok=True)
    conv = run_docling_convert(str(pdf), str(struct_dir), do_ocr=do_ocr, do_tables=do_tables)
    if not conv.get("ok"):
        return {"ok": False, "error": conv.get("error")}

    md = pathlib.Path(conv["md"]).read_text(encoding="utf-8")
    chunks = chunk_markdown(md)
    # enrich metadata
    for c in chunks:
        md_meta = c.setdefault("metadata", {})
        md_meta["doc_name"] = pdf.name
        md_meta["abs_path"] = str(pdf.resolve())
    upsert_chunks(chunks)
    return {"ok": True, "md": conv["md"], "json": conv["json"], "count_chunks": len(chunks)}
