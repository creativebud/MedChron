# docling_runner.py
import os, json, pathlib
from typing import Dict, Any, List
from datetime import datetime

def run_docling_convert(pdf_path: str, out_dir: str, do_ocr: bool = False, do_tables: bool = False) -> Dict[str, Any]:
    """
    Use the Docling Python API (not shell) to parse PDF into:
      - Markdown (.md)
      - Lossless JSON (.json)
    
    Args:
        pdf_path: Path to PDF file
        out_dir: Output directory for MD and JSON
        do_ocr: Enable OCR (slower, use for scanned PDFs)
        do_tables: Enable table structure extraction (slower)
    
    Returns dict with paths.
    """
    out = {"ok": False, "md": None, "json": None, "error": None}
    pdf = pathlib.Path(pdf_path)
    od = pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    stem = pdf.stem

    try:
        # Import inside try block to catch missing dependencies
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as ie:
            out["error"] = f"Docling import failed: {ie}. Ensure accelerate and torch are installed."
            return out
        
        # Force CPU-only mode to avoid accelerate device_map issues
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        
        # Configure processing options based on user preferences
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr  # User-controlled OCR
        pipeline_options.do_table_structure = do_tables  # User-controlled table extraction
        pipeline_options.images_scale = 1.0  # Don't upscale images
        pipeline_options.generate_page_images = False  # Don't generate page images
        pipeline_options.generate_picture_images = False  # Don't extract pictures
        
        # Create converter with optimized settings
        conv = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        mode = "with OCR" if do_ocr else "fast mode (no OCR)"
        print(f"Converting {pdf.name} with Docling ({mode})...")
        result = conv.convert(pdf_path)  # parses structure, OCR if enabled
        # Save markdown
        md_path = od / f"{stem}.md"
        md_text = result.document.export_to_markdown()
        md_path.write_text(md_text, encoding="utf-8")
        # Save JSON (lossless structure)
        json_path = od / f"{stem}.json"
        json_dict = result.document.export_to_dict()
        json_path.write_text(json.dumps(json_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        out.update({"ok": True, "md": str(md_path), "json": str(json_path)})
    except Exception as e:
        import traceback
        out["error"] = f"{str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
    return out

def chunk_markdown(md_text: str, max_chunks: int = 10000, target_chunk_size: int = 800) -> List[Dict[str, Any]]:
    """
    Simple heading-aware chunker for Markdown. Keeps metadata stubs for page/section when present.
    
    Args:
        md_text: Markdown text to chunk
        max_chunks: Maximum number of chunks to generate (prevents memory overflow)
        target_chunk_size: Target lines per chunk (increases if needed to stay under max_chunks)
    """
    import re
    lines = md_text.splitlines()
    chunks = []
    buf, header, page = [], None, None
    min_lines_per_chunk = 10  # Minimum lines before forcing a flush

    def flush():
        if not buf: 
            return
        if len(chunks) >= max_chunks:
            return  # Stop creating chunks if we hit the limit
        text = "\n".join(buf).strip()
        if text:
            chunks.append({
                "text": text[:8000],  # Increased from 4000 to hold more content per chunk
                "metadata": { "section": header or "", "page_label": page or "" }
            })
        buf.clear()

    for ln in lines:
        # Stop processing if we've hit the chunk limit
        if len(chunks) >= max_chunks:
            print(f"⚠️  Warning: Reached max_chunks limit of {max_chunks}.")
            print(f"   Document processing stopped. Consider processing document in sections.")
            break
            
        # page label heuristics
        m = re.search(r"\bpage\s+(\d+)\b", ln.lower())
        if m:
            page = m.group(1)
        if ln.startswith("#"):
            # Only flush if we have enough content
            if len(buf) >= min_lines_per_chunk:
                flush()
            header = ln.lstrip("# ").strip()
            buf.append(ln)  # Include the header in the chunk
            continue
        buf.append(ln)
        
        # Also flush if buffer gets too large (prevents mega-chunks)
        if len(buf) >= target_chunk_size:
            flush()
            
    flush()
    
    print(f"✓ Generated {len(chunks)} chunks from document ({len(lines)} lines)")
    
    return chunks
