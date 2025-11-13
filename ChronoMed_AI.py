#!/usr/bin/env python3
# MedRAG: Local OCR → RAG → Narrative & Chronology (HIPAA-friendly)
# All processing is local: OCR (ocrmypdf), text extraction (PyMuPDF),
# local embeddings via sentence-transformers (nomic) and LLM via Ollama (llama3.1:8b).
# -------------------------------------------------------------------

# CRITICAL: Set environment variables BEFORE any imports to force CPU-only mode
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import re, io, sys, json, time, math, pathlib, tempfile, shutil, subprocess, html
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd

# ------------------------------ Config --------------------------------

APP_NAME = "ChronoMed AI : Llama 3.1 Medical Summary & Chronology"
OLLAMA_URL = "http://127.0.0.1:11434"   # localhost only (HIPAA)
LLM_MODEL  = "llama3.1:8b"
EMB_MODEL  = "nomic-embed-text"
EMB_DIM    = 768
NARRATIVE_MODEL = LLM_MODEL  # model used specifically for narrative filtering/generation

# Docling configuration: use the docling CLI from a venv or system PATH. Do not enable VLM artifacts.
DOCLING_PYTHON = "/Users/bernardpelgrimiii/Documents/Python Projects/venv_Docling/bin/python"

# ------------------------------ OCR (DEPRECATED) ----------------------
# NOTE: OCR functionality has been removed from the UI since Docling handles
# OCR internally when processing PDFs. The code below is preserved but unused.
# Docling's built-in OCR is preferred as it's integrated with structure extraction.
# 
# If you need standalone OCR preprocessing, you can re-enable this, but it
# duplicates Docling's OCR capabilities and adds unnecessary processing time.
# ----------------------------------------------------------------------

# def have_ocrmypdf() -> bool:
#     return shutil.which("ocrmypdf") is not None

# def ocr_pdf(in_path: str) -> str:
#     """
#     DEPRECATED: Docling handles OCR internally.
#     
#     If ocrmypdf is available, produce a searchable copy and return its path.
#     Otherwise return original path. Keeps files local; locks perms to 600.
#     """
#     in_path = str(in_path)
#     if not have_ocrmypdf():
#         return in_path
#     out_dir = pathlib.Path(tempfile.mkdtemp(prefix="ocr_"))
#     out_path = out_dir / (pathlib.Path(in_path).stem + "_ocr.pdf")
#     cmd = [
#         "ocrmypdf",
#         "--skip-text",         # don't re-OCR pages that already have text
#         "--rotate-pages",
#         "--deskew",
#         "--optimize", "0",
#         in_path, str(out_path)
#     ]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         os.chmod(out_path, 0o600)
#         return str(out_path)
#     except Exception:
#         # fall back to original if OCR fails
#         return in_path

# --------------------------- PDF text ---------------------------------

def extract_pages_text(pdf_path: str) -> List[str]:
    """Prefer PyMuPDF; fall back to pypdf."""
    p = str(pdf_path)
    try:
        import fitz
        doc = fitz.open(p)
        out = []
        for page in doc:
            try:
                out.append(page.get_text("text") or "")
            except Exception:
                out.append("")
        doc.close()
        return out
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        r = PdfReader(p)
        return [(pg.extract_text() or "") for pg in r.pages]
    except Exception:
        return [""]

# --------------------------- Date parsing ------------------------------

import dateparser

DATE_PAT = re.compile(
    r"\b(19|20)\d{2}[-/\.](0[1-9]|1[0-2])[-/\.](0[1-9]|[12]\d|3[01])\b|"   # YYYY-MM-DD
    r"\b(0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])[-/\.](\d{2,4})\b|"   # MM/DD/YYYY or MM-DD-YY
    r"\b([A-Za-z]{3,9})\s+(0?[1-9]|[12]\d|3[01]),?\s+(19\d{2}|20\d{2})\b" # Month 2, 2021
)

TIME_PAT = re.compile(r"\b([01]?\d|2[0-3]):?([0-5]\d)\b")  # HH:MM or HHMM

def parse_first_date(text: str) -> Optional[date]:
    m = DATE_PAT.search(text or "")
    if not m:
        return None
    try:
        d = dateparser.parse(m.group(0))
        return d.date() if d else None
    except Exception:
        return None

def parse_first_time(text: str) -> Optional[str]:
    m = TIME_PAT.search(text or "")
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    return f"{hh:02d}:{mm:02d}"

# ---------------------------- HTTP & RAG Setup -----------------------------
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Module-level HTTP helper: requests.Session with retries/backoff and default timeout
class HTTP:
    def __init__(self, timeout: int = 8, retries: int = 3, backoff: float = 0.3):
        self.timeout = timeout
        self.session = requests.Session()
        # Retry on common transient server errors and rate limits
        retry = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST", "PUT", "DELETE", "HEAD")
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def post(self, *args, **kwargs):
        # Ensure a default timeout is always present unless explicitly overridden
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        return self.session.post(*args, **kwargs)


# shared HTTP instance used throughout the module (for LLM generation only)
http = HTTP(timeout=3, retries=2, backoff=0.2)


def _assert_local_url(url: str):
    if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost") or url.startswith("http://[::1]")):
        raise RuntimeError("External network requests are disabled for PHI safety.")


_assert_local_url(OLLAMA_URL)


# RAG store: ChromaDB persistent vector store with local embeddings
import rag_store
import ingest
import retrieval
import llm_chains


# ---------------------------- LLM (Ollama) ----------------------------

def ask_llama(prompt: str, model: str = LLM_MODEL, num_predict: int = 1600) -> str:
    payload = {
        "model": model,
        "prompt": "Reply in plain text only.\n\n" + prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": num_predict}
    }
    # Use shared http helper (has default timeout and retries); allow a slightly
    # longer timeout for generation but keep it bounded so UI doesn't hang forever.
    r = http.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

# --------------------------- Row model --------------------------------

from dataclasses import dataclass

@dataclass
class Row:
    dt: datetime
    doc_name: str
    page_index: int
    page_label: str
    page_section: Optional[str]
    summary: str          # full page/event excerpt
    entry_summary: str    # concise: first line
    author: str
    link: str

def first_line(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().splitlines()[0])[:400]


# Extract a concise event summary from a page or element of text.
# Strategy:
# 1. Look for short lines containing medical keywords (Diagnosis, Impression, Procedure, Medication, Result, etc.)
# 2. Otherwise split into sentences and return the first sentence that contains a keyword
# 3. Fallback to the first non-empty line or the first sentence truncated
KEYWORD_PATTERN = re.compile(
    r"\b(diagnosis|impression|assessment|plan|discharge|admit|admitted|presenting complaint|chief complaint|procedure|operation|operative|medication|medications|mg\b|mcg\b|tablet\b|pill\b|dose\b|lab|result|imaging|ct|mri|x-?ray|ekg|ecg|radiology|allergy|vital|bp\b|blood pressure|pulse|fever|temperature|systolic|diastolic)\b",
    re.I,
)


def extract_event_summary(text: str, max_len: int = 400) -> str:
    if not text:
        return ""
    # Normalize and split into lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 1) Check short lines for keywords
    for ln in lines:
        if len(ln) < 200 and KEYWORD_PATTERN.search(ln):
            return re.sub(r"\s+", " ", ln)[:max_len]

    # 2) Sentence-split and look for keywords
    # Simple sentence split by punctuation followed by space
    text_norm = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[\.\?!])\s+", text_norm)
    for s in sentences:
        if KEYWORD_PATTERN.search(s):
            return s.strip()[:max_len]

    # 3) Fallback to first non-empty line
    if lines:
        return re.sub(r"\s+", " ", lines[0])[:max_len]

    # 4) As a last resort, return the first sentence
    if sentences:
        return sentences[0].strip()[:max_len]

    return ""


def summarize_table_as_rows(table: Dict[str, Any], pdf_uri: str, default_date: date) -> List[Row]:
    """Convert a table JSON (from Docling or pdfplumber) into a list of Row objects.

    Expected table format (flexible): {
        'page': int,
        'header': [h1, h2, ...],
        'rows': [[c1, c2, ...], ...]
    }
    """
    out: List[Row] = []
    page_num = int(table.get("page", 0))
    headers = table.get("header") or []
    rows = table.get("rows") or []

    # Create a summary for each row; include all rows (do not truncate)
    for ridx, r in enumerate(rows):
        try:
            # Map header->cell when headers exist
            if headers and len(headers) == len(r):
                pairs = [f"{h}: {c}" for h, c in zip(headers, r) if str(c).strip()]
                summary = "; ".join(pairs)
            else:
                # Join first few cells
                cells = [str(c).strip() for c in r if str(c).strip()]
                summary = " | ".join(cells[:5])

            if not summary:
                continue

            # Try to parse a date/time from any cell
            dt = None
            for c in (headers or []) + list(map(str, r)):
                d = parse_first_date(str(c))
                if d:
                    tm = parse_first_time(str(c)) or "00:00"
                    dt = datetime.combine(d, datetime.strptime(tm, "%H:%M").time())
                    break

            if dt is None:
                # fallback to default_date at midnight
                dt = datetime.combine(default_date, datetime.strptime("00:00", "%H:%M").time())

            entry = extract_event_summary(summary)

            out.append(Row(
                dt=dt,
                doc_name=pathlib.Path(pdf_uri).stem,
                page_index=page_num,
                page_label=str(page_num + 1),
                page_section="Table",
                summary=summary,
                entry_summary=entry,
                author="",
                link=f"{pathlib.Path(pdf_uri).resolve().as_uri()}#page={page_num+1}",
            ))
        except Exception:
            continue

    return out


def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Attempt to extract tables from a PDF using pdfplumber as a fallback.

    Returns a list of table dicts with keys: page (0-based), header (list), rows (list of lists).
    If pdfplumber is not available or extraction fails, returns an empty list.
    """
    try:
        import pdfplumber
    except Exception:
        return []

    out_tables: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(pdf_path) as doc:
            for pi, page in enumerate(doc.pages):
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for t in tables:
                    # t is a list of rows (each row is a list of cells)
                    if not t:
                        continue
                    header = t[0]
                    rows = t[1:]
                    out_tables.append({"page": pi, "header": header, "rows": rows})
    except Exception:
        return []

    return out_tables

def detect_section(text: str) -> Optional[str]:
    # very light heuristics; extend as needed
    CANDIDATES = [
        "History & Physical", "H&P", "Emergency Department", "ED Physician",
        "Nurses Notes", "Progress Note", "Consultation", "Operative", "Procedure",
        "Discharge Summary", "Radiology", "Laboratory", "Orders", "Flow Sheet", "Allergies",
        "Clinical Summary", "Medications", "Immunizations", "Vital Signs", "Assessment Plan",
        "Care Plan", "Medical History", "Encounters", "Observations", "Procedures" ,  
        "Medical Conditions", "Medication History", "Immunization History", "Allergy Information",
        "Recent Health Information", "Care Plans", "Medical Observations", "Medical Procedures",
        "Medical Encounters", "Medical Directives", "Advance Directives", "Social History",
        "Family History", "Review of Systems", "Physical Examination", "Chief Complaint",
        "History of Present Illness", "Past Medical History", "Past Surgical History",
        "Hospital Course", "Follow-up Instructions", "Discharge Instructions", "Treatment Plan",
        "Consultant Recommendations", "Nursing Assessment", "Dietary Instructions", "Activity Orders",
        "Respiratory Therapy", "Physical Therapy", "Occupational Therapy", "Speech Therapy",
        "Psychiatric Evaluation", "Substance Use History", "Preventive Care", "Screening Tests",
        "Health Maintenance", "Patient Education", "Care Coordination", "Social Work Notes",
        "Case Management", "Pharmacy Notes", "Infection Control", "Pain Management", "Wound Care",
        "Endocrine Management", "Cardiac Monitoring", "Neurological Assessment", "Gastrointestinal Notes",
        "Renal Function", "Hematology Reports", "Oncology Notes", "Palliative Care", "Rehabilitation Notes",
        "Discharge Medications", "Follow-up Appointments", "Referral Notes", "Advance Care Planning",
        "Insurance Information", "Billing Codes", "Legal Documentation", "Consent Forms", "Insurance Forms",
        "Code Blue", "Chronic Conditions", "Acute Conditions", "Surgical History", "Medication Reconciliation",
        "Diagnostic Imaging", "Laboratory Results", "Pathology Reports", "Basic Metabolic Panel", "Complete Blood Count",
        "Liver Function Tests", "Renal Panel", "Lipid Profile", "Thyroid Function Tests", "Coagulation Studies",
        "Urinalysis", "Blood Gas Analysis", "Microbiology Reports", "Electrolyte Panel", "Inflammatory Markers",
        "Tumor Markers", "Cardiac Enzymes", "Hormone Levels", "Nutritional Markers", "Immunological Tests", "Genetic Testing",
        "Triage Note", "Presenting Complaint", "ICU Transfer", "Code Status", "Subjective"
    ]
    tl = (text or "").lower()
    for c in CANDIDATES:
        if c.lower() in tl:
            return c
    return None

def detect_author(text: str) -> str:
    pats = [
        r"Signed by:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"Author:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"Dictated by:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    for p in pats:
        m = re.search(p, text or "")
        if m:
            return m.group(1)
    return ""

# ========================================================================
# LEGACY CODE BLOCK - NOT USED IN SIMPLIFIED UI
# ========================================================================
# The functions below (build_rows_from_pdf, build_records_for_rag, 
# render_chronology, etc.) were part of the old row-based processing pipeline.
# 
# They have been REPLACED by the modular system:
#   • ingest.py - PDF ingestion via Docling
#   • retrieval.py - RAG retrieval with seed queries
#   • llm_chains.py - LangChain-based narrative/chronology generation
#
# This code is preserved for reference but is NOT executed in the current
# simplified UI. To remove this legacy code entirely, delete lines 386-650.
# ========================================================================

def build_rows_from_pdf(pdf_path: str, save_dir: pathlib.Path) -> List[Row]:
    """Create rows by page: date/time detection + section/author heuristics."""
    texts = extract_pages_text(pdf_path)
    # Try to extract tables via pdfplumber as a fallback (may be empty)
    tables = extract_tables_from_pdf(pdf_path)
    rows: List[Row] = []
    ppath = pathlib.Path(pdf_path)
    link_base = ppath.resolve().as_uri()

    for i, t in enumerate(texts):
        d = parse_first_date(t) or datetime.now().date()
        tm = parse_first_time(t) or "00:00"
        dt = datetime.combine(d, datetime.strptime(tm, "%H:%M").time())
        section = detect_section(t)
        author = detect_author(t)
        rows.append(
            Row(
                dt=dt,
                doc_name=ppath.stem,
                page_index=i,
                page_label=str(i+1),
                page_section=section,
                summary=(t or "").strip(),
                entry_summary=extract_event_summary(t),
                author=author,
                link=f"{link_base}#page={i+1}",
            )
        )
        # Append any table-derived rows for this page
        for tbl in [x for x in tables if x.get("page", 0) == i]:
            try:
                tbl_rows = summarize_table_as_rows(tbl, pdf_path, d)
                rows.extend(tbl_rows)
            except Exception:
                continue
    return rows

# ------------------------- RAG evidence build -------------------------

def build_records_for_rag(rows: List[Row]) -> List[Dict[str, Any]]:
    recs = []
    for r in rows:
        when = r.dt.strftime("%Y-%m-%d %H:%M")
        sec = r.page_section or r.doc_name or "Document"
        line = f"{when} — {sec} — {first_line(r.summary)}"
        recs.append({
            "text": line,
            "doc_name": r.doc_name,
            "page_label": r.page_label,
            "author": r.author,
            "link": r.link,
        })
    return recs

def dedupe_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for h in sorted(hits, key=lambda x: (-x["score"], x["rank"])):
        key = (h["doc_name"], h["page_label"], h["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out

# ------------------------- Chronology export --------------------------

from jinja2 import Template

HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Chronology</title>
<style>
:root { color-scheme: light dark; }
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;margin:18px}
h1{margin-bottom:4px}
.small{color:#666;font-size:12px}
table{border-collapse:collapse;width:100%;border:1px solid #ddd}
th,td{border:1px solid #e6e6e6;padding:8px;text-align:left;vertical-align:top}
thead th{position:sticky;top:0;background:#fafafa;z-index:3}
a{color:#0366d6;text-decoration:none}
</style></head>
<body>
<h1>Medical Chronology</h1>
<div class="small">Generated {{ ts }}</div>
<table>
  <thead><tr><th>Date</th><th>Time</th><th>Event & Summary</th><th>Author</th><th>Page</th></tr></thead>
  <tbody>
  {% for r in rows %}
  <tr>
    <td>{{ r.date }}</td>
    <td>{{ r.time }}</td>
    <td><b>{{ r.section }}</b>{% if r.section %}: {% endif %}{{ r.entry }}</td>
    <td>{{ r.author }}</td>
    <td><a href="{{ r.link }}">p. {{ r.page }}</a></td>
  </tr>
  {% endfor %}
  </tbody>
</table>
</body></html>
"""

def build_chronology_df(rows: List[Row]) -> pd.DataFrame:
    rows_sorted = sorted(rows, key=lambda r: (r.dt, r.doc_name, r.page_index))
    data = [{
        "Date": r.dt.strftime("%Y-%m-%d"),
        "Time": r.dt.strftime("%H:%M"),
        "Event & Summary": f"{(r.page_section or r.doc_name or 'Document')}: {r.entry_summary}",
        "Author": r.author,
        "Page": r.page_label,
        "Link": r.link
    } for r in rows_sorted]
    return pd.DataFrame(data)

def chronology_html(rows: List[Row]) -> bytes:
    rows_sorted = sorted(rows, key=lambda r: (r.dt, r.doc_name, r.page_index))
    mapped = [{
        "date": r.dt.strftime("%Y-%m-%d"),
        "time": r.dt.strftime("%H:%M"),
        "section": r.page_section or r.doc_name or "",
        "entry": r.entry_summary,
        "author": r.author,
        "page": r.page_label,
        "link": r.link
    } for r in rows_sorted]
    html_out = Template(HTML_TEMPLATE).render(rows=mapped, ts=datetime.now().strftime("%Y-%m-%d %H:%M"))
    return html_out.encode("utf-8")

def chronology_csv(df: pd.DataFrame) -> bytes:
    # Keep columns in requested order + a hidden Link column
    export = df[["Date", "Time", "Event & Summary", "Author", "Page", "Link"]].copy()
    return export.to_csv(index=False).encode("utf-8")


def render_chronology(rows: List[Row]): 
    """Render chronology with collapse/expand controls for table groups.

    Groups consecutive Row objects that are table-derived (page_section == 'Table')
    and show a single summary line with an expander that reveals the full table rows.
    """
    if not rows:
        st.info("No chronology entries to show.")
        return

    rows_sorted = sorted(rows, key=lambda r: (r.dt, r.doc_name, r.page_index))
    st.subheader("Chronological Timeline")

    i = 0
    n = len(rows_sorted)
    while i < n:
        r = rows_sorted[i]
        if (r.page_section or "").lower() == "table":
            # Collect contiguous table rows for same doc and page
            group = []
            j = i
            while j < n and (rows_sorted[j].page_section or "").lower() == "table" and rows_sorted[j].doc_name == r.doc_name and rows_sorted[j].page_index == r.page_index:
                group.append(rows_sorted[j])
                j += 1

            # Summary row
            cols = st.columns([1, 1, 6, 2, 1])
            cols[0].write(r.dt.strftime("%Y-%m-%d"))
            cols[1].write(r.dt.strftime("%H:%M"))
            summary = f"Table ({len(group)} rows) — {first_line(group[0].summary)[:200]}"
            cols[2].write(summary)
            cols[3].write(r.author or "")
            cols[4].write(f"p. {r.page_label}")

            # Expander with full table rows (including Link)
            with st.expander(f"Show table details ({len(group)} rows)"):
                tdata = [{
                    "Date": rr.dt.strftime("%Y-%m-%d"),
                    "Time": rr.dt.strftime("%H:%M"),
                    "Event & Summary": rr.entry_summary,
                    "Author": rr.author,
                    "Page": rr.page_label,
                    "Link": rr.link,
                } for rr in group]
                if tdata:
                    st.dataframe(pd.DataFrame(tdata)[["Date", "Time", "Event & Summary", "Author", "Page", "Link"]], use_container_width=True)

            i = j
            continue

        # Non-table row: render inline
        cols = st.columns([1, 1, 6, 2, 1])
        cols[0].write(r.dt.strftime("%Y-%m-%d"))
        cols[1].write(r.dt.strftime("%H:%M"))
        cols[2].write((r.page_section or "") + (": " if r.page_section else "") + (r.entry_summary or ""))
        cols[3].write(r.author or "")
        cols[4].write(f"p. {r.page_label}")
        i += 1


def dedupe_rows_for_chronology(rows: List[Row]) -> List[Row]:
    """Remove duplicate entries from chronology based on datetime + summary."""
    seen = set()
    unique = []
    for r in rows:
        key = (r.dt, r.summary[:100])  # Use first 100 chars of summary
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def filter_rows_with_llama(rows: List[Row], batch_size: int = 200) -> List[Row]:
    """Use Llama to filter chronology entries to only pertinent medical information.
    
    Processes in batches to manage prompt length for large input sets.
    Returns only rows deemed medically relevant by the LLM.
    """
    if not rows:
        return []
    
    _assert_local_url(OLLAMA_URL)  # Safety check
    
    filtered = []
    total_batches = (len(rows) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(rows))
        batch = rows[start_idx:end_idx]
        
        # Build numbered list of entries for Llama
        entries_text = "\n".join([
            f"{i+1}. [{r.dt.strftime('%Y-%m-%d %H:%M')}] {r.entry_summary}"
            for i, r in enumerate(batch)
        ])
        
        prompt = f"""You are a medical AI assistant. Review this list of {len(batch)} medical record entries and identify which ones contain PERTINENT medical information (diagnoses, procedures, medications, test results, vital signs, treatments, symptoms, or significant clinical events).

Entries:
{entries_text}

Return ONLY the numbers of pertinent entries as a comma-separated list (e.g., "1,3,7,12"). If none are pertinent, return "none"."""

        try:
            resp = http.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": NARRATIVE_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip().lower()
            
            if answer == "none":
                continue
                
            # Parse numbers from response
            import re
            nums = re.findall(r'\d+', answer)
            pertinent_indices = {int(n) - 1 for n in nums if n.isdigit() and 0 <= int(n) - 1 < len(batch)}
            
            # Add pertinent rows
            for idx in pertinent_indices:
                if 0 <= idx < len(batch):
                    filtered.append(batch[idx])
                    
        except Exception as e:
            # On error, include all entries from this batch (fail-safe)
            print(f"Llama filtering error for batch {batch_idx+1}/{total_batches}: {e}")
            filtered.extend(batch)
    
    return filtered


def run_docling_convert(pdf_path: str, out_dir: str, timeout: int = 60*60) -> Tuple[bool, str]:
    """Run Docling CLI using the Docling venv. Returns (success, path_or_err).

    Sets env vars to point Docling at local VLM artifacts when enabled.
    """
    # Invoke the docling CLI to convert the PDF to JSON/MD. deliberately do
    # NOT set any VLM artifact environment variables so this runs without qwen.
    # Prefer the docling script from the provided venv, fallback to system PATH.
    docling_cli = None
    try:
        pv = pathlib.Path(DOCLING_PYTHON)
        cand = pv.parent / "docling"
        if cand.exists():
            docling_cli = str(cand)
    except Exception:
        docling_cli = None

    if docling_cli is None:
        docling_path = shutil.which("docling")
        if docling_path:
            docling_cli = docling_path

    if docling_cli is None:
        return False, "docling CLI not found (install docling or set DOCLING_PYTHON)"

    cmd = [docling_cli, str(pdf_path), "--to", "json", "--to", "md"]
    env = os.environ.copy()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, cwd=out_dir)
        out_lines = []
        if proc.stdout:
            for L in proc.stdout:
                out_lines.append(L.rstrip())
        ret = proc.wait(timeout=timeout)
        out_text = "\n".join(out_lines)
        if ret != 0:
            return False, out_text
        # Expect a .json file in out_dir
        stem = pathlib.Path(pdf_path).stem
        candidate = pathlib.Path(out_dir) / f"{stem}.json"
        if candidate.exists():
            return True, str(candidate)
        # fallback: any .json
        for p in pathlib.Path(out_dir).glob("*.json"):
            return True, str(p)
        return True, out_text
    except Exception as e:
        return False, str(e)


# (Docling JSON parsing removed — Docling/VLM integration has been removed.)

# ---------------------------- Streamlit UI ----------------------------

st.set_page_config(page_title="MedChron — Local Medical Summarizer", layout="wide")
st.title("📄 MedChron — Local Medical Summarizer (Docling → Chroma → Llama)")

st.warning("PHI notice: This app runs **entirely local**. Only upload documents you are allowed to process.")
phi_ok = st.checkbox("I acknowledge the PHI warning and accept responsibility for uploaded documents", value=False)

workspace = st.text_input("Workspace directory (stores structured files and Chroma index):", value=os.path.join(os.getcwd(), "workspace"))

# Processing options
with st.expander("⚙️ Processing Options (Advanced)"):
    enable_ocr = st.checkbox("Enable OCR (slower, use for scanned/image PDFs)", value=False, 
                            help="Disable for text-based PDFs to process 5-10x faster. Enable only if PDFs are scanned images.")
    enable_tables = st.checkbox("Enable table structure extraction (slower)", value=False,
                               help="Extract detailed table structures. Disable for faster processing if tables aren't critical.")
    st.info("💡 Tip: For fastest processing, keep both options disabled. Text-based medical records will still be extracted fully.")

uploaded = st.file_uploader("Upload medical PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded and phi_ok:
    st.subheader("Ingestion")
    os.makedirs(workspace, exist_ok=True)
    statuses = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, f in enumerate(uploaded):
        tmp = pathlib.Path(workspace) / "uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        pdf_path = tmp / f.name
        pdf_path.write_bytes(f.read())
        
        status_text.text(f"Processing {idx+1}/{len(uploaded)}: {f.name}...")
        start_time = time.time()
        
        mode_str = "fast mode" if not (enable_ocr or enable_tables) else "OCR mode" if enable_ocr else "with tables"
        with st.spinner(f"Converting {f.name} ({mode_str})..."):
            res = ingest.ingest_pdf_to_chroma(str(pdf_path), workspace, 
                                             do_ocr=enable_ocr, 
                                             do_tables=enable_tables)
        
        elapsed = time.time() - start_time
        statuses.append((f.name, res.get("ok"), res.get("error"), res.get("count_chunks"), elapsed))
        progress_bar.progress((idx + 1) / len(uploaded))
    
    status_text.empty()
    progress_bar.empty()
    st.success("Ingestion complete.")
    
    total_chunks = 0
    for name, ok, err, cnt, elapsed in statuses:
        total_chunks += (cnt or 0)
        chunk_warning = ""
        if cnt and cnt >= 9500:  # Near the 10000 limit
            chunk_warning = " ⚠️ (Large document - may be truncated)"
        st.write(f"• {name}: {'OK' if ok else 'ERROR'} — chunks={cnt or 0}{chunk_warning} — {elapsed:.1f}s {'— ' + err if err else ''}")
    
    st.info(f"💾 Total: {total_chunks} chunks indexed across {len(statuses)} document(s)")
    
    if total_chunks > 8000:
        st.warning("⚠️ Large document set detected. For best results with 1000+ page medical records:\n"
                  "- Processing may take longer during retrieval\n"
                  "- Consider processing volumes separately if performance is slow\n"
                  "- All content is indexed and searchable")

    st.markdown("---")
    st.subheader("RAG-assisted narrative & chronology")
    if st.button("Generate Narrative + Chronology (Local Llama 3.1:8B)"):
        with st.spinner("Retrieving relevant passages (enhanced retrieval for comprehensive coverage)..."):
            hits = retrieval.retrieve_for_summary(k_per_seed=25)  # Increased for better coverage
            st.info(f"Retrieved {len(hits)} unique chunks for analysis")

        with st.spinner("Generating comprehensive narrative..."):
            narrative = llm_chains.narrative_from_hits(hits)

        with st.spinner("Generating comprehensive chronology with verbatim excerpts..."):
            chron_rows = llm_chains.chronology_from_hits(hits)
            
            if not chron_rows:
                st.warning("⚠️ Chronology generation returned 0 entries. This may indicate:")
                st.write("- LLM output format issue (check terminal for raw output)")
                st.write("- Insufficient date-stamped content in retrieved chunks")
                st.write("- JSON parsing error")
                df = pd.DataFrame(columns=["datetime","summary","verbatim_text","author","document_type","page","abs_path"])
                # Show recent debug trace if available to help diagnose (written by llm_chains.chronology_from_hits)
                try:
                    with open("chronology_debug.log", "r", encoding="utf-8") as dbgf:
                        dbg_text = dbgf.read()
                        if dbg_text:
                            # Show the last chunk only to avoid overwhelming the UI
                            preview = dbg_text[-5000:]
                            with st.expander("Show raw LLM response & debug trace (chronology_debug.log)"):
                                st.code(preview, language="json")
                except FileNotFoundError:
                    pass
            else:
                df = pd.DataFrame(chron_rows)
            
            # Create clickable PDF links (file:// protocol for local PDFs)
            if not df.empty and "abs_path" in df.columns and "page" in df.columns:
                df["pdf_link"] = df.apply(
                    lambda row: f'<a href="file://{row["abs_path"]}#page={row["page"]}" target="_blank">📄 Page {row["page"]}</a>' 
                    if row["abs_path"] and row["page"] else f'Page {row["page"]}',
                    axis=1
                )

        st.header("Narrative")
        st.write(narrative or "(empty)")
        st.download_button("Download narrative.txt", data=(narrative or "").encode("utf-8"),
                           file_name="narrative.txt", mime="text/plain")

        st.header("Chronology - Medical Timeline with Source Documentation")
        if len(df) == 0:
            st.error("❌ No chronology entries generated. See warning above for potential causes.")
        else:
            st.success(f"✓ Generated {len(df)} clinically significant events")
            st.info("📋 Each entry shows: Summary narrative + Verbatim text from medical record + Provider + Source page")
        
        # Display Option B format: Summary with verbatim below
        if not df.empty:
            # CSS for Option B stacked layout
            st.markdown("""
            <style>
            .chrono-entry {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 20px;
                background-color: #ffffff;
            }
            .chrono-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
                padding-bottom: 8px;
                border-bottom: 2px solid #e0e0e0;
            }
            .chrono-datetime {
                font-weight: bold;
                font-size: 16px;
                color: #1f77b4;
            }
            .chrono-author {
                font-size: 14px;
                color: #666;
                font-style: italic;
            }
            .chrono-doc-type {
                font-size: 13px;
                background-color: #e3f2fd;
                padding: 4px 8px;
                border-radius: 4px;
                color: #1565c0;
            }
            .chrono-summary {
                margin-bottom: 12px;
                line-height: 1.6;
                color: #333;
                font-size: 15px;
            }
            .chrono-verbatim {
                background-color: #f5f5f5;
                border-left: 4px solid #2196f3;
                padding: 12px;
                margin: 12px 0;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                color: #555;
                white-space: pre-wrap;
            }
            .chrono-verbatim-label {
                font-weight: bold;
                font-size: 12px;
                color: #2196f3;
                text-transform: uppercase;
                margin-bottom: 6px;
            }
            .chrono-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 12px;
                padding-top: 8px;
                border-top: 1px solid #e0e0e0;
                font-size: 13px;
                color: #777;
            }
            .page-link {
                background-color: #4caf50;
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                text-decoration: none;
                font-weight: bold;
                font-size: 12px;
            }
            .page-link:hover {
                background-color: #45a049;
                text-decoration: none;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Generate HTML for each chronology entry
            html_output = []
            for idx, row in df.iterrows():
                page_link = f'<a class="page-link" href="file://{row["abs_path"]}#page={row["page"]}" target="_blank">📄 View Page {row["page"]}</a>' if row.get("abs_path") else f'Page {row.get("page", "?")}'
                
                entry_html = f'''
                <div class="chrono-entry">
                    <div class="chrono-header">
                        <span class="chrono-datetime">🕐 {row.get("datetime", "Unknown date/time")}</span>
                        <span class="chrono-doc-type">{row.get("document_type", "Medical Record")}</span>
                    </div>
                    
                    <div class="chrono-summary">
                        {row.get("summary", "")}
                    </div>
                    
                    <div class="chrono-verbatim">
                        <div class="chrono-verbatim-label">📋 Verbatim from Medical Record:</div>
                        {row.get("verbatim_text", "")}
                    </div>
                    
                    <div class="chrono-footer">
                        <span class="chrono-author">👤 {row.get("author", "Unknown provider")}</span>
                        <span>{page_link}</span>
                    </div>
                </div>
                '''
                html_output.append(entry_html)
            
            # Display all entries
            st.markdown("\n".join(html_output), unsafe_allow_html=True)
            
            # Download button for CSV
            csv_data = df[["datetime", "summary", "verbatim_text", "author", "document_type", "page", "doc_name", "abs_path"]].to_csv(index=False)
            st.download_button(
                "Download chronology.csv",
                data=csv_data.encode("utf-8"),
                file_name="chronology.csv",
                mime="text/csv"
            )

elif uploaded and not phi_ok:
    st.error("You must acknowledge the PHI warning to proceed before processing.")