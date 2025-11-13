# llm_chains.py
from typing import List, Dict
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import datetime
import json as _json

# Force the local 3.1 8B model
OLLAMA_MODEL = "llama3.1:8b"

_LLM_CACHE = {}

def _llm(temp: float = 0.0, max_tokens: int = 4096, json_mode: bool = False):
    """Increased max_tokens from 1800 to 4096 to handle large chronologies
    json_mode: When True, forces Ollama to output valid JSON

    Small cache is used to avoid re-instantiating the client repeatedly.
    Cache key is (model, temp, num_predict, json_mode)
    """
    key = (OLLAMA_MODEL, float(temp), int(max_tokens), bool(json_mode))
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]

    kwargs = {
        "model": OLLAMA_MODEL,
        "temperature": temp,
        "num_ctx": 8192,
        "num_predict": max_tokens
    }
    if json_mode:
        kwargs["format"] = "json"
    client = ChatOllama(**kwargs)
    _LLM_CACHE[key] = client
    return client

def narrative_from_hits(hits: List[Dict]) -> str:
    """Hits: [{text, metadata:{page_label, section, doc_name}}]"""
    evidence = []
    for h in hits:
        m = h.get("metadata") or {}
        sec = (m.get("section") or "Document").strip()
        pg  = (m.get("page_label") or "").strip()
        doc = (m.get("doc_name") or "").strip()
        # Normalize possible text keys coming from different retrieval functions
        text = h.get("text") or h.get("page_content") or h.get("content") or ""
        evidence.append(f"[{doc} p.{pg}] {sec} — {text[:1200]}")
    ev = "\n".join(evidence[:300])  # Increased from 120 to 300

    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert medico-legal medical record analyst creating a comprehensive NARRATIVE SUMMARY.\n\n"
         "YOUR GOAL: Write a cohesive, flowing narrative that tells the patient's complete medical story.\n"
         "This is NOT a chronology - it's a descriptive analysis that synthesizes the medical course.\n\n"
         "Include concrete, verifiable facts and data points from the evidence to support the narrative.\n"
         "Specifically, where present in the evidence, include: dates and times of key events, exact lab values\n"
         "(with units), vital signs (with times), medication names and doses, procedures (with times), imaging\n"
         "findings, culture results, provider names/roles, page/document references, lengths of stay, and final\n"
         "disposition. When summarizing clinical reasoning, cite short, exact verbatim phrases from the evidence\n"
         "in quotation marks to support important assertions (use verbatim sparingly and only as direct evidence).\n\n"
         "When concrete data is missing, say explicitly 'not documented' rather than guessing. Keep the narrative\n"
         "cohesive and flowing, but prioritize factual specificity and measurable findings over vague language.\n\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "CRITICAL REQUIREMENTS:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "✓ Use the ACTUAL PATIENT NAME from the records (never generic names)\n"
         "✓ Write in flowing narrative paragraphs (NOT bullet points or lists)\n"
         "✓ Synthesize information thematically, not just chronologically\n"
         "✓ Include dates/times for KEY events only (not every single action)\n"
         "✓ Focus on the OVERALL PICTURE: patterns, trends, clinical reasoning\n"
         "✓ Be DESCRIPTIVE and ANALYTICAL, not just a list of events\n\n"
         
         "═══════════════════════════════════════════════════════════════════════\n"
         "NARRATIVE STRUCTURE (seamless flow between sections):\n"
         "═══════════════════════════════════════════════════════════════════════\n\n"
         
         "OPENING - Patient Profile & Context:\n"
         "Start with WHO the patient is: name, age, relevant medical history, baseline health status.\n"
         "Then describe WHAT BROUGHT THEM TO CARE: presenting symptoms, circumstances of presentation.\n"
         "Example: 'Mr. John Smith, a 67-year-old male with a history of hypertension and diabetes,\n"
         "presented to Chino Valley Medical Center on February 13, 2023, with acute onset of severe\n"
         "chest pain that had been worsening over the previous two hours.'\n\n"
         
         "MIDDLE - Clinical Course & Medical Management:\n"
         "Describe the OVERALL TRAJECTORY of the hospitalization in narrative form:\n"
         "• What was the working diagnosis? How did it evolve?\n"
         "• What treatments were attempted? What was the clinical reasoning?\n"
         "• How did the patient respond? Were there complications?\n"
         "• What consultations were obtained? What were their recommendations?\n"
         "• Synthesize lab results, imaging findings, and clinical observations into a coherent picture\n\n"
         
         "Example: 'The initial evaluation revealed significant cardiac abnormalities, with EKG findings\n"
         "consistent with an acute inferior myocardial infarction. Troponin levels were markedly elevated\n"
         "at 15.2 ng/mL (normal <0.04), and the patient was experiencing ongoing chest pain despite\n"
         "initial interventions. The cardiology team was consulted emergently and recommended immediate\n"
         "cardiac catheterization. Throughout the first 24 hours, the patient remained hemodynamically\n"
         "unstable, requiring escalating doses of vasopressor support...'\n\n"
         
         "CLOSING - Outcomes & Critical Analysis:\n"
         "Summarize the OVERALL OUTCOME and provide ANALYTICAL INSIGHTS:\n"
         "• What was the final disposition? (discharge, transfer, death)\n"
         "• What was the patient's condition at discharge/outcome?\n"
         "• Were there any complications or unexpected events?\n"
         "• CRITICAL ANALYSIS: Note any deviations from standard care, delays in treatment,\n"
         "  opportunities for different management, or areas of concern\n"
         "• What alternative approaches could have been considered?\n\n"
         
         "═══════════════════════════════════════════════════════════════════════\n"
         "WRITING STYLE:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "✓ DESCRIPTIVE: Paint a picture of what happened and why\n"
         "✓ ANALYTICAL: Explain clinical reasoning, patterns, and decision-making\n"
         "✓ COHESIVE: Each paragraph flows naturally into the next\n"
         "✓ PRECISE: Include exact values, doses, measurements when relevant to the narrative\n"
         "✓ CONTEXTUAL: Explain the significance of findings (not just list them)\n"
         "✓ PROFESSIONAL: Medical terminology used appropriately\n"
         "✓ CITED: Include (p.X) references inline where appropriate\n\n"
         
         "What to EMPHASIZE:\n"
         "• Key diagnoses and their clinical significance\n"
         "• Major interventions and their rationale\n"
         "• Critical events and how they were managed\n"
         "• Trends in clinical status (improving, declining, stable)\n"
         "• Complications and their impact\n"
         "• Clinical decision-making and reasoning\n\n"
         
         "What to DE-EMPHASIZE or omit:\n"
         "• Routine vital signs if stable and normal\n"
         "• Standard admission procedures unless clinically significant\n"
         "• Repetitive information that doesn't add to the story\n"
         "• Minor medications or interventions that didn't impact the course\n\n"
         
         "═══════════════════════════════════════════════════════════════════════\n"
         "EXAMPLE NARRATIVE EXCERPT:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "'Sarah Johnson, a 58-year-old female with a complex medical history including poorly\n"
         "controlled type 2 diabetes and chronic kidney disease, was admitted to the intensive\n"
         "care unit on March 10, 2023, following a catastrophic deterioration in her clinical\n"
         "status. The patient had initially presented to the emergency department three days\n"
         "earlier with symptoms of severe abdominal pain and vomiting, which were initially\n"
         "attributed to acute gastroenteritis.\n\n"
         
         "However, the clinical picture evolved rapidly to reveal a much more serious underlying\n"
         "process. Serial laboratory studies demonstrated progressively worsening renal function,\n"
         "with creatinine rising from an baseline of 2.1 mg/dL to 4.8 mg/dL over 48 hours,\n"
         "accompanied by severe metabolic acidosis. Imaging studies revealed bilateral pleural\n"
         "effusions and pulmonary edema, suggesting volume overload in the setting of acute-on-chronic\n"
         "kidney injury. The nephrology service was consulted and recommended urgent hemodialysis.\n\n"
         
         "The patient's hospital course was further complicated by an unexpected cardiac arrest on\n"
         "the morning of March 10th. The code blue team responded immediately, and after 15 minutes\n"
         "of advanced cardiac life support including multiple rounds of epinephrine and two\n"
         "defibrillation attempts, return of spontaneous circulation was achieved. Post-resuscitation\n"
         "care included mechanical ventilation and vasopressor support with norepinephrine...\n\n"
         
         "From a critical analysis perspective, several questions arise regarding the management\n"
         "of this case. The initial presentation was treated as gastroenteritis, but given the\n"
         "patient's known chronic kidney disease, earlier recognition of acute kidney injury might\n"
         "have prompted more aggressive fluid management and earlier nephrology consultation...'\n\n"
         
         "═══════════════════════════════════════════════════════════════════════\n"
         "Remember: This is a NARRATIVE SUMMARY, not a chronology. Tell the story.\n"
         "═══════════════════════════════════════════════════════════════════════"),
        ("human",
         "EVIDENCE FROM MEDICAL RECORDS:\n{evidence}\n\n"
         "Write a comprehensive narrative summary that tells this patient's complete medical story.\n"
         "Focus on the overall picture, clinical reasoning, and key events - NOT a chronological list.")
    ])
    chain = template | _llm(temp=0.0, max_tokens=6000)  # Large token budget for comprehensive narrative
    return chain.invoke({"evidence": ev}).content.strip()

def chronology_from_hits(hits: List[Dict]) -> List[Dict]:
    """
    Create chronological timeline with verbatim + summary (Option B).
    Returns: [{datetime, summary, verbatim_text, author, document_type, page, abs_path}]
    """
    # Build metadata lookup by doc+page
    metadata_lookup = {}
    for h in hits:
        m = h.get("metadata") or {}
        doc = (m.get("doc_name") or "").strip()
        pg = (m.get("page_label") or "").strip()
        key = f"{doc}|{pg}"
        if key not in metadata_lookup:
            metadata_lookup[key] = {
                "abs_path": m.get("abs_path", ""),
                "section": (m.get("section") or "Document").strip(),
                "author": m.get("author", "")
            }
    
    # Build evidence with increased context
    evidence = []
    for h in hits:
        m = h.get("metadata") or {}
        sec = (m.get("section") or "Document").strip()
        pg  = (m.get("page_label") or "").strip()
        doc = (m.get("doc_name") or "").strip()
        author = (m.get("author") or "").strip()
        
    # Include author in evidence if available
    author_str = f" [Author: {author}]" if author else ""
    # Normalize text key (some upstream code uses 'page_content' or 'content')
    text = h.get("text") or h.get("page_content") or h.get("content") or ""
    evidence.append(f"[{doc} p.{pg}]{author_str} {sec} — {text[:1500]}")  # Increased to 1500
    
    ev = "\n".join(evidence[:450])  # Increased to 450 for more context

    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a medical records analyst extracting chronological events from medical documentation.\n\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "CRITICAL RULES - FOLLOW EXACTLY:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "1. Extract ONLY events explicitly documented in the evidence below\n"
         "2. NEVER invent, assume, or infer information not present in evidence\n"
         "3. For verbatim_text: Copy EXACT phrases from evidence (do not paraphrase)\n"
         "4. For datetime: Use ONLY dates/times explicitly stated in evidence\n"
         "5. If author name not in evidence, use provider role from section name\n"
         "6. Sort output chronologically by date/time\n\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "OUTPUT FORMAT (JSON Array):\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "Each event must have these EXACT field names:\n"
         "- datetime: ISO format or descriptive (e.g., '2023-02-15 14:30' or 'February 15, 2023 at 2:30 PM')\n"
         "- summary: Brief description of what happened (YOUR summary)\n"
         "- verbatim_text: EXACT quote from medical record (must be word-for-word)\n"
         "- author: Provider name or role (from evidence)\n"
         "- document_type: Type of document (e.g., 'Progress Note', 'Lab Report')\n"
         "- page: Page number from evidence\n"
         "- doc_name: Document filename\n\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "EXAMPLES OF GOOD EVENTS:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "{{\n"
         '  "datetime": "2023-02-15 14:30",\n'
         '  "summary": "Patient admitted with severe chest pain and elevated troponin",\n'
         '  "verbatim_text": "Patient presents with crushing substernal chest pain radiating to left arm. Troponin I: 15.2 ng/mL (normal <0.04). EKG shows ST elevation in inferior leads.",\n'
         '  "author": "Dr. Sarah Johnson, Cardiologist",\n'
         '  "document_type": "Emergency Department Note",\n'
         '  "page": "3",\n'
         '  "doc_name": "ED_Visit_02-15-2023.pdf"\n'
         "}}\n\n"
         "{{\n"
         '  "datetime": "2023-02-16 08:00",\n'
         '  "summary": "Cardiac catheterization revealed 95% occlusion of RCA",\n'
         '  "verbatim_text": "Cardiac cath performed at 0800. Right coronary artery shows 95% stenosis at proximal segment. Successfully placed drug-eluting stent with TIMI 3 flow restored.",\n'
         '  "author": "Dr. Michael Chen, Interventional Cardiology",\n'
         '  "document_type": "Procedure Note",\n'
         '  "page": "7",\n'
         '  "doc_name": "Cardiac_Cath_Report.pdf"\n'
         "}}\n\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "VERBATIM TEXT REQUIREMENTS:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "✓ Copy EXACT words from evidence - character for character\n"
         "✓ Include relevant context (not just isolated phrases)\n"
         "✓ Can be 1-3 sentences capturing the key information\n"
         "✓ Must support the summary you wrote\n"
         "✓ Do NOT paraphrase, summarize, or modify the verbatim text\n"
         "✓ Do NOT add interpretations or explanations\n\n"
         "BAD verbatim (paraphrased): 'The patient had high blood pressure'\n"
         "GOOD verbatim (exact quote): 'Blood pressure measured at 180/110 mmHg at 1400 hours'\n\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "WHAT TO EXTRACT:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "• Admission/discharge dates and times\n"
         "• Diagnoses made or ruled out\n"
         "• Procedures performed (with times)\n"
         "• Significant lab results or vital signs\n"
         "• Medications started/changed/stopped\n"
         "• Consultations requested or completed\n"
         "• Clinical deteriorations or complications\n"
         "• Treatment decisions and their rationale\n"
         "• Patient transfers between units\n"
         "• Code events or emergencies\n\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "WHAT NOT TO EXTRACT:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "✗ Routine vital signs if stable/normal\n"
         "✗ Repetitive information already captured\n"
         "✗ Administrative details (insurance, billing codes)\n"
         "✗ Anything not explicitly stated in the evidence\n\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "FIELD NAME REQUIREMENTS:\n"
         "═══════════════════════════════════════════════════════════════════════\n"
         "Use these EXACT field names (case-sensitive):\n"
         "- datetime\n"
         "- summary\n"
         "- verbatim_text\n"
         "- author\n"
         "- document_type\n"
         "- page\n"
         "- doc_name\n\n"
         "Do NOT use any other field names. Do NOT abbreviate. Follow the examples EXACTLY.\n"
         "Output format: [{{\"datetime\": \"...\", \"summary\": \"...\", \"verbatim_text\": \"...\", ...}}]\n\n"
         "Output ONLY the JSON array, nothing else."),
        ("human",
         "EVIDENCE FROM MEDICAL RECORDS:\n{evidence}\n\n"
         "Extract chronological events following ALL rules above.\n"
         "Remember: EXACT verbatim quotes, ONLY information from evidence, NO inventions.")
    ])
    
    chain = template | _llm(temp=0.0, max_tokens=10000, json_mode=True)  # JSON mode forces valid JSON output when possible
    response = chain.invoke({"evidence": ev}).content.strip()

    # Debug: Print raw LLM output
    print("=" * 80)
    print("RAW CHRONOLOGY OUTPUT FROM LLM:")
    print(response[:1500])  # Print first 1500 chars
    print("=" * 80)

    # Write debug trace to chronology_debug.log for UI/diagnostics
    try:
        dbg = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "evidence_snippet": ev[:4000],
            "raw_response_preview": response[:20000]
        }
        with open("chronology_debug.log", "a", encoding="utf-8") as df:
            df.write("\n" + "="*120 + "\n")
            df.write(f"TIMESTAMP: {dbg['ts']}\n")
            df.write("--- EVIDENCE (truncated) ---\n")
            df.write(dbg['evidence_snippet'] + "\n")
            df.write("--- RAW RESPONSE (truncated) ---\n")
            df.write(dbg['raw_response_preview'] + "\n")
            df.write("\n")
    except Exception:
        # Never fail on debug logging
        pass

    # Parsing & validation helpers
    import json
    import re

    def extract_between_markers(text: str):
        """Extract JSON-like substring between known markers or code fences."""
        # Preferred explicit markers
        m = re.search(r'<!--JSON_START-->(.*?)<!--JSON_END-->', text, flags=re.S)
        if m:
            return m.group(1).strip()
        # Fenced json block
        m2 = re.search(r'```json\s*(.*?)```', text, flags=re.S | re.I)
        if m2:
            return m2.group(1).strip()
        # First JSON array
        m3 = re.search(r'\[\s*\{.*?\}\s*\]', text, flags=re.S)
        if m3:
            return m3.group(0)
        # First JSON object
        m4 = re.search(r'\{.*\}', text, flags=re.S)
        if m4:
            return m4.group(0)
        return None

    def parse_json_tolerant(s: str):
        """Try strict json, then json5, demjson3, then simple repairs."""
        if s is None:
            return None
        s = s.strip()
        try:
            parsed = json.loads(s)
            return parsed
        except Exception:
            pass
        # try json5
        try:
            import json5
            return json5.loads(s)
        except Exception:
            pass
        # try demjson3
        try:
            import demjson3
            return demjson3.decode(s)
        except Exception:
            pass
        # quick heuristic repair: remove trailing commas before ] or }
        try:
            s2 = re.sub(r',\s*([\]\}])', r'\1', s)
            return json.loads(s2)
        except Exception:
            return None

    def repair_with_llm(raw_text: str):
        """Ask the model to reformat its previous output into strict JSON array between markers."""
        # Build a minimal repair prompt using ChatPromptTemplate (will be monkeypatched in tests)
        repair_template = ChatPromptTemplate.from_messages([
            ("system", "You will be given a previous model output that may contain malformed JSON or extra prose. Return ONLY a valid JSON ARRAY of chronology event objects between the markers <!--JSON_START--> and <!--JSON_END-->. Do NOT include explanation."),
            ("human", "PREVIOUS OUTPUT:\n{raw}\n\nReturn only valid JSON between <!--JSON_START--> and <!--JSON_END-->. If no events, return [].")
        ])
        chain_repair = repair_template | _llm(temp=0.0, max_tokens=1500, json_mode=False)
        repaired = chain_repair.invoke({"raw": raw_text}).content.strip()
        return repaired

    def validate_chronology_entry(entry, evidence_text):
        """Validate entry for accuracy - returns list of warnings"""
        warnings = []
        # 1. Check datetime format (relaxing requirement by accepting many formats)
        datetime_str = entry.get('datetime', '')
        if not entry.get('datetime_estimated'):
            # Accept if there is any digit-like date or ISO-like format
            if not re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}', datetime_str):
                warnings.append(f"Invalid datetime format: {datetime_str}")

        # 2. Verify verbatim text appears in evidence (fuzzy match)
        verbatim = entry.get('verbatim_text', '')
        if len(verbatim) > 30:
            verbatim_normalized = re.sub(r'\s+', ' ', verbatim.lower().strip())
            evidence_normalized = re.sub(r'\s+', ' ', evidence_text.lower())
            words = verbatim_normalized.split()
            if len(words) >= 5:
                chunks = [' '.join(words[i:i+5]) for i in range(0, min(len(words)-4, 10))]
                matches = sum(1 for chunk in chunks if chunk in evidence_normalized)
                match_ratio = matches / len(chunks) if chunks else 0
                if match_ratio < 0.3:
                    warnings.append(f"⚠️  Verbatim quote may not be accurate: '{verbatim[:60]}...'")

        # 3. Check for DOB (should be filtered)
        summary_lower = entry.get('summary', '').lower()
        if 'date of birth' in summary_lower or 'dob' in summary_lower or 'born on' in summary_lower:
            warnings.append("Contains DOB - should be filtered")

        # 4. Check author format
        author = entry.get('author', '')
        if author and author not in ['Not documented', '']:
            has_name = bool(re.search(r'[A-Z][a-z]+', author))
            has_credential = bool(re.search(r'MD|DO|RN|NP|PA|Physician|Nurse|Team|Staff', author, re.I))
            if not (has_name or has_credential):
                warnings.append(f"Author format suspicious: '{author}'")

        # 5. Summary style (soft warning)
        summary = entry.get('summary', '')
        if summary and not re.match(r'^On\s+\w+\s+\d+,?\s+\d{4}', summary):
            warnings.append("Summary should start with 'On [date]...' (soft)")

        return warnings

    rows = []

    # Parsing pipeline: try markers -> strict parse -> tolerant parse -> repair LLM -> heuristic
    parsed_events = None

    candidate = extract_between_markers(response)
    if candidate:
        parsed = parse_json_tolerant(candidate)
        if parsed is not None:
            if isinstance(parsed, list):
                parsed_events = parsed
            elif isinstance(parsed, dict):
                parsed_events = [parsed]

    # If no candidate or parsing failed, try parsing the whole response strictly
    if parsed_events is None:
        try:
            parsed_whole = json.loads(response)
            if isinstance(parsed_whole, list):
                parsed_events = parsed_whole
            elif isinstance(parsed_whole, dict):
                parsed_events = [parsed_whole]
        except Exception:
            parsed_events = None

    # If still missing, attempt tolerant parse on the full response
    if parsed_events is None:
        parsed_tolerant = parse_json_tolerant(response)
        if parsed_tolerant is not None:
            if isinstance(parsed_tolerant, list):
                parsed_events = parsed_tolerant
            elif isinstance(parsed_tolerant, dict):
                parsed_events = [parsed_tolerant]

    # If still missing, try repair via LLM and retry parsing
    if parsed_events is None:
        repaired = repair_with_llm(response)
        # write repaired to debug log as well
        try:
            with open("chronology_debug.log", "a", encoding="utf-8") as df:
                df.write("--- REPAIRED OUTPUT (truncated) ---\n")
                df.write(repaired[:20000] + "\n")
        except Exception:
            pass
        candidate2 = extract_between_markers(repaired) or repaired
        parsed_rep = parse_json_tolerant(candidate2)
        if parsed_rep is not None:
            if isinstance(parsed_rep, list):
                parsed_events = parsed_rep
            elif isinstance(parsed_rep, dict):
                parsed_events = [parsed_rep]

    if not parsed_events:
        print("❌ ERROR: Could not find or parse JSON in response")
    else:
        # Process parsed_events (should be a list)
        for event in parsed_events:
            if isinstance(event, dict):
                # Filter out DOB entries
                summary_lower = event.get('summary', '').lower()
                if 'date of birth' in summary_lower or 'dob' in summary_lower or 'born' in summary_lower:
                    print(f"⏭️  Skipping DOB entry: {event.get('datetime', 'unknown date')}")
                    continue

                # Ensure every event has a usable datetime. If LLM didn't provide one,
                # attempt to estimate a nearby/related date from the evidence or hits.
                dt_raw = (event.get('datetime') or '').strip()
                date_rx = re.compile(r"\d{1,2}[\\/\-]\d{1,2}[\\/\-]\d{2,4}|\d{4}-\d{2}-\d{2}")
                estimated = False
                dt_value = dt_raw

                if not dt_raw or not date_rx.search(dt_raw):
                    # 1) Search the evidence blob (ev) for any date-like substring
                    m = date_rx.search(ev)
                    if m:
                        dt_value = m.group(0)
                        estimated = True
                    else:
                        # 2) Try to find a date in the originating hit (same doc/page) if available
                        doc_name = event.get('doc_name', '')
                        page_num = event.get('page', '')
                        found = None
                        for h_scan in hits:
                            m_h = h_scan.get('metadata') or {}
                            if str(m_h.get('doc_name', '')) == str(doc_name) and (str(m_h.get('page_label') or m_h.get('page') or '') == str(page_num)):
                                text_scan = h_scan.get('text') or h_scan.get('page_content') or h_scan.get('content') or ''
                                m2 = date_rx.search(text_scan)
                                if m2:
                                    found = m2.group(0)
                                    break
                        if found:
                            dt_value = found
                            estimated = True
                        else:
                            # 3) Fallback: search across all hits for the closest available date
                            for h_scan in hits:
                                text_scan = h_scan.get('text') or h_scan.get('page_content') or h_scan.get('content') or ''
                                m3 = date_rx.search(text_scan)
                                if m3:
                                    found = m3.group(0)
                                    estimated = True
                                    dt_value = found
                                    break
                        # 4) If still not found, mark as Unknown and estimated
                        if not dt_value:
                            dt_value = dt_raw or "Unknown"
                            estimated = True

                # Normalize final datetime string and mark estimated with an asterisk
                if estimated and dt_value and not dt_value.endswith("*"):
                    dt_presentable = f"{dt_value}*"
                else:
                    dt_presentable = dt_value

                # Look up abs_path from metadata
                doc_name = event.get('doc_name', '')
                page_num = event.get('page', '')
                lookup_key = f"{doc_name}|{page_num}"
                meta_info = metadata_lookup.get(lookup_key, {})

                entry_data = {
                    "datetime": dt_presentable,
                    "datetime_estimated": bool(estimated),
                    "summary": event.get('summary', ''),
                    "verbatim_text": event.get('verbatim_text', ''),
                    "author": event.get('author', 'Not documented'),
                    "document_type": event.get('document_type', ''),
                    "page": page_num,
                    "doc_name": doc_name,
                    "abs_path": meta_info.get('abs_path', ''),
                }

                # Validate entry
                validation_warnings = validate_chronology_entry(entry_data, ev)
                if validation_warnings:
                    print(f"⚠️  Validation warnings for {entry_data['datetime']}:")
                    for w in validation_warnings:
                        print(f"   • {w}")

                rows.append(entry_data)

        print(f"✓ Successfully parsed {len(rows)} events from JSON")

    # Fallback: try to parse line by line only if we found nothing above
    if not rows:
        lines = response.splitlines()
        for line in lines:
            if '"datetime"' in line and '"summary"' in line:
                try:
                    # Try to parse individual event
                    event = json.loads(line.strip().rstrip(','))
                    
                    doc_name = event.get('doc_name', '')
                    page_num = event.get('page', '')
                    lookup_key = f"{doc_name}|{page_num}"
                    meta_info = metadata_lookup.get(lookup_key, {})
                    
                    entry_data = {
                        "datetime": event.get('datetime', ''),
                        "summary": event.get('summary', ''),
                        "verbatim_text": event.get('verbatim_text', ''),
                        "author": event.get('author', 'Not documented'),
                        "document_type": event.get('document_type', ''),
                        "page": page_num,
                        "doc_name": doc_name,
                        "abs_path": meta_info.get('abs_path', ''),
                    }
                    rows.append(entry_data)
                except Exception:
                    continue

    # Deduplicate rows by key (datetime, summary, doc_name, page, verbatim snippet)
    seen = set()
    deduped = []
    for r in rows:
        key = (r.get('datetime'), (r.get('summary') or '')[:120], r.get('doc_name'), r.get('page'), (r.get('verbatim_text') or '')[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    rows = deduped
    
    # Final output summary
    print(f"=" * 80)
    print(f"📊 FINAL CHRONOLOGY SUMMARY")
    print(f"=" * 80)
    print(f"Total events extracted: {len(rows)}")
    if rows:
        print(f"\nFirst event:")
        print(f"  {rows[0]['datetime']} - {rows[0]['summary'][:100]}...")
        print(f"  Author: {rows[0].get('author', 'Unknown')}")
        print(f"\nLast event:")
        print(f"  {rows[-1]['datetime']} - {rows[-1]['summary'][:100]}...")
        print(f"  Author: {rows[-1].get('author', 'Unknown')}")
    print(f"=" * 80)
    
    return rows
