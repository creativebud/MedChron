# retrieval.py
from typing import List, Dict
from rag_store import search

DEFAULT_SEEDS = [
    "patient name demographics date of birth DOB age sex gender",
    "CODE BLUE cardiac arrest resuscitation CPR ACLS rapid response emergency",
    "ED physician documentation emergency department physician note assessment",
    "triage OR presenting complaint OR HPI chief complaint",
    "highest acuity event OR ICU transfer OR code OR critical condition",
    "resuscitation efforts intubation defibrillation shock compressions epinephrine",
    "diagnostics labs imaging CT MRI xray ultrasound",
    "interventions medications procedures dose drip infusion",
    "clinical course improvement deterioration change in condition",
    "disposition discharge transfer outcome death expired",
    "allergy list OR allergies adverse reactions",
    "medication list OR home meds admission medications",
    "ED triage note OR emergency department triage",
    "History & Physical OR H&P admission note",
    "Nurses notes shift reassessment nursing documentation",
    "Labs: CBC CMP lactate troponin creatinine sodium potassium glucose",
    "Imaging report chest x-ray CT head CT abdomen MRI ultrasound findings",
    "Medications administered dose route drip infusion titration bolus",
    "Consultation report specialty consult cardiology neurology",
    "Discharge summary instructions follow-up",
    "date time timestamp when occurred documented", # Help retrieve date-stamped content
    "signed by dictated by authored by provider name physician nurse", # Help get author info
]

def unique_hits(hits: List[Dict], max_len: int = 600) -> List[Dict]:  # Increased from 500 to 600
    seen = set()
    out = []
    for h in hits:
        key = (h["text"][:200], (h.get("metadata") or {}).get("page_label"))
        if key in seen: 
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= max_len:
            break
    return out

def retrieve_for_summary(k_per_seed: int = 25) -> List[Dict]:  # Increased from 20 to 25
    """Retrieve chunks for summary generation with enhanced coverage."""
    all_hits: List[Dict] = []
    for q in DEFAULT_SEEDS:
        all_hits += search(q, k=k_per_seed)
    return unique_hits(all_hits, max_len=600)  # Increased from 500 to 600
    return unique_hits(all_hits, max_len=500)
