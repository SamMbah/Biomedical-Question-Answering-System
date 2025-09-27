#!/usr/bin/env python3
# test_external_enrichment.py

import os
import pickle
import requests
from Bio import Entrez
import spacy
import pandas as pd
from dotenv import load_dotenv

# ─── 0) ENTZRE EMAIL ────────────────────────────────────────────────────────────
# Make sure NCBI knows who you are for PubMed calls:
Entrez.email = os.getenv("EMAIL_FOR_NCBI", "demo@example.com")

# ─── 1) Load environment variables from .env in the current working directory ──
load_dotenv(dotenv_path=".env")
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY", "").strip()
if not BIOPORTAL_API_KEY:
    raise RuntimeError("Please set BIOPORTAL_API_KEY in your .env")

# ─── 2) BIOPORTAL LOOKUP (returns concept ID, prefLabel, definition) ─────────────
def bioportal_lookup_term_with_id(term: str) -> tuple[str, str, str]:
    """
    Query BioPortal’s /search endpoint for 'term'.
    If a top hit is found, return (ConceptID, prefLabel, definition).
    If nothing is found, return ("", "", "").
    """
    url = "http://data.bioontology.org/search"
    params = {
        "q": term,
        "ontologies": "MESH,NCIT,HP,OMIM,MEDDRA",
        "apikey": BIOPORTAL_API_KEY,
        "require_definitions": "true",
        "pagesize": 1
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        coll = payload.get("collection", [])
        if not coll:
            return "", "", ""
        top = coll[0]

        # Extract the concept code from "@id", e.g. "…/NCIT/C27359" → "C27359"
        full_id = top.get("@id", "")
        concept_code = ""
        if full_id:
            concept_code = full_id.rstrip("/").split("/")[-1]

        # Get the preferred label
        pref = top.get("prefLabel", "").strip()

        # Get the first non-empty definition, if any
        defs = top.get("definition", [])
        definition = ""
        if isinstance(defs, list):
            for d in defs:
                if isinstance(d, str) and d.strip():
                    definition = d.strip()
                    break
        elif isinstance(defs, str) and defs.strip():
            definition = defs.strip()

        return concept_code, pref, definition

    except Exception:
        return "", "", ""


# ─── 3) PUBMED (ENTREZ) LOOKUP ───────────────────────────────────────────────────
def fetch_pubmed_abstract(query: str, max_results: int = 1) -> str:
    """
    Use Entrez to search PubMed for 'query'. Return the first abstract text,
    or an empty string if none is found.
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        id_list = record.get("IdList", [])
        if not id_list:
            return ""

        pmid = id_list[0]
        handle2 = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstract = handle2.read().strip()
        handle2.close()
        return abstract
    except Exception:
        return ""


# ─── 4) MAIN SCRIPT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # A) Load testqa.pkl and grab the first 10 questions
    with open("testqa.pkl", "rb") as f:
        df_test = pickle.load(f)
    if not isinstance(df_test, pd.DataFrame) or "QUESTION" not in df_test.columns:
        raise RuntimeError("testqa.pkl must be a pandas.DataFrame with a 'QUESTION' column.")

    all_questions = df_test["QUESTION"].tolist()
    sample_questions = all_questions[:10]  # only test the first 10

    print(f"Loaded {len(all_questions)} total test questions; testing the first {len(sample_questions)}.\n")

    # B) Load spaCy for noun-chunk fallback (if available)
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model 'en_core_web_sm' for noun-chunk fallback.\n")
    except OSError:
        try:
            nlp = spacy.load("en_core_sci_sm")
            print("Loaded spaCy model 'en_core_sci_sm' for noun-chunk fallback.\n")
        except OSError:
            print("Warning: No spaCy noun-chunk model available. BioPortal will be tried on full question only.\n")
            nlp = None

    # C) Loop through the 10 questions
    for idx, question in enumerate(sample_questions, start=1):
        print(f"--- Question {idx} ---")
        print(f"Q: {question}\n")

        # 1) Attempt BioPortal via noun-chunks if spaCy is available
        concept_code, pref_label, definition = "", "", ""
        if nlp is not None:
            doc = nlp(question)
            noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
            for chunk in noun_chunks:
                c_id, label, defs = bioportal_lookup_term_with_id(chunk)
                if label:
                    concept_code, pref_label, definition = c_id, label, defs
                    break

        # 2) If no noun-chunk hit, try the full question
        if not pref_label:
            concept_code, pref_label, definition = bioportal_lookup_term_with_id(question)

        # 3) Print BioPortal result (include C-code, label, definition)
        if concept_code:
            if definition:
                print(f"  [BioPortal] {concept_code}: {pref_label} ↪ {definition}")
            else:
                print(f"  [BioPortal] {concept_code}: {pref_label} (no definition provided)")
        else:
            print("  [BioPortal] (no concept with definition found)")

            # 4) PubMed fallback if BioPortal failed
            pm_abstract = fetch_pubmed_abstract(question)
            if pm_abstract:
                indented = "\n   ".join(pm_abstract.split("\n"))
                print(f"\n  [PubMed] Top Abstract:\n   {indented}")
            else:
                print("\n  [PubMed] (no abstract found)")

        print("\n" + "─" * 80 + "\n")

    print("Done testing BioPortal (with PubMed fallback) on the sample questions.\n")
