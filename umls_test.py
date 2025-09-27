#!/usr/bin/env python3
# test_enrichment.py

"""
Quick‐start tester for various biomedical “external knowledge” sources.
For each sample question, this script will attempt, in order:

  1) Europe PMC (paper abstract)
  2) MyGene.info (gene/protein summary)
  3) Reactome (pathway summary)
  4) DrugBank (drug mechanism/description) – requires DRUGBANK_API_KEY
  5) OpenFDA (adverse event / label) – no key needed for basic use
  6) GO via OLS (ontology definition)

You can edit the `SAMPLE_QUESTIONS` list below (currently 5 items) to test
on any questions you like.  Run:

  (medqa-gpu) $ export DRUGBANK_API_KEY="your-drugbank-key"   # if you want to test DrugBank
  (medqa-gpu) $ python test_enrichment.py

Each question will be printed along with whichever enrichment (if any) succeeded.
"""

import os
import requests
import spacy

# ───────────────────────────────────────────────────────────────────────────────
# 0) SAMPLE QUESTIONS (edit as desired)
# ───────────────────────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "Is valvular chondromodulin-1 expression downregulated in infective endocarditis?",
    "Does thrombopoietin level increase in asphyxiated neonates?",
    "How does TP53 mutation affect apoptosis in cancer cells?",
    "What is the mechanism of action of metformin in type 2 diabetes?",
    "Which pathways regulate PI3K/AKT signaling in breast cancer?",
]

# ───────────────────────────────────────────────────────────────────────────────
# 1) Europe PMC lookup: fetch a short abstract for a keyword
# ───────────────────────────────────────────────────────────────────────────────
def fetch_europe_pmc_abstract(keyword: str):
    """
    Query Europe PMC for the most recent article mentioning `keyword`.
    Returns the first ~300 characters of the abstract, or None.
    """
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": keyword,
        "format": "json",
        "pageSize": 1,      # only top hit
        "resultType": "core"
    }
    try:
        resp = requests.get(url, params=params, timeout=5.0)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    data   = resp.json()
    hits   = data.get("resultList", {}).get("result", [])
    if not hits:
        return None

    abstr = hits[0].get("abstractText", "").strip()
    if not abstr:
        return None
    return abstr[:300] + ("…" if len(abstr) > 300 else "")

# ───────────────────────────────────────────────────────────────────────────────
# 2) MyGene.info lookup: gene/protein summary
# ───────────────────────────────────────────────────────────────────────────────
def fetch_mygene_summary(gene_symbol: str):
    """
    Given a human gene symbol (e.g. "TP53" or "EGFR"), return a brief 'summary'
    text from MyGene.info, or None if not found.
    """
    base_url = "https://mygene.info/v3/query"
    params   = {"q": f"symbol:{gene_symbol}", "species": "human", "size": 1}
    try:
        r = requests.get(base_url, params=params, timeout=5.0)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    hits = r.json().get("hits", [])
    if not hits:
        return None

    entrez_id = hits[0].get("entrezgene")
    if not entrez_id:
        return None

    doc_url = f"https://mygene.info/v3/gene/{entrez_id}"
    try:
        r2 = requests.get(doc_url, params={"fields": "summary"}, timeout=5.0)
        r2.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    summary = r2.json().get("summary", "").strip()
    return summary if summary else None

# ───────────────────────────────────────────────────────────────────────────────
# 3) Reactome lookup: fetch a pathway summary
# ───────────────────────────────────────────────────────────────────────────────
def fetch_reactome_summary(pathway_name: str):
    """
    Query Reactome for the pathway whose name best matches `pathway_name`.
    Returns the 'summation' field (concise pathway description), or None.
    """
    search_url = "https://reactome.org/ContentService/search/query"
    params     = {"query": pathway_name, "species": "Homo sapiens", "pageSize": 1}
    try:
        r = requests.get(search_url, params=params, timeout=5.0)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    entries = r.json().get("entries", [])
    if not entries:
        return None

    reactome_id = entries[0].get("stId")
    if not reactome_id:
        return None

    summary_url = f"https://reactome.org/ContentService/data/pathway/{reactome_id}/summary"
    try:
        r2 = requests.get(summary_url, timeout=5.0)
        r2.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    summ = r2.json().get("summation", "").strip()
    return summ if summ else None

# ───────────────────────────────────────────────────────────────────────────────
# 4) DrugBank lookup: requires DRUGBANK_API_KEY in env
# ───────────────────────────────────────────────────────────────────────────────
def fetch_drugbank_info(drug_name: str):
    """
    Query DrugBank for `drug_name`. Returns either the 'mechanism_of_action'
    or 'description' field, or None if not found. Requires DRUGBANK_API_KEY
    exported as an environment variable.
    """
    api_key = os.getenv("DRUGBANK_API_KEY", "").strip()
    if not api_key:
        return None

    url     = "https://api.drugbank.com/v1/us/drugs"
    headers = {"Authorization": f"Bearer {api_key}"}
    params  = {"name": drug_name}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=5.0)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    data = r.json().get("data", [])
    if not data:
        return None

    entry = data[0]
    moa   = entry.get("mechanism_of_action", "").strip()
    desc  = entry.get("description", "").strip()
    return moa if moa else (desc if desc else None)

# ───────────────────────────────────────────────────────────────────────────────
# 5) OpenFDA lookup: adverse reactions or label data (no key needed)
# ───────────────────────────────────────────────────────────────────────────────
def fetch_openfda_side_effects(drug: str):
    """
    Query OpenFDA for the drug label or adverse events. Returns a short snippet,
    or None if not found. 
    """
    # First try the “drug label” endpoint (adverse_reactions)
    label_url = "https://api.fda.gov/drug/label.json"
    params1   = {"search": f"openfda.brand_name:{drug}", "limit": 1}
    try:
        r1 = requests.get(label_url, params=params1, timeout=5.0)
        r1.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    results1 = r1.json().get("results", [])
    if results1:
        reactions = results1[0].get("adverse_reactions", [])
        if isinstance(reactions, list) and reactions:
            text = " ".join(reactions)
            return text[:300] + ("…" if len(text) > 300 else "")

    # If no label-based adverse_reactions, try FAERS event endpoint
    event_url = "https://api.fda.gov/drug/event.json"
    params2   = {"search": f"patient.drug.medicinalproduct:{drug}", "limit": 1}
    try:
        r2 = requests.get(event_url, params=params2, timeout=5.0)
        r2.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    results2 = r2.json().get("results", [])
    if results2:
        reactions = results2[0].get("patient", {}).get("reaction", [])
        if isinstance(reactions, list) and reactions:
            rxn_list = [rct.get("reactionmeddrapt", "") for rct in reactions[:5]]
            text = ", ".join(rxn_list)
            return text[:300] + ("…" if len(text) > 300 else "")

    return None

# ───────────────────────────────────────────────────────────────────────────────
# 6) GO via OLS lookup: fetch a GO term definition
# ───────────────────────────────────────────────────────────────────────────────
def fetch_go_definition(go_term: str):
    """
    Use EBI's OLS to search GO for `go_term` string. Returns the first 'definition'
    annotation it finds, or None.
    """
    search_url = "https://www.ebi.ac.uk/ols/api/search"
    params     = {"q": go_term, "ontology": "go", "type": "class", "rows": 1}
    try:
        r = requests.get(search_url, params=params, timeout=5.0)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    docs = r.json().get("response", {}).get("docs", [])
    if not docs:
        return None

    iri = docs[0].get("iri")
    if not iri:
        return None

    # Percent‐encode the IRI for the terms endpoint
    from requests.utils import quote
    iri_enc = quote(iri, safe="")
    term_url = f"https://www.ebi.ac.uk/ols/api/ontologies/go/terms/{iri_enc}"
    try:
        r2 = requests.get(term_url, timeout=5.0)
        r2.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    embedded = r2.json().get("_embedded", {}).get("terms", [])
    if not embedded:
        return None

    annotations = embedded[0].get("annotation", [])
    for ann in annotations:
        if ann.get("property") == "definition":
            return ann.get("value", "").strip()

    return None

# ───────────────────────────────────────────────────────────────────────────────
# 7) MAIN TEST LOOP
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Testing Enrichment for Sample Questions ===\n")

    # Load spaCy model once (for any noun-chunk heuristics if you want)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    for idx, q in enumerate(SAMPLE_QUESTIONS):
        print(f"--- Question {idx+1} ---")
        print("Q:", q)

        # 1) Europe PMC
        epmc_text = fetch_europe_pmc_abstract(q)
        if epmc_text:
            print("  Europe PMC →", epmc_text)
        else:
            print("  Europe PMC → No hit")

        # 2) MyGene.info – try to detect a gene symbol (all‐caps or contains digits)
        gene_found = None
        for chunk in nlp(q).noun_chunks:
            chunk_txt = chunk.text.strip()
            if chunk_txt.isupper() or any(c.isdigit() for c in chunk_txt):
                gene_found = chunk_txt
                break
        if gene_found:
            mg_text = fetch_mygene_summary(gene_found)
            if mg_text:
                print(f"  MyGene ({gene_found}) →", mg_text)
            else:
                print(f"  MyGene ({gene_found}) → No hit")
        else:
            print("  MyGene → No obvious gene symbol in question")

        # 3) Reactome – look for a known pathway keyword in Q
        reactome_keywords = ["PI3K/AKT", "MAPK", "mTOR", "TNF", "WNT"]
        reactome_hit = next((kw for kw in reactome_keywords if kw.upper() in q.upper()), None)
        if reactome_hit:
            rw_text = fetch_reactome_summary(reactome_hit)
            if rw_text:
                print(f"  Reactome ({reactome_hit}) →", rw_text)
            else:
                print(f"  Reactome ({reactome_hit}) → No hit")
        else:
            print("  Reactome → No pathway keyword detected")

        # 4) DrugBank – look for a known drug name in Q
        drugbank_drugs = ["METFORMIN", "IBUPROFEN", "ASPIRIN", "RITUXIMAB"]
        drug_hit = next((d for d in drugbank_drugs if d in q.upper()), None)
        if drug_hit:
            db_text = fetch_drugbank_info(drug_hit.lower())
            if db_text:
                print(f"  DrugBank ({drug_hit}) →", db_text)
            else:
                print(f"  DrugBank ({drug_hit}) → No hit or missing API key")
        else:
            print("  DrugBank → No drug keyword detected")

        # 5) OpenFDA – same drug list as above
        if drug_hit:
            ofd_text = fetch_openfda_side_effects(drug_hit.lower())
            if ofd_text:
                print(f"  OpenFDA ({drug_hit}) →", ofd_text)
            else:
                print(f"  OpenFDA ({drug_hit}) → No hit")
        else:
            print("  OpenFDA → No drug keyword detected")

        # 6) GO via OLS – look for a GO‐term phrase in Q
        go_terms = ["APOPTOTIC PROCESS", "LIPID METABOLIC PROCESS", "CELL PROLIFERATION"]
        go_hit   = next((g for g in go_terms if g in q.upper()), None)
        if go_hit:
            go_text = fetch_go_definition(go_hit)
            if go_text:
                print(f"  GO ({go_hit}) →", go_text)
            else:
                print(f"  GO ({go_hit}) → No hit")
        else:
            print("  GO → No GO‐term phrase detected")

        print()  # blank line between questions

    print("=== Done testing enrichment. ===\n")
