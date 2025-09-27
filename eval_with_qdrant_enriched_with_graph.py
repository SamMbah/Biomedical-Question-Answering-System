#!/usr/bin/env python3
"""
eval_with_qdrant_enriched_with_graph.py

Runs retrieval + GPT-3.5 + ROUGE on a 50-question sample from MedQA’s test set,
always including BioPortal or PubMed external knowledge, and builds a small
knowledge graph that records which source(s) contributed. Uses Qdrant for
nearest-neighbor retrieval of contexts. Implements retry logic for both
OpenAI calls and Qdrant queries to handle intermittent timeouts.

Requirements:
    - Python 3.8+
    - conda environment with: openai, qdrant-client, numpy, pandas, torch,
      sentence_transformers, rouge-score, requests, biopython (Entrez), spaCy
      (en_core_web_sm), networkx

Environment Variables (export before running):
    export OPENAI_API_KEY="..."
    export BIOPORTAL_API_KEY="..."
    export EMAIL_FOR_NCBI="your_email@example.com"

Usage:
    python eval_with_qdrant_enriched_with_graph.py
"""
import os
import time
import random
import pickle
import requests
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sentence_transformers import util
import openai
from openai import error as openai_error
from rouge_score import rouge_scorer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from Bio import Entrez
import spacy

# ─── A) CONFIG ──────────────────────────────────────────────────────────────────
SEED        = 42
SAMPLE_SIZE = 50
TOP_K       = 3
TEMP        = 0.0
MAX_TOKENS  = 256

# Model names (singles + all pairwise combos, 38 total)
single_models = [
    "INSTR", "JINA", "E5", "SPECTER",
    "MPNET", "MINILM6", "MINILM12", "LABSE",
    "WORD2VEC", "FASTTEXT"
]

pairwise_combos = {
    "INSTR_E5":       [("INSTR", 0.7),   ("E5",     0.3)],
    "INSTR_SPECTER":  [("INSTR", 0.7),   ("SPECTER",0.3)],
    "INSTR_JINA":     [("INSTR", 0.7),   ("JINA",   0.3)],
    "INSTR_MPNET":    [("INSTR", 0.7),   ("MPNET",  0.3)],
    "INSTR_LABSE":    [("INSTR", 0.7),   ("LABSE",  0.3)],

    "E5_SPECTER":     [("E5",    0.7),   ("SPECTER",0.3)],
    "JINA_E5":        [("JINA",  0.7),   ("E5",     0.3)],
    "E5_MPNET":       [("E5",    0.7),   ("MPNET",  0.3)],
    "JINA_SPECTER":   [("JINA",  0.3),   ("SPECTER",0.7)],
    "SPECTER_MPNET":  [("SPECTER",0.7),  ("MPNET",  0.3)],

    "JINA_MPNET":     [("JINA",   0.7),  ("MPNET",  0.3)],
    "E5_LABSE":       [("E5",     0.7),  ("LABSE",  0.3)],
    "SPECTER_LABSE":  [("SPECTER",0.7),  ("LABSE",  0.3)],
    "JINA_LABSE":     [("JINA",   0.7),  ("LABSE",  0.3)],
    "MPNET_LABSE":    [("MPNET",  0.7),  ("LABSE",  0.3)],

    "MINILM6_MINILM12":[("MINILM6",0.5), ("MINILM12",0.5)],

    "W2V_INSTR":      [("WORD2VEC",0.3), ("INSTR",  0.7)],
    "FT_INSTR":       [("FASTTEXT",0.3), ("INSTR",  0.7)],

    "FT_E5":          [("FASTTEXT",0.3), ("E5",     0.7)],
    "W2V_E5":         [("WORD2VEC",0.3), ("E5",     0.7)],

    "W2V_SPECTER":    [("WORD2VEC",0.3), ("SPECTER",0.7)],
    "FT_SPECTER":     [("FASTTEXT",0.3), ("SPECTER",0.7)],

    "W2V_JINA":       [("WORD2VEC",0.3), ("JINA",   0.7)],
    "FT_JINA":        [("FASTTEXT",0.3), ("JINA",   0.7)],

    "FT_MPNET":       [("FASTTEXT",0.3), ("MPNET",  0.7)],
    "W2V_MPNET":      [("WORD2VEC",0.3), ("MPNET",  0.7)],

    "FT_LABSE":       [("FASTTEXT",0.3), ("LABSE",  0.7)],
    "W2V_LABSE":      [("WORD2VEC",0.3), ("LABSE",  0.7)],
}

experiments = {name: [(name, 1.0)] for name in single_models}
for combo, weights in pairwise_combos.items():
    experiments[combo] = weights

print(f"→ Total experiments: {len(experiments)}")
print("→ Experiments:", list(experiments.keys()))

# ─── B) SET SEEDS & DEVICE ───────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"→ Using device: {DEVICE}")

# ─── C) SET UP BioPortal & Entrez ─────────────────────────────────────────────────
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY", "")
Entrez.email = os.getenv("EMAIL_FOR_NCBI", "demo@example.com")
nlp = spacy.load("en_core_web_sm")  # for noun-chunk fallback

# ─── D) EXTERNAL KNOWLEDGE LOOKUP ─────────────────────────────────────────────────
def get_bioportal_definition(term: str) -> str:
    """Return a definition from BioPortal or \"\"."""
    if not BIOPORTAL_API_KEY:
        return ""
    url = "http://data.bioontology.org/search"
    params = {
        "q": term,
        "apikey": BIOPORTAL_API_KEY,
        "ontologies": "MESH,NCIT,GO,BTO,PR,DOID",
        "pagesize": 1
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        if data.get("collection"):
            first = data["collection"][0]
            definition = first.get("prefLabel") or first.get("definition") or ""
            return definition
    except Exception:
        return ""
    return ""

def get_pubmed_abstract(term: str) -> str:
    """Return the first ~300 chars of the top PubMed abstract for `term`, or \"\"."""
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=1)
        rec = Entrez.read(handle)
        handle.close()
        ids = rec.get("IdList", [])
        if not ids:
            return ""
        handle = Entrez.efetch(db="pubmed", id=ids[0], rettype="abstract", retmode="text")
        ab = handle.read().strip()
        handle.close()
        snippet = ab.split("\n\n")[0]
        return snippet[:300] + "…" if len(snippet) > 300 else snippet
    except Exception:
        return ""

def enrich_question(q: str) -> (str, str):
    """
    Always attempt external knowledge: BioPortal full‐question → noun chunks → PubMed.
    Return (source, text). If BioPortal finds nothing, still return ("PubMed", snippet_or_empty).
    """
    # 1) Full question to BioPortal
    d = get_bioportal_definition(q)
    if d:
        return "BioPortal", d
    # 2) Noun-chunk fallback
    doc = nlp(q)
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if len(text) < 3:
            continue
        d2 = get_bioportal_definition(text)
        if d2:
            return "BioPortal", d2
    # 3) PubMed fallback
    snippet = get_pubmed_abstract(q)
    if snippet:
        return "PubMed", snippet
    # 4) Fallback: still record "PubMed" but with empty text
    return "PubMed", ""

# ─── E) ROUGE SETUP ───────────────────────────────────────────────────────────────
metrics = ["rouge1", "rouge2", "rougeL"]
scorer  = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

def eval_rouge(hyps, refs):
    """
    Compute simple average precision, recall, F1 for each metric across all pairs.
    Returns a dict with keys:
      rouge1_P, rouge1_R, rouge1_F,
      rouge2_P, rouge2_R, rouge2_F,
      rougeL_P, rougeL_R, rougeL_F
    """
    all_scores = {m: {"P":[], "R":[], "F":[]} for m in metrics}
    for hyp, ref in zip(hyps, refs):
        scores = scorer.score(ref, hyp)
        for m in metrics:
            all_scores[m]["P"].append(scores[m].precision)
            all_scores[m]["R"].append(scores[m].recall)
            all_scores[m]["F"].append(scores[m].fmeasure)
    out = {}
    for m in metrics:
        out[f"{m}_P"] = float(np.mean(all_scores[m]["P"]))
        out[f"{m}_R"] = float(np.mean(all_scores[m]["R"]))
        out[f"{m}_F"] = float(np.mean(all_scores[m]["F"]))
    return out

# ─── F) LOAD DATASETS ─────────────────────────────────────────────────────────────
print("\n→ Loading full ‘yes’ dataset and test set…")
df_full = pd.read_json("ori_pqaa.json", orient="index")
df_full = df_full[df_full["final_decision"] == "yes"].reset_index(drop=True)

df_test = pickle.load(open("testqa.pkl", "rb"))
assert isinstance(df_test, pd.DataFrame), "testqa.pkl must be a DataFrame"

# Sample 50 random test questions
np.random.seed(SEED)
sample_idxs = np.random.choice(len(df_test), size=SAMPLE_SIZE, replace=False)
df_sample = df_test.iloc[sample_idxs].reset_index(drop=True)
all_contexts = df_full["CONTEXTS"].tolist()
ref_answers   = df_sample["LONG_ANSWER"].tolist()

print(f"→ Loaded {len(df_test)} test questions; sampling {SAMPLE_SIZE} for eval.")

# ─── G) LOAD TEST EMBEDDINGS ───────────────────────────────────────────────────────
embeddings = {}
for model_name in single_models:
    fname = f"test_emb_{model_name.lower()}.pkl"
    try:
        obj = pickle.load(open(fname, "rb"))
        if isinstance(obj, dict):
            Qt = np.asarray(obj["Q_test"], dtype=np.float32)
            Ct = np.asarray(obj["C_train"], dtype=np.float32)
        elif isinstance(obj, (list, tuple)) and len(obj) == 2:
            Qt = np.asarray(obj[0], dtype=np.float32)
            Ct = np.asarray(obj[1], dtype=np.float32)
        else:
            raise RuntimeError(f"{fname} has unexpected structure.")
        embeddings[model_name] = (Qt, Ct)
        print(f"→ Loaded test embeddings for: {model_name}")
    except FileNotFoundError:
        print(f"  • WARNING: {fname} not found; skipping {model_name}")

missing = set(single_models) - set(embeddings.keys())
if missing:
    raise RuntimeError(f"Missing embeddings for: {missing}")

# ─── H) CONNECT TO QDRANT ─────────────────────────────────────────────────────────
qdrant = QdrantClient(url="http://localhost:6333")
collections = [c.name for c in qdrant.get_collections().collections]
for m in single_models:
    if m not in collections:
        raise RuntimeError(f"Qdrant collection `{m}` not found.")
print("\n→ Qdrant collections verified. Ready to evaluate.")

# ─── I) MAIN EVAL LOOP ─────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
records = []
graph = nx.DiGraph()

for exp_idx, (exp_name, parts) in enumerate(experiments.items(), start=1):
    print(f"\n→ Experiment ({exp_idx}/{len(experiments)}): {exp_name}")
    hyps, sims_all = [], []

    # Prepare weighted ensemble for C_train
    C_parts = [embeddings[m][1] for m, _ in parts]
    ws      = np.array([w for _, w in parts], dtype=np.float32)
    C_stack = np.stack(C_parts, axis=0)  # shape: (num_parts, N_train, D)

    for i, row in df_sample.iterrows():
        q_text  = row["QUESTION"]
        idx0    = sample_idxs[i]

        # Build ensembled Q vector
        Q_parts = [embeddings[m][0][idx0] for m, _ in parts]
        Q_stack = np.stack(Q_parts, axis=0)                # (num_parts, D)
        Q_ens   = np.tensordot(ws, Q_stack, axes=(0, 0))   # (D,)

        # Build ensembled C matrix
        C_ens = np.tensordot(ws, C_stack, axes=(0, 0))     # (N_train, D)

        # Compute all cosine similarities for metrics
        cos_scores = util.cos_sim(
            torch.from_numpy(Q_ens),
            torch.from_numpy(C_ens)
        ).numpy().ravel()
        sims_all.extend(cos_scores.tolist())

        # Qdrant retrieval with retry logic
        for attempt in range(3):
            try:
                resp = qdrant.search(
                    collection_name=parts[0][0],
                    query_vector=Q_ens.tolist(),
                    limit=TOP_K
                )
                break
            except ResponseHandlingException:
                if attempt < 2:
                    time.sleep(1 * (2 ** attempt))
                    continue
                else:
                    raise
        hits = sorted(resp, key=lambda h: -h.score)
        best_idxs = [h.payload["idx"] for h in hits]

        # Get chosen contexts (join if list)
        chosen_contexts = []
        for j in best_idxs:
            cval = all_contexts[j]
            if isinstance(cval, list):
                chosen_contexts.append("\n\n".join(cval))
            else:
                chosen_contexts.append(cval)

        # Always enrich with external knowledge
        source, enrichment_text = enrich_question(q_text)

        # Build prompt
        prompt_lines = []
        prompt_lines.append("Use the following context + external knowledge to answer succinctly.\n\n")
        prompt_lines.append("Context:\n")
        prompt_lines.append("\n\n".join(chosen_contexts) + "\n\n")
        prompt_lines.append(f"[External Knowledge ({source})]: {enrichment_text}\n\n")
        prompt_lines.append(f"Question: {q_text}\nAnswer:")
        prompt = "".join(prompt_lines)

        # Call GPT-3.5-turbo with retry
        answer = ""
        for attempt in range(3):
            try:
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMP,
                    max_tokens=MAX_TOKENS,
                )
                answer = chat.choices[0].message.content.strip()
                break
            except openai_error.ServiceUnavailableError:
                if attempt < 2:
                    time.sleep(1 * (2 ** attempt))
                    continue
                else:
                    raise
        hyps.append(answer)

        # Build graph edges
        for j in best_idxs:
            graph.add_edge(f"Q{i}", f"C{j}", label="uses_context")
        graph.add_edge(f"Q{i}", source, label=f"enriched_by_{source}")

    # Compute metrics
    rouge_stats = eval_rouge(hyps, ref_answers := ref_answers)
    mean_cos   = float(np.mean(sims_all))

    # Mean Euclidean (top‐1 per question)
    euc_all = []
    for i, row in df_sample.iterrows():
        idx0 = sample_idxs[i]
        Q_parts = [embeddings[m][0][idx0] for m, _ in parts]
        Q_stack = np.stack(Q_parts, axis=0)
        Q_ens   = np.tensordot(ws, Q_stack, axes=(0, 0))
        C_ens   = np.tensordot(ws, C_stack, axes=(0, 0))
        sims    = util.cos_sim(torch.from_numpy(Q_ens), torch.from_numpy(C_ens)).numpy().ravel()
        best_j  = sims.argsort()[::-1][0]
        c_vec   = C_ens[best_j]
        euc_all.append(distance.euclidean(Q_ens, c_vec))
    mean_euc = float(np.mean(euc_all))

    record = {
        "model":       exp_name,
        "mean_cosine": mean_cos,
        "mean_euclid": mean_euc,
        **rouge_stats
    }
    records.append(record)

# ─── J) SAVE & PRINT RESULTS ───────────────────────────────────────────────────────
df_out = pd.DataFrame(records).sort_values("mean_cosine", ascending=False)
print("\n=== Final results ===")
print(df_out.to_string(index=False))
df_out.to_csv("eval_with_qdrant_enriched_with_graph_results.csv", index=False)
print("→ Saved → eval_with_qdrant_enriched_with_graph_results.csv")

# ─── K) SAVE KNOWLEDGE GRAPH ───────────────────────────────────────────────────────
nx.write_gml(graph, "knowledge_graph.gml")
print("→ Knowledge graph saved to knowledge_graph.gml")
