#!/usr/bin/env python3
# eval_test_enriched.py

"""
Runs 38 embedding+GPT-3.5 experiments on 50 sampled test questions from MedQA,
enriching each prompt with UMLS, BioPortal, and MeSH definitions in union.
Implements retry logic around Qdrant searches to avoid timeouts crashing the run.
"""

import os
import json
import pickle
import random
import requests
import networkx as nx
import xml.etree.ElementTree as ET
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sentence_transformers import util
from qdrant_client import QdrantClient
from openai import OpenAI
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split

from UmlsClient import UmlsClient  # placed alongside this script

# ───────────────────────────────────────────────────────────────────────────────
# 0) CONFIG
# ───────────────────────────────────────────────────────────────────────────────
SEED              = 42
TEST_SIZE         = 0.2
TEST_SAMPLE_SIZE  = 50
TOP_K             = 3
TEMP              = 0.0
MAX_TOKENS        = 256   # max GPT tokens per completion
QDRANT_MAX_RETRIES = 2    # number of times to retry Qdrant search if it times out
QDRANT_RETRY_SLEEP = 1.0  # seconds to sleep between Qdrant retries

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ───────────────────────────────────────────────────────────────────────────────
# 1) LOAD FULL DATA & SAMPLE 50 TEST QUESTIONS
# ───────────────────────────────────────────────────────────────────────────────
print("Step A: Loading full dataset and sampling 50 test questions…")
df_full = pd.read_json("ori_pqaa.json", orient="index")
df_full = df_full.drop_duplicates(subset="QUESTION").reset_index(drop=True)

# Ensure CONTEXTS is a single string (take first element if it's a list)
df_full["CONTEXTS"] = df_full["CONTEXTS"].apply(
    lambda x: x[0] if isinstance(x, list) and x else x
)

# If there is a “final_decision” column, keep only “yes”
if "final_decision" in df_full.columns:
    df_full = df_full[df_full["final_decision"] == "yes"].reset_index(drop=True)

# Keep master list of contexts (index must match Qdrant ingestion)
full_contexts = df_full["CONTEXTS"].tolist()

# Split into train/test so contexts align with Qdrant’s “train” ingestion
train_df, test_df = train_test_split(
    df_full, test_size=TEST_SIZE, random_state=SEED
)

# Sample exactly 50 questions from test set
test_df = test_df.sample(n=TEST_SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
test_questions = test_df["QUESTION"].tolist()
test_refs      = test_df["LONG_ANSWER"].tolist()
n_test         = len(test_questions)
print(f"   • Sampled {n_test} test questions.\n")

# ───────────────────────────────────────────────────────────────────────────────
# 2) LOAD ALL TEST EMBEDDINGS (test_emb_<MODEL>.pkl)
# ───────────────────────────────────────────────────────────────────────────────
print("Step B: Loading all test embeddings (10 models)…")
embeddings = {}
for fn in sorted(os.listdir(".")):
    if fn.startswith("test_emb_") and fn.endswith(".pkl"):
        name = fn.replace("test_emb_", "").replace(".pkl", "").upper()
        Q_test, C_train_arr = pickle.load(open(fn, "rb"))
        embeddings[name] = (
            np.asarray(Q_test, dtype=np.float32),
            np.asarray(C_train_arr, dtype=np.float32),
        )

expected_keys = {
    "INSTR","JINA","E5","SPECTER","MPNET","MINILM6",
    "MINILM12","LABSE","WORD2VEC","FASTTEXT"
}
missing = expected_keys - set(embeddings.keys())
if missing:
    raise RuntimeError(f"Missing test embeddings for: {missing}")
print("   All 10 test embeddings loaded.\n")

# ───────────────────────────────────────────────────────────────────────────────
# 3) DEFINE 38 EXPERIMENTS (10 singletons + 28 two-way mixtures)
# ───────────────────────────────────────────────────────────────────────────────
print("Step C: Defining the 38 experiments…")
experiments = { name: [(name, 1.0)] for name in embeddings.keys() }

# SBERT‐to‐SBERT mixes (0.7/0.3 weights)
sb_pairs = [
    ("INSTR","E5"), ("INSTR","SPECTER"), ("INSTR","JINA"),
    ("INSTR","MPNET"), ("INSTR","LABSE"), ("E5","SPECTER"),
    ("E5","MPNET"), ("E5","LABSE"), ("JINA","E5"),
    ("JINA","SPECTER"), ("JINA","MPNET"), ("JINA","LABSE"),
    ("SPECTER","MPNET"), ("SPECTER","LABSE"), ("MPNET","LABSE"),
    ("MINILM6","MINILM12"),
]
for A, B in sb_pairs:
    experiments[f"{A}+{B}"] = [(A,0.7),(B,0.3)]

# WORD2VEC/FASTTEXT to SBERT mixes (0.3/0.7 weights)
for base in ["INSTR","E5","SPECTER","JINA","MPNET","LABSE"]:
    experiments[f"W2V×0.3+{base}×0.7"] = [("WORD2VEC",0.3),(base,0.7)]
    experiments[f"FT×0.3+{base}×0.7"]  = [("FASTTEXT",0.3),(base,0.7)]

exp_names = sorted(experiments.keys())
print(f"   • Experiments count: {len(exp_names)}")
for name in exp_names:
    print(f"     • {name}")
print()

# ───────────────────────────────────────────────────────────────────────────────
# 4) INITIALIZE ROUGE, OPENAI, UMLS CLIENT
# ───────────────────────────────────────────────────────────────────────────────
print("Step D: Initializing ROUGE, OpenAI, and UMLS client…")
scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
openai = OpenAI()

# UMLS client (handles 404s internally)
umls = UmlsClient(api_key=os.getenv("UMLS_API_KEY"))
umls.authenticate()
print("    Ready.\n")

# ───────────────────────────────────────────────────────────────────────────────
# 5) CONNECT TO QDRANT
# ───────────────────────────────────────────────────────────────────────────────
print("Step E: Connecting to Qdrant…")
qdrant = QdrantClient(url="http://localhost:6333")
print("    Connected.\n")

# ───────────────────────────────────────────────────────────────────────────────
# 6) HELPER FUNCTIONS FOR BIOPORTAL & MeSH
# ───────────────────────────────────────────────────────────────────────────────

def fetch_bioportal_definition(text: str):
    """
    Query BioPortal (SNOMEDCT + RxNorm) for a definition or prefLabel.
    Returns a string if found, else None.
    """
    api_key = os.getenv("BIOPORTAL_API_KEY")
    if not api_key:
        return None

    url = "https://data.bioontology.org/search"
    params = {
        'q': text,
        'ontologies': "SNOMEDCT,RXNORM",
        'pagesize': 5,
        'apikey': api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=5.0)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    data = resp.json()
    coll = data.get("collection", [])
    if not coll:
        return None

    first = coll[0]
    defs = first.get("definition", [])
    if isinstance(defs, list) and defs:
        return defs[0].strip()

    pl = first.get("prefLabel")
    if pl:
        return pl.strip()

    return None


def fetch_mesh_definition(text: str):
    """
    1) ESearch: find MeSH ID for `text`
    2) EFetch: retrieve the MeSH descriptor record (XML) and extract <ScopeNote> or <Definition>
    Returns a string if found, else None.
    """
    # (A) ESearch
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = { 'db': 'mesh', 'term': text, 'retmode': 'json' }
    try:
        resp = requests.get(esearch_url, params=params, timeout=5.0)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    data = resp.json()
    idlist = data.get("esearchresult", {}).get("idlist", [])
    if not idlist:
        return None

    mesh_id = idlist[0]
    # (B) EFetch
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = { 'db': 'mesh', 'id': mesh_id, 'retmode': 'xml' }
    try:
        resp = requests.get(efetch_url, params=params, timeout=5.0)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    try:
        xml_root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    # Look for <ScopeNote> first
    scope_node = xml_root.find(".//ScopeNote")
    if scope_node is not None and scope_node.text:
        return scope_node.text.strip()

    # Then <Definition>
    definition_node = xml_root.find(".//Definition/DescriptorName")
    if definition_node is not None and definition_node.text:
        return definition_node.text.strip()

    return None


# ───────────────────────────────────────────────────────────────────────────────
# 7) MAIN “ENRICHED QA” LOOP (with Qdrant retry logic)
# ───────────────────────────────────────────────────────────────────────────────
print("Step F: Running enriched QA on the sampled test set…\n")

records      = []
examples     = []
all_kg_edges = []

for exp_name in exp_names:
    parts = experiments[exp_name]
    print(f"\n→ Experiment: `{exp_name}`")

    cos_sims   = []
    euc_dists  = []
    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []
    rl_p, rl_r, rl_f = [], [], []

    ex_q, ex_gen, ex_gt, ex_know = None, None, None, None
    exp_kg_edges = { f"Q{i}": [] for i in range(n_test) }

    for i, question in enumerate(test_questions):
        q_key = f"Q{i}"

        # 1) Build ensemble question embedding
        Q_parts = [ embeddings[m][0][i] * w for (m, w) in parts ]
        Q_ens   = np.sum(Q_parts, axis=0)

        # 2) RETRIEVE TOP‐K CONTEXTS FROM QDRANT WITH RETRY LOGIC
        score_acc = {}
        for (m, w) in parts:
            resp = []
            attempts = 0
            while attempts < QDRANT_MAX_RETRIES:
                try:
                    resp = qdrant.search(
                        collection_name=m,
                        query_vector=Q_ens.tolist(),
                        limit=TOP_K
                    )
                    break
                except Exception as e:
                    attempts += 1
                    print(f" Warning: Qdrant search timeout for model `{m}` (exp `{exp_name}`, Q{i}). Attempt {attempts}/{QDRANT_MAX_RETRIES}. Error: {e}")
                    time.sleep(QDRANT_RETRY_SLEEP)
            # If still empty (resp==[]), we just skip (no contexts retrieved for this model)
            for hit in resp:
                j = hit.payload["idx"]
                score_acc[j] = score_acc.get(j, 0.0) + w * hit.score

        # Take top‐K by descending weighted score
        topk = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        chosen_idxs = [ j for (j, _) in topk ]

        # Record KG edges: Q<i> → retrieved_context “C{j}”
        for j in chosen_idxs:
            ctx_key = f"C{j}"
            exp_kg_edges[q_key].append((q_key, "retrieved_context", ctx_key))

        # 3) Compute cosine + Euclid for each chosen j
        for j in chosen_idxs:
            C_parts = [ embeddings[m][1][j] * w for (m, w) in parts ]
            C_ens   = np.sum(C_parts, axis=0)
            cos_sims.append(
                util.cos_sim(torch.from_numpy(Q_ens),
                             torch.from_numpy(C_ens)).item()
            )
            euc_dists.append(distance.euclidean(Q_ens, C_ens))

        # 4) CASCADING + UNION ENRICHMENT (UMLS ∪ BioPortal ∪ MeSH)
        definition_snippets = []
        concept_set = set()

        # (a) CUIs from question
        try:
            q_search = umls.find_cuis_for_text(question)
            for res in q_search["result"]["results"][:5]:
                cui = res["ui"]
                concept_set.add(cui)
                exp_kg_edges[q_key].append((q_key, "mentions_CUI", cui))
        except Exception:
            pass

        # (b) CUIs from retrieved contexts
        for j in chosen_idxs:
            ctx_key  = f"C{j}"
            ctx_text = full_contexts[j]
            try:
                c_search = umls.find_cuis_for_text(ctx_text[:200])
                for res in c_search["result"]["results"][:3]:
                    cui = res["ui"]
                    concept_set.add(cui)
                    exp_kg_edges[q_key].append((ctx_key, "mentions_CUI", cui))
            except Exception:
                pass

        # (c) For each CUI, fetch UMLS definitions
        umls_found = False
        for cuid in list(concept_set)[:5]:
            try:
                details   = umls.get_cui_definitions(cuid)
                defs_list = details.get("result", [])
            except Exception:
                defs_list = []

            if isinstance(defs_list, list) and defs_list:
                d0  = defs_list[0]
                val = d0.get("value", "")
                src = d0.get("rootSource", "")
                if val:
                    snippet = f"UMLS {cuid} ({src}): {val}"
                    definition_snippets.append(snippet)
                    umls_found = True

                # Record parent‐child edges
                for parent in d0.get("parents", []):
                    parent_cui = parent.get("ui")
                    if parent_cui:
                        exp_kg_edges[q_key].append((cuid, "child_of", parent_cui))

        # (d) Always attempt BioPortal on question + contexts
        bp_found = False
        bp_def_q = fetch_bioportal_definition(question)
        if bp_def_q:
            definition_snippets.append(f"BioPortal: {bp_def_q}")
            exp_kg_edges[q_key].append((q_key, "mentions_BioPortal", "BioPortal"))
            bp_found = True

        if not bp_found:
            for j in chosen_idxs:
                ctx_snip    = full_contexts[j][:200]
                bp_def_ctx  = fetch_bioportal_definition(ctx_snip)
                if bp_def_ctx:
                    snippet = f"BioPortal: {bp_def_ctx}"
                    definition_snippets.append(snippet)
                    ctx_key = f"C{j}"
                    exp_kg_edges[q_key].append((ctx_key, "mentions_BioPortal", "BioPortal"))
                    bp_found = True
                    break

        # (e) If neither UMLS nor BioPortal yielded anything, fall back to MeSH
        mesh_found = False
        if not (umls_found or bp_found):
            mesh_def_q = fetch_mesh_definition(question)
            if mesh_def_q:
                definition_snippets.append(f"MeSH: {mesh_def_q}")
                exp_kg_edges[q_key].append((q_key, "mentions_MeSH", "MeSH"))
                mesh_found = True
            else:
                for j in chosen_idxs:
                    mesh_def_ctx = fetch_mesh_definition(full_contexts[j][:200])
                    if mesh_def_ctx:
                        snippet = f"MeSH: {mesh_def_ctx}"
                        definition_snippets.append(snippet)
                        ctx_key = f"C{j}"
                        exp_kg_edges[q_key].append((ctx_key, "mentions_MeSH", "MeSH"))
                        mesh_found = True
                        break

        # (f) If all three sources yield nothing, add placeholder
        if not (umls_found or bp_found or mesh_found):
            definition_snippets.append("No external knowledge found.")

        # 5) Build final prompt (contexts + all definitions)
        retrieved_text = "\n\n".join(full_contexts[j] for j in chosen_idxs)
        ext_know_text  = "\n".join(definition_snippets)

        prompt = (
            "Use the following context AND external knowledge to answer succinctly.\n\n"
            "=== Retrieved Contexts ===\n" +
            retrieved_text +
            "\n\n=== External Knowledge ===\n" +
            ext_know_text +
            f"\n\nQuestion: {question}\nAnswer:"
        )

        # 6) Call GPT-3.5 to generate answer
        chat = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=TEMP,
            max_tokens=MAX_TOKENS
        )
        answer = chat.choices[0].message.content.strip()

        # 7) Compute ROUGE vs ground truth
        gt = test_refs[i]
        sc = scorer.score(gt, answer)
        r1_p.append(sc["rouge1"].precision)
        r1_r.append(sc["rouge1"].recall)
        r1_f.append(sc["rouge1"].fmeasure)
        r2_p.append(sc["rouge2"].precision)
        r2_r.append(sc["rouge2"].recall)
        r2_f.append(sc["rouge2"].fmeasure)
        rl_p.append(sc["rougeL"].precision)
        rl_r.append(sc["rougeL"].recall)
        rl_f.append(sc["rougeL"].fmeasure)

        # Save the first question’s example for this experiment
        if i == 0:
            ex_q     = question
            ex_gen   = answer
            ex_gt    = gt
            ex_know  = ext_know_text

    # 8) Aggregate metrics for this experiment
    stats = {
        "model":       exp_name,
        "mean_cosine": float(np.mean(cos_sims)) if cos_sims else 0.0,
        "mean_euclid": float(np.mean(euc_dists)) if euc_dists else 0.0,
        "rouge1_P":    float(np.mean(r1_p)) if r1_p else 0.0,
        "rouge1_R":    float(np.mean(r1_r)) if r1_r else 0.0,
        "rouge1_F":    float(np.mean(r1_f)) if r1_f else 0.0,
        "rouge2_P":    float(np.mean(r2_p)) if r2_p else 0.0,
        "rouge2_R":    float(np.mean(r2_r)) if r2_r else 0.0,
        "rouge2_F":    float(np.mean(r2_f)) if r2_f else 0.0,
        "rougeL_P":    float(np.mean(rl_p)) if rl_p else 0.0,
        "rougeL_R":    float(np.mean(rl_r)) if rl_r else 0.0,
        "rougeL_F":    float(np.mean(rl_f)) if rl_f else 0.0,
    }
    records.append(stats)

    # 9) Save one representative example for this experiment
    examples.append({
        "model":             exp_name,
        "question":          ex_q,
        "generated_answer":  ex_gen,
        "ground_truth":      ex_gt,
        "external_knowledge": ex_know
    })

    # 10) Merge KG edges (prefix nodes with experiment name)
    for qidx in exp_kg_edges:
        for (subj, pred, obj) in exp_kg_edges[qidx]:
            subj_node = f"{exp_name}:{subj}"
            # If this is a CUI, prefix with “CUI:”
            if pred == "mentions_CUI":
                obj_node = f"{exp_name}:CUI:{obj}"
            else:
                # obj might be “MeSH”, “BioPortal”, or a context key “C12345”
                obj_node = f"{exp_name}:{obj}"
            all_kg_edges.append((exp_name, subj_node, pred, obj_node))

# ───────────────────────────────────────────────────────────────────────────────
# 8) SAVE FINAL METRICS & EXAMPLES
# ───────────────────────────────────────────────────────────────────────────────
print("\nStep G: Saving final results…")
df_metrics = pd.DataFrame(records).sort_values("mean_cosine", ascending=False)
print(df_metrics.to_markdown(index=False))

df_metrics.to_csv("eval_test_enriched_results.csv", index=False)
print("    Saved evaluation metrics → eval_test_enriched_results.csv")

pd.DataFrame(examples).to_csv("examples_test_enriched.csv", index=False)
with open("examples_test_enriched.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)
print("    Saved example Q/A/GT → examples_test_enriched.csv / .json\n")

# ───────────────────────────────────────────────────────────────────────────────
# 9) BUILD & SAVE COMBINED KNOWLEDGE GRAPH
# ───────────────────────────────────────────────────────────────────────────────
print("Step H: Building combined Knowledge Graph…")
G = nx.DiGraph()
for (_, subj_node, pred, obj_node) in all_kg_edges:
    G.add_node(subj_node)
    G.add_node(obj_node)
    G.add_edge(subj_node, obj_node, relation=pred)

kg_data = {
    "nodes": list(G.nodes),
    "edges": [
        {"source": u, "target": v, "relation": G[u][v]["relation"]}
        for u, v in G.edges()
    ]
}
with open("knowledge_graph.json", "w", encoding="utf-8") as f:
    json.dump(kg_data, f, indent=2)

print("    Saved knowledge graph → knowledge_graph.json\n")
print(" All 38 enriched experiments completed successfully.")
