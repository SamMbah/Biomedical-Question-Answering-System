#!/usr/bin/env python3
"""
eval_test_only.py

Load all test embeddings (test_emb_<model>.pkl) from the current directory,
re‐construct the train/test split (to get test questions + ground truths), 
then run retrieval + GPT‐3.5 + ROUGE.  **This version only evaluates on 50 test examples.**

Defines exactly 38 experiments (10 singletons + 16 SBERT‐SBERT + 12 W2V/FT‐SBERT).
Outputs:
  • eval_test_results.csv   – metrics table (38 rows)
  • examples_test.csv       – one Q/A/GT example per experiment
  • examples_test.json      – same examples in JSON
"""

import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sentence_transformers import util
from qdrant_client import QdrantClient
from openai import OpenAI
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SEED              = 42
TEST_SIZE         = 0.2
TEST_SAMPLE_SIZE  = 50      # ← CHANGED: only evaluate on 50 test examples
TOP_K             = 3
TEMP              = 0.0
MAX_TOKENS        = 256

# ─── A) Load full dataset, split out test set, but then sample 50 questions ────
print("Step A: Loading full dataset and splitting out test questions…")
df_full = pd.read_json("ori_pqaa.json", orient="index")

# 1) Drop duplicates by QUESTION
df_full = df_full.drop_duplicates(subset="QUESTION").reset_index(drop=True)

# 2) Flatten contexts (take first element if it's a nonempty list)
df_full["CONTEXTS"] = df_full["CONTEXTS"].apply(
    lambda x: x[0] if isinstance(x, list) and x else x
)

# 3) Keep only final_decision == "yes" if that column exists
if "final_decision" in df_full.columns:
    df_full = df_full[df_full["final_decision"] == "yes"].reset_index(drop=True)

# Save the full list of contexts (for retrieval via Qdrant)
full_contexts = df_full["CONTEXTS"].tolist()
n_full = len(full_contexts)
print(f"   • Full dataset contexts count: {n_full}")

# 4) Split into train/test (to extract test questions + ground truths)
train_df, test_df = train_test_split(
    df_full, test_size=TEST_SIZE, random_state=SEED
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"   • After splitting: train = {len(train_df)}, test = {len(test_df)}")

# 5) Now take a random sample of 50 test rows (for evaluation)
test_df = test_df.sample(n=TEST_SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)  # ← CHANGED
test_questions = test_df["QUESTION"].tolist()
test_refs      = test_df["LONG_ANSWER"].tolist()
n_test         = len(test_questions)
print(f"   • Sampled {n_test} test questions (for fast evaluation)\n")

train_contexts = train_df["CONTEXTS"].tolist()

# ─── B) Load all test embeddings from current directory ─────────────────────────
print("Step B: Loading all test embeddings (test_emb_<model>.pkl) from CWD…")
embeddings = {}  # key → (Q_test_array, C_train_array)

for fn in sorted(os.listdir(".")):
    if fn.startswith("test_emb_") and fn.endswith(".pkl"):
        name = fn.replace("test_emb_","").replace(".pkl","").upper()
        Q_test, C_train_arr = pickle.load(open(fn, "rb"))
        embeddings[name] = (
            np.asarray(Q_test, dtype=np.float32),
            np.asarray(C_train_arr, dtype=np.float32),
        )

loaded_keys = sorted(embeddings.keys())
print("   • Found test embeddings for models:")
print("     ", loaded_keys, "\n")

expected_keys = {
    "INSTR","JINA","E5","SPECTER","MPNET","MINILM6","MINILM12","LABSE","WORD2VEC","FASTTEXT"
}
missing_keys = expected_keys - set(loaded_keys)
if missing_keys:
    print("    ERROR: Missing embeddings for:", missing_keys)
    print("   Make sure you have generated all ten test_emb_*.pkl files.\n")
    exit(1)

print("    All ten test embeddings present.\n")

# ─── C) Define all 38 experiments ───────────────────────────────────────────────
print("Step C: Defining the 38 experiments…")
experiments = { name: [(name, 1.0)] for name in embeddings.keys() }

# (1) Two‐way SBERT ensembles (0.7/0.3) – including JINA+E5 and JINA+SPECTER
sb_pairs = [
    ("INSTR","E5"),
    ("INSTR","SPECTER"),
    ("INSTR","JINA"),
    ("INSTR","MPNET"),
    ("INSTR","LABSE"),
    ("E5","SPECTER"),
    ("E5","MPNET"),
    ("E5","LABSE"),
    ("JINA","E5"),
    ("JINA","SPECTER"),
    ("JINA","MPNET"),
    ("JINA","LABSE"),
    ("SPECTER","MPNET"),
    ("SPECTER","LABSE"),
    ("MPNET","LABSE"),
    ("MINILM6","MINILM12"),
]
for A, B in sb_pairs:
    experiments[f"{A}+{B}"] = [(A, 0.7), (B, 0.3)]

# (2) W2V×0.3 + SBERT×0.7  and  FT×0.3 + SBERT×0.7
for base in ["INSTR","E5","SPECTER","JINA","MPNET","LABSE"]:
    experiments[f"W2V×0.3+{base}×0.7"] = [("WORD2VEC", 0.3), (base, 0.7)]
    experiments[f"FT×0.3+{base}×0.7"]  = [("FASTTEXT", 0.3), (base, 0.7)]

exp_names = sorted(experiments.keys())
print(f"   • Total experiments defined: {len(exp_names)}")
for name in exp_names:
    print("     •", name)
print()

# ─── D) Prepare ROUGE & OpenAI client ──────────────────────────────────────────
print("Step D: Preparing ROUGE scorer and OpenAI client…")
metrics = ["rouge1","rouge2","rougeL"]
scorer  = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
openai  = OpenAI()
print("    Ready.\n")

# ─── E) Connect to Qdrant (all contexts were ingested previously) ───────────────
print("Step E: Connecting to Qdrant…")
qdrant = QdrantClient(url="http://localhost:6333")
print("    Connected.\n")

# ─── F) Main evaluation loop (38 experiments, each on 50 test questions) ──────
print("Step F: Running retrieval + GPT-3.5 + ROUGE on the sampled test set…\n")
records  = []
examples = []

for exp_name in exp_names:
    parts = experiments[exp_name]
    print(f"\n→ Experiment: `{exp_name}`")

    cos_sims   = []
    euc_dists  = []
    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []
    rl_p, rl_r, rl_f = [], [], []

    ex_q, ex_gen, ex_gt = None, None, None

    for i, question in enumerate(test_questions):
        # 1) Compute Q ensemble for this test question
        Q_parts = [ embeddings[m][0][i] * w for (m, w) in parts ]
        Q_ens   = np.sum(Q_parts, axis=0)

        # 2) Retrieve top-K contexts from Qdrant (all contexts) and accumulate weighted scores
        score_acc = {}
        for (m, w) in parts:
            resp = qdrant.search(
                collection_name=m,
                query_vector=Q_ens.tolist(),
                limit=TOP_K
            )
            for hit in resp:
                j = hit.payload["idx"]  # index into full_contexts
                score_acc[j] = score_acc.get(j, 0.0) + w * hit.score

        # 3) Select the top-K indices
        topk = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        chosen_idxs = [ j for (j, _) in topk ]

        # 4) For each chosen index j, compute cosine & Euclid between Q_ens and C_ens
        for j in chosen_idxs:
            C_parts = [ embeddings[m][1][j] * w for (m, w) in parts ]
            C_ens   = np.sum(C_parts, axis=0)
            cos_sims.append(
                util.cos_sim(torch.from_numpy(Q_ens),
                             torch.from_numpy(C_ens)).item()
            )
            euc_dists.append(distance.euclidean(Q_ens, C_ens))

        # 5) Build prompt using full_contexts[j]
        retrieved_ctxs = [ full_contexts[j] for j in chosen_idxs ]
        prompt = (
            "Use the following context to answer succinctly.\n\n"
            "Context:\n" + "\n\n".join(retrieved_ctxs) +
            f"\n\nQuestion: {question}\nAnswer:"
        )

        # 6) Call GPT‐3.5 to generate answer
        chat   = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=TEMP,
            max_tokens=MAX_TOKENS
        )
        answer = chat.choices[0].message.content.strip()

        # 7) Compute ROUGE against the ground‐truth
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

        # 8) Save the first example (i=0) for this experiment
        if i == 0:
            ex_q   = question
            ex_gen = answer
            ex_gt  = gt

    # 9) Aggregate metrics (simple mean)
    stats = {
        "model":        exp_name,
        "mean_cosine":  float(np.mean(cos_sims)),
        "mean_euclid":  float(np.mean(euc_dists)),
        "rouge1_P":     float(np.mean(r1_p)),
        "rouge1_R":     float(np.mean(r1_r)),
        "rouge1_F":     float(np.mean(r1_f)),
        "rouge2_P":     float(np.mean(r2_p)),
        "rouge2_R":     float(np.mean(r2_r)),
        "rouge2_F":     float(np.mean(r2_f)),
        "rougeL_P":     float(np.mean(rl_p)),
        "rougeL_R":     float(np.mean(rl_r)),
        "rougeL_F":     float(np.mean(rl_f)),
    }
    records.append(stats)

    # 10) Store that example for CSV/JSON
    examples.append({
        "model": exp_name,
        "question": ex_q,
        "generated_answer": ex_gen,
        "ground_truth": ex_gt
    })

# ─── G) Save final results ─────────────────────────────────────────────────────
print("\nStep G: Saving final results…")
df_metrics = pd.DataFrame(records).sort_values("mean_cosine", ascending=False)
print(df_metrics.to_markdown(index=False))

df_metrics.to_csv("eval_test_results.csv", index=False)
print("   Saved evaluation metrics → eval_test_results.csv")

pd.DataFrame(examples).to_csv("examples_test.csv", index=False)
with open("examples_test.json", "w") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)
print("    Saved example Q/A/GT → examples_test.csv + examples_test.json\n")

print(" All 38 experiments finished (each ran on 50 test questions).")
