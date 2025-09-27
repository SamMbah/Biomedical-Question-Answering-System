#!/usr/bin/env python3
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient
from openai import OpenAI
from rouge_score import rouge_scorer
from scipy.spatial import distance
from sentence_transformers import util

# ─── 0) CONFIG ────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 50
TOP_K       = 3
TEMP        = 0.0
MAX_TOKENS  = 256

# ─── 1) LOAD & SAMPLE DATA ────────────────────────────────────────────────────
df_full = pd.read_json("ori_pqaa.json", orient="index")
df_yes  = df_full[df_full["final_decision"] == "yes"].reset_index(drop=True)
min_len = df_yes.shape[0]  # will truncate below to match embeddings
print(f"→ {min_len} positive examples in dataset")

# ─── 2) LOAD EMBEDDINGS INTO MEMORY ───────────────────────────────────────────
embeddings = {}

# SBERT pickles
SBERT_FILES = [
    "emb_instr.pkl","emb_jina.pkl","emb_e5.pkl","emb_specter.pkl",
    "emb_mpnet.pkl","emb_minilm6.pkl","emb_minilm12.pkl","emb_labse.pkl"
]
for fn in SBERT_FILES:
    name = fn.replace("emb_", "").replace(".pkl", "").upper()
    Q_all, C_all = pickle.load(open(fn, "rb"))
    embeddings[name] = (np.asarray(Q_all), np.asarray(C_all))

# Word2Vec + FastText
df_w2v = pickle.load(open("emb_w2v_ft.pkl", "rb"))
embeddings["WORD2VEC"] = (
    np.vstack(df_w2v["WORD2VEC_EMB"].tolist()),
    np.vstack(df_w2v["WORD2VEC_C_EMB"].tolist())
)
embeddings["FASTTEXT"] = (
    np.vstack(df_w2v["FASTTEXT_Q"].tolist()),
    np.vstack(df_w2v["FASTTEXT_C"].tolist())
)

# align dataset to smallest embedding length
min_len = min(Q.shape[0] for Q, _ in embeddings.values())
df      = df_yes.iloc[:min_len].reset_index(drop=True)
all_ctx = df["CONTEXTS"].tolist()
refs    = df["LONG_ANSWER"].tolist()

# sample indices
random.seed(42)
sample_idxs = random.sample(range(min_len), SAMPLE_SIZE)
samples     = df.loc[sample_idxs].reset_index(drop=True)
print(f"→ Sampled {SAMPLE_SIZE} indices")

# ─── 3) DEFINE EXPERIMENTS ────────────────────────────────────────────────────
experiments = { name: [(name,1.0)] for name in embeddings.keys() }
sb_pairs = [
    ("INSTR","E5"),("INSTR","SPECTER"),("INSTR","JINA"),
    ("E5","MPNET"),("MPNET","LABSE"),("MINILM6","MINILM12")
]
for A, B in sb_pairs:
    experiments[f"{A}+{B}"] = [(A,0.7), (B,0.3)]
for base in ["INSTR","E5","SPECTER","JINA","MPNET","LABSE"]:
    experiments[f"W2V×0.3+{base}×0.7"] = [("WORD2VEC",0.3),(base,0.7)]
    experiments[f"FT×0.3+{base}×0.7"]  = [("FASTTEXT",0.3),(base,0.7)]

print("→ Experiments:", list(experiments.keys()))

# ─── 4) QDRANT & ROUGE SETUP ──────────────────────────────────────────────────
qdrant = QdrantClient(url="http://localhost:6333")
scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

openai = OpenAI()

# ─── 5) MAIN EVAL LOOP ────────────────────────────────────────────────────────
records = []
for exp_name, parts in experiments.items():
    print(f"\n→ Running {exp_name}")
    hyps = []
    cos_sims = []
    euc_dists = []

    # per-ROUGE lists
    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []
    rl_p, rl_r, rl_f = [], [], []

    for local_i, row in samples.iterrows():
        idx0 = sample_idxs[local_i]

        # ensemble question vector
        Q_parts = [embeddings[m][0][idx0] * w for m,w in parts]
        Q_ens   = np.sum(Q_parts, axis=0)

        # retrieve and aggregate scores
        score_acc = {}
        for m,w in parts:
            resp = qdrant.search(
                collection_name=m,
                query_vector=Q_ens.tolist(),
                limit=TOP_K
            )
            for hit in resp:
                i = hit.payload["idx"]
                score_acc[i] = score_acc.get(i, 0.0) + w*hit.score

        topk = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        chosen_idxs = [i for i,_ in topk]

        # compute retrieval metrics
        for i in chosen_idxs:
            C_parts = [embeddings[m][1][i] * w for m,w in parts]
            C_ens   = np.sum(C_parts, axis=0)
            cos_sims.append(util.cos_sim(
                torch.from_numpy(Q_ens),
                torch.from_numpy(C_ens)
            ).item())
            euc_dists.append(distance.euclidean(Q_ens, C_ens))

        # build prompt contexts
        ctxs = []
        for i in chosen_idxs:
            ctx = all_ctx[i]
            ctxs.append("\n\n".join(ctx) if isinstance(ctx, list) else ctx)

        prompt = (
            "Use the following context to answer succinctly.\n\n"
            "Context:\n" + "\n\n".join(ctxs) +
            f"\n\nQuestion: {row['QUESTION']}\nAnswer:"
        )
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=TEMP,
            max_tokens=MAX_TOKENS
        )
        answer = resp.choices[0].message.content.strip()
        hyps.append(answer)

        # compute ROUGE for this pair
        sc = scorer.score(row["LONG_ANSWER"], answer)
        r1_p.append(sc["rouge1"].precision)
        r1_r.append(sc["rouge1"].recall)
        r1_f.append(sc["rouge1"].fmeasure)
        r2_p.append(sc["rouge2"].precision)
        r2_r.append(sc["rouge2"].recall)
        r2_f.append(sc["rouge2"].fmeasure)
        rl_p.append(sc["rougeL"].precision)
        rl_r.append(sc["rougeL"].recall)
        rl_f.append(sc["rougeL"].fmeasure)

    # aggregate metrics
    stats = {
        "mean_cosine": float(np.mean(cos_sims)),
        "mean_euclid": float(np.mean(euc_dists)),
        "rouge1_P": float(np.mean(r1_p)),
        "rouge1_R": float(np.mean(r1_r)),
        "rouge1_F": float(np.mean(r1_f)),
        "rouge2_P": float(np.mean(r2_p)),
        "rouge2_R": float(np.mean(r2_r)),
        "rouge2_F": float(np.mean(r2_f)),
        "rougeL_P": float(np.mean(rl_p)),
        "rougeL_R": float(np.mean(rl_r)),
        "rougeL_F": float(np.mean(rl_f)),
    }

    records.append({"model": exp_name, **stats})

# ─── 6) REPORT ────────────────────────────────────────────────────────────────
df_out = pd.DataFrame(records).sort_values("mean_cosine", ascending=False)
print("\n=== Final results ===")
print(df_out.to_markdown(index=False))
df_out.to_csv("eval_full_metrics.csv", index=False)
print("→ Saved → eval_full_metrics.csv")
