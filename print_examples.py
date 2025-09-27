#!/usr/bin/env python3
import pickle
import random
import json

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 50
TOP_K       = 3
TEMP        = 0.0
MAX_TOKENS  = 256

# ─── 1) LOAD & TRUNCATE DATA ──────────────────────────────────────────────────
df = pd.read_json("ori_pqaa.json", orient="index")
df = df[df["final_decision"]=="yes"].reset_index(drop=True)

# ─── 2) LOAD EMBEDDINGS ───────────────────────────────────────────────────────
embeddings = {}
for fn in [
    "emb_instr.pkl","emb_jina.pkl","emb_e5.pkl","emb_specter.pkl",
    "emb_mpnet.pkl","emb_minilm6.pkl","emb_minilm12.pkl","emb_labse.pkl"
]:
    name = fn.replace("emb_","").replace(".pkl","").upper()
    Q_all, C_all = pickle.load(open(fn,"rb"))
    embeddings[name] = (np.asarray(Q_all), np.asarray(C_all))

df_w2v = pickle.load(open("emb_w2v_ft.pkl","rb"))
embeddings["WORD2VEC"] = (
    np.vstack(df_w2v["WORD2VEC_EMB"].tolist()),
    np.vstack(df_w2v["WORD2VEC_C_EMB"].tolist())
)
embeddings["FASTTEXT"] = (
    np.vstack(df_w2v["FASTTEXT_Q"].tolist()),
    np.vstack(df_w2v["FASTTEXT_C"].tolist())
)

# align to smallest embedding length
min_len = min(Q.shape[0] for Q,_ in embeddings.values())
df     = df.iloc[:min_len].reset_index(drop=True)
all_ctx = df["CONTEXTS"].tolist()
gold    = df["LONG_ANSWER"].tolist()

# ─── 3) REPRO SAMPLE ──────────────────────────────────────────────────────────
random.seed(42)
sample_idxs = random.sample(range(min_len), SAMPLE_SIZE)

# ─── 4) DEFINE EXPERIMENTS ────────────────────────────────────────────────────
experiments = { name: [(name,1.0)] for name in embeddings }
for A,B in [("INSTR","E5"),("INSTR","SPECTER"),("INSTR","JINA"),
            ("E5","MPNET"),("MPNET","LABSE"),("MINILM6","MINILM12")]:
    experiments[f"{A}+{B}"] = [(A,0.7),(B,0.3)]
for base in ["INSTR","E5","SPECTER","JINA","MPNET","LABSE"]:
    experiments[f"W2V×0.3+{base}×0.7"] = [("WORD2VEC",0.3),(base,0.7)]
    experiments[f"FT×0.3+{base}×0.7"]  = [("FASTTEXT",0.3),(base,0.7)]

# ─── 5) SETUP CLIENTS ─────────────────────────────────────────────────────────
qdrant = QdrantClient(url="http://localhost:6333")
openai  = OpenAI()

# ─── 6) RUN & COLLECT EXAMPLES ────────────────────────────────────────────────
examples = []

for exp_name, parts in experiments.items():
    # pick the *first* sample in the list
    idx0 = sample_idxs[0]
    question = df.at[idx0, "QUESTION"]

    # ensemble question vector
    Q_ens = np.sum([embeddings[m][0][idx0]*w for m,w in parts], axis=0).tolist()

    # retrieve + aggregate
    scores = {}
    for m,w in parts:
        hits = qdrant.search(collection_name=m, query_vector=Q_ens, limit=TOP_K)
        for h in hits:
            i = h.payload["idx"]
            scores[i] = scores.get(i, 0.0) + w*h.score

    topk = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
    # build context string
    ctxs = []
    for i in topk:
        c = all_ctx[i]
        ctxs.append("\n\n".join(c) if isinstance(c, list) else c)

    prompt = (
        "Use the following context to answer succinctly.\n\n"
        "Context:\n" + "\n\n".join(ctxs) +
        f"\n\nQuestion: {question}\nAnswer:"
    )

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=TEMP,
        max_tokens=MAX_TOKENS
    )
    generated = resp.choices[0].message.content.strip()
    groundtruth = gold[idx0]

    examples.append({
        "model": exp_name,
        "question": question,
        "generated_answer": generated,
        "ground_truth": groundtruth
    })

    # also print to console
    print(f"\n─── {exp_name} ───")
    print("Q :", question)
    print("A :", generated)
    print("GT:", groundtruth)

# ─── 7) SAVE TO DISK ──────────────────────────────────────────────────────────
# as CSV
pd.DataFrame(examples).to_csv("examples.csv", index=False)
# as JSON
with open("examples.json","w") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print("\n→ Saved examples.csv and examples.json")
