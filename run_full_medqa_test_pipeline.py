#!/usr/bin/env python3
"""
run_full_medqa_test_pipeline.py

1) Re-create the MedQA test split from ori_pqaa.json (80/20, seed=42).
2) Generate test‐question embeddings for:
     - Instructor (SBERT)
     - Jina      (SBERT)
     - E5        (SBERT)
     - SPECTER   (SBERT)
     - MPNET     (SBERT)
     - MiniLM6   (SBERT)
     - MiniLM12  (SBERT)
     - LaBSE     (SBERT)
     - Word2Vec  (gensim Word2Vec trained on test questions)
     - FastText  (gensim FastText trained on test questions)
3) Ingest all _training_ context embeddings into Qdrant (collections named by model).
4) Run retrieval (Top‐K=3) + GPT‐3.5 on test questions for every single model & ensemble,
   compute mean cosine, mean Euclid, ROUGE‐1/2/L P/R/F.
5) Save results:
     - testqa.pkl
     - test_emb_<model>.pkl  (one per model, containing (Q_test, C_train))
     - eval_test_results.csv (metrics table)
     - examples_test.csv      (one Q/A/GT example per model)
     - examples_test.json
"""

import os
import sys
import pickle
import random
import subprocess
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import train_test_split
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI
from rouge_score import rouge_scorer

# ─── 0) BASIC CONFIG ───────────────────────────────────────────────────────────
SEED       = 42
TEST_SIZE  = 0.2
TOP_K      = 3
TEMP       = 0.0
MAX_TOKENS = 256

OUT_DIR    = "test_emb"
os.makedirs(OUT_DIR, exist_ok=True)

# Make sure your OpenAI key is set
if "OPENAI_API_KEY" not in os.environ:
    print(" ERROR: Please set the OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

# ─── 1) RECOVER & SAVE MedQA TEST SPLIT ─────────────────────────────────────────
print("→ Step 1: Recovering the MedQA test split…")
df_full = pd.read_json("ori_pqaa.json", orient="index")

# 1a) Drop duplicates by QUESTION (as originally done)
df_full = df_full.drop_duplicates(subset="QUESTION").reset_index(drop=True)

# 1b) Flatten contexts (take first element if it's a nonempty list)
df_full["CONTEXTS"] = df_full["CONTEXTS"].apply(lambda x: x[0] if isinstance(x, list) and x else x)

# 1c) Keep only final_decision == "yes" if that was applied for train
if "final_decision" in df_full.columns:
    df_full = df_full[df_full["final_decision"] == "yes"].reset_index(drop=True)

# 1d) Split into train/test
train_df, test_df = train_test_split(
    df_full, test_size=TEST_SIZE, random_state=SEED
)
print(f"   • Total rows: {len(df_full)} → train: {len(train_df)}, test: {len(test_df)}")

# 1e) Save test split to testqa.pkl
test_df = test_df.reset_index(drop=True)
with open("testqa.pkl", "wb") as f:
    pickle.dump(test_df, f)
print("    Saved test split → testqa.pkl\n")

# ─── 2) LOAD TRAIN CONTEXT EMBEDDINGS & INGEST INTO QDRANT ────────────────────
print("→ Step 2: Loading training context embeddings and ingesting into Qdrant…")

# 2a) Restart Qdrant
print("   • Restarting Qdrant container…")
subprocess.run(["docker", "rm", "-f", "qdrant"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
subprocess.run(["docker", "run", "-d", "--name", "qdrant", "-p", "6333:6333", "qdrant/qdrant:latest"])

# 2b) Connect to Qdrant
qdrant = QdrantClient(url="http://localhost:6333")

# 2c) SBERT‐style train pickles (already computed)
SBERT_PKLS = [
    "emb_instr.pkl",
    "emb_jina.pkl",
    "emb_e5.pkl",
    "emb_specter.pkl",
    "emb_mpnet.pkl",
    "emb_minilm6.pkl",
    "emb_minilm12.pkl",
    "emb_labse.pkl",
]
W2VFT_PKL  = "emb_w2v_ft.pkl"

# We'll keep context‐embeddings in a dict for later use
context_embeddings = {}

# 2d) Ingest each SBERT context pool into Qdrant under its model name
for fn in SBERT_PKLS:
    model_name = fn.replace("emb_","").replace(".pkl","").upper()
    print(f"   • Loading {fn} → collection `{model_name}`")
    Q_train, C_train = pickle.load(open(fn, "rb"))
    C_arr = np.asarray(C_train, dtype=np.float32)
    context_embeddings[model_name] = C_arr

    # (Re)create the collection with cosine distance
    qdrant.recreate_collection(
        collection_name=model_name,
        vectors_config=qm.VectorParams(size=C_arr.shape[1], distance=qm.Distance.COSINE),
    )
    # Upsert in batches
    for offset in range(0, len(C_arr), 1024):
        chunk = C_arr[offset: offset+1024]
        points = [
            qm.PointStruct(id=idx, vector=vec.tolist(), payload={"idx": idx})
            for idx, vec in enumerate(chunk, start=offset)
        ]
        qdrant.upsert(collection_name=model_name, points=points)
    print(f"      Done `{model_name}` ({C_arr.shape[0]} vectors)")

# 2e) Load W2V+FT DF and ingest WORD2VEC & FASTTEXT context pools
print(f"   • Loading {W2VFT_PKL} → collections WORD2VEC & FASTTEXT")
df_w2v = pickle.load(open(W2VFT_PKL, "rb"))

Cw_arr = np.vstack(df_w2v["WORD2VEC_C_EMB"].tolist()).astype(np.float32)
context_embeddings["WORD2VEC"] = Cw_arr
qdrant.recreate_collection(
    collection_name="WORD2VEC",
    vectors_config=qm.VectorParams(size=Cw_arr.shape[1], distance=qm.Distance.COSINE),
)
for offset in range(0, len(Cw_arr), 1024):
    chunk = Cw_arr[offset: offset+1024]
    points = [
        qm.PointStruct(id=idx, vector=vec.tolist(), payload={"idx": idx})
        for idx, vec in enumerate(chunk, start=offset)
    ]
    qdrant.upsert(collection_name="WORD2VEC", points=points)
print(f"      Done `WORD2VEC` ({Cw_arr.shape[0]} vectors)")

Cf_arr = np.vstack(df_w2v["FASTTEXT_C"].tolist()).astype(np.float32)
context_embeddings["FASTTEXT"] = Cf_arr
qdrant.recreate_collection(
    collection_name="FASTTEXT",
    vectors_config=qm.VectorParams(size=Cf_arr.shape[1], distance=qm.Distance.COSINE),
)
for offset in range(0, len(Cf_arr), 1024):
    chunk = Cf_arr[offset: offset+1024]
    points = [
        qm.PointStruct(id=idx, vector=vec.tolist(), payload={"idx": idx})
        for idx, vec in enumerate(chunk, start=offset)
    ]
    qdrant.upsert(collection_name="FASTTEXT", points=points)
print(f"      Done `FASTTEXT` ({Cf_arr.shape[0]} vectors)\n")

# ─── 3) LOAD TEST DATA & PREPARE TEXT LISTS ───────────────────────────────────
print("→ Step 3: Loading test questions for embedding…")
test_df = pickle.load(open("testqa.pkl", "rb"))
test_df = test_df.reset_index(drop=True)
questions = test_df["QUESTION"].tolist()
print(f"   • Number of test questions: {len(questions)}\n")

# ─── 4) INSTANTIATE & RUN QUESTION EMBEDDERS ─────────────────────────────────
print("→ Step 4: Instantiating SBERT models & generating test embeddings…")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4a) Instructor (hkunlp/instructor-large)
instr_model = SentenceTransformer("hkunlp/instructor-large", device=DEVICE)

# 4b) Jina (jinaai/jina-embeddings-v2-base-en)
jina_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", device=DEVICE)

# 4c) E5 (intfloat/e5-large)
e5_model = SentenceTransformer("intfloat/e5-large", device=DEVICE)

# 4d) SPECTER (allenai-specter)
specter_model = SentenceTransformer("allenai-specter", device=DEVICE)

# 4e) MPNET (all-mpnet-base-v2)
mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)

# 4f) MiniLM6 (all-MiniLM-L6-v2) & MiniLM12 (all-MiniLM-L12-v2)
minilm6_model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
minilm12_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=DEVICE)

# 4g) LaBSE (sentence-transformers/LaBSE)
labse_model = SentenceTransformer("sentence-transformers/LaBSE", device=DEVICE)

# 4h) Train toy Word2Vec & FastText on the test questions (vector_size=768)
print("   • Training Word2Vec & FastText on test questions…")
tokenized_q = [q.lower().split() for q in questions]
w2v_q = Word2Vec(sentences=tokenized_q, vector_size=768, window=2, min_count=1, workers=4)
ft_q  = FastText(sentences=tokenized_q, vector_size=768, window=2, min_count=1, workers=4)

# 4i) Helper to save (Q_test, C_train)
def save_test_embedding(name, Q_test, C_train):
    out_path = os.path.join(OUT_DIR, f"test_emb_{name.lower()}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((Q_test.astype(np.float32), C_train.astype(np.float32)), f)
    print(f"    Saved test_emb_{name.lower()}.pkl (Q_test: {Q_test.shape}, C_train: {C_train.shape})")

print("\n→ Generating all test‐question embeddings…")
# (i) Instructor Q_test
print("   • Instructor Q_test (batch=16)…")
instr_inputs   = [[ "Represent the question for retrieval:", q ] for q in questions]
instr_q_test   = instr_model.encode(instr_inputs, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
save_test_embedding("instructor", instr_q_test, context_embeddings["INSTRUCTOR"])

# (ii) Jina Q_test
print("   • Jina Q_test (batch=32)…")
jina_q_test = jina_model.encode(questions, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
save_test_embedding("jina", jina_q_test, context_embeddings["JINA"])

# (iii) E5 Q_test
print("   • E5 Q_test (batch=32)…")
e5_q_test = e5_model.encode(questions, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
save_test_embedding("e5", e5_q_test, context_embeddings["E5"])

# (iv) SPECTER Q_test
print("   • SPECTER Q_test (batch=32)…")
specter_q_test = specter_model.encode(questions, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
save_test_embedding("specter", specter_q_test, context_embeddings["SPECTER"])

# (v) MPNET Q_test
print("   • MPNET Q_test (batch=32)…")
mpnet_q_test = mpnet_model.encode(questions, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
save_test_embedding("mpnet", mpnet_q_test, context_embeddings["MPNET"])

# (vi) MiniLM6 Q_test
print("   • MiniLM6 Q_test (batch=64)…")
minilm6_q_test = minilm6_model.encode(questions, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
save_test_embedding("minilm6", minilm6_q_test, context_embeddings["MINILM6"])

# (vii) MiniLM12 Q_test
print("   • MiniLM12 Q_test (batch=32)…")
minilm12_q_test = minilm12_model.encode(questions, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
save_test_embedding("minilm12", minilm12_q_test, context_embeddings["MINILM12"])

# (viii) LaBSE Q_test
print("   • LaBSE Q_test (batch=32)…")
labse_q_test = labse_model.encode(questions, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
save_test_embedding("labse", labse_q_test, context_embeddings["LABSE"])

# (ix) Word2Vec Q_test
print("   • Word2Vec Q_test…")
Qw_list = []
for q in questions:
    toks = q.lower().split()
    vecs = [w2v_q.wv[t] for t in toks if t in w2v_q.wv]
    Qw_list.append(np.mean(vecs, axis=0) if len(vecs)>0 else np.zeros(768))
Qw_test = np.vstack(Qw_list).astype(np.float32)
save_test_embedding("word2vec", Qw_test, context_embeddings["WORD2VEC"])

# (x) FastText Q_test
print("   • FastText Q_test…")
Qf_list = []
for q in questions:
    toks = q.lower().split()
    vecs = [ft_q.wv[t] for t in toks if t in ft_q.wv]
    Qf_list.append(np.mean(vecs, axis=0) if len(vecs)>0 else np.zeros(768))
Qf_test = np.vstack(Qf_list).astype(np.float32)
save_test_embedding("fasttext", Qf_test, context_embeddings["FASTTEXT"])

print("\n All test embeddings generated and saved under ‘test_emb/’\n")

# ─── 5) EVALUATE ON TEST SET (Retrieval+GPT-3.5+ROUGE) ──────────────────────────
print("→ Step 5: Running retrieval + GPT-3.5 + ROUGE on the test set…")

# 5a) Re‐load all test embeddings into memory
embeddings = {}
for fn in sorted(os.listdir(OUT_DIR)):
    if fn.startswith("test_emb_") and fn.endswith(".pkl"):
        name = fn.replace("test_emb_","").replace(".pkl","").upper()
        Q_test, C_train = pickle.load(open(os.path.join(OUT_DIR, fn), "rb"))
        embeddings[name] = (np.asarray(Q_test, dtype=np.float32),
                            np.asarray(C_train, dtype=np.float32))
print("   • Loaded test embeddings for:", list(embeddings.keys()))

# 5b) Define experiments exactly as on train:
experiments = { name: [(name,1.0)] for name in embeddings.keys() }

sb_pairs = [
    ("INSTRUCTOR","E5"),
    ("INSTRUCTOR","SPECTER"),
    ("INSTRUCTOR","JINA"),
    ("INSTRUCTOR","MPNET"),
    ("INSTRUCTOR","LABSE"),
    ("E5","SPECTER"),
    ("E5","MPNET"),
    ("E5","LABSE"),
    ("SPECTER","MPNET"),
    ("SPECTER","LABSE"),
    ("JINA","MPNET"),
    ("JINA","LABSE"),
    ("MPNET","LABSE"),
    ("MINILM6","MINILM12"),
]
for A,B in sb_pairs:
    experiments[f"{A}+{B}"] = [(A,0.7),(B,0.3)]

for base in ["INSTRUCTOR","E5","SPECTER","JINA","MPNET","LABSE"]:
    experiments[f"W2V×0.3+{base}×0.7"] = [("WORD2VEC",0.3),(base,0.7)]
    experiments[f"FT×0.3+{base}×0.7"]  = [("FASTTEXT",0.3),(base,0.7)]

print("   • Experiments to run (count={}):".format(len(experiments)))
for name in sorted(experiments.keys()):
    print("     •", name)
print()

# 5c) Setup ROUGE & OpenAI
metrics = ["rouge1","rouge2","rougeL"]
scorer  = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
openai  = OpenAI()

# 5d) Prepare test contexts & references
all_contexts = test_df["CONTEXTS"].tolist()
refs         = test_df["LONG_ANSWER"].tolist()

records  = []
examples = []

# 5e) Main evaluation loop
for exp_name, parts in experiments.items():
    print(f"\n   → Running experiment `{exp_name}`")
    hyps       = []
    cos_sims   = []
    euc_dists  = []
    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []
    rl_p, rl_r, rl_f = [], [], []

    ex_q, ex_gen, ex_gt = None, None, None

    for i, row in test_df.iterrows():
        # 1) Build ensembled question embedding
        idx0 = i
        Q_parts = [embeddings[m][0][idx0] * w for m,w in parts]
        Q_ens   = np.sum(Q_parts, axis=0)

        # 2) Retrieve top‐K across each part, weighted sum of scores
        score_acc = {}
        for m,w in parts:
            resp = qdrant.search(
                collection_name=m,
                query_vector=Q_ens.tolist(),
                limit=TOP_K
            )
            for h in resp:
                j = h.payload["idx"]
                score_acc[j] = score_acc.get(j, 0.0) + w * h.score

        # 3) Pick top‐K context indices
        topk = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        chosen_idxs = [j for j,_ in topk]

        # 4) Compute retrieval metrics
        for j in chosen_idxs:
            C_parts = [embeddings[m][1][j] * w for m,w in parts]
            C_ens   = np.sum(C_parts, axis=0)
            cos_sims.append(
                util.cos_sim(torch.from_numpy(Q_ens),
                             torch.from_numpy(C_ens)).item()
            )
            euc_dists.append(distance.euclidean(Q_ens, C_ens))

        # 5) Build prompt
        ctxs = []
        for j in chosen_idxs:
            cval = all_contexts[j]
            if isinstance(cval, list):
                ctxs.append("\n\n".join(cval))
            else:
                ctxs.append(cval)

        prompt = (
            "Use the following context to answer succinctly.\n\n"
            "Context:\n" + "\n\n".join(ctxs) +
            f"\n\nQuestion: {row['QUESTION']}\nAnswer:"
        )

        # 6) Call GPT-3.5
        chat = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=TEMP,
            max_tokens=MAX_TOKENS
        )
        answer = chat.choices[0].message.content.strip()
        hyps.append(answer)

        # 7) Compute ROUGE for this pair
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

        # 8) Capture first example for this experiment
        if i == 0:
            ex_q  = row["QUESTION"]
            ex_gen = answer
            ex_gt  = row["LONG_ANSWER"]

    # 9) Aggregate metrics (simple means)
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

    # 10) Store that example for this model
    examples.append({
        "model": exp_name,
        "question": ex_q,
        "generated_answer": ex_gen,
        "ground_truth": ex_gt
    })

# ─── 6) SAVE FINAL RESULTS ─────────────────────────────────────────────────────
print("\n→ Step 6: Saving final results…")
df_metrics = pd.DataFrame(records).sort_values("mean_cosine", ascending=False)
print(df_metrics.to_markdown(index=False))

df_metrics.to_csv("eval_test_results.csv", index=False)
print("    Saved evaluation metrics → eval_test_results.csv")

pd.DataFrame(examples).to_csv("examples_test.csv", index=False)
with open("examples_test.json","w") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)
print("   Saved example Q/A/GT → examples_test.csv + examples_test.json\n")

print(" All done! Test‐set pipeline complete.")
