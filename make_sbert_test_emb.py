#!/usr/bin/env python3
"""
make_sbert_test_emb.py

Generate test‐question embeddings for all SBERT‐style models in one script,
**ensuring we use the same model IDs (and thus vector dims) as on the training side**.

Models:
  - INSTR    (hkunlp/instructor-large)           → 768-dim
  - JINA     (jinaai/jina-embeddings-v2-base-en)  → 768-dim
  - E5       (intfloat/e5-base)                   → 768-dim
  - SPECTER  (allenai-specter)                    → 768-dim
  - MPNET    (all-mpnet-base-v2)                  → 768-dim
  - MINILM6  (all-MiniLM-L6-v2)                   → 384-dim
  - MINILM12 (all-MiniLM-L12-v2)                  → 384-dim
  - LABSE    (sentence-transformers/LaBSE)        → 768-dim

After running this, you should have:
  test_emb_instr.pkl
  test_emb_jina.pkl
  test_emb_e5.pkl
  test_emb_specter.pkl
  test_emb_mpnet.pkl
  test_emb_minilm6.pkl
  test_emb_minilm12.pkl
  test_emb_labse.pkl
"""

import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ─── 1) Load test questions ────────────────────────────────────────────────────
test_df = pickle.load(open("testqa.pkl", "rb"))
questions = test_df["QUESTION"].tolist()
n_test = len(questions)
print(f"→ Loaded testqa.pkl with {n_test} questions\n")

# ─── 2) Define SBERT specs (name, train‐pickle, model_id, needs_prompt) ─────────
MODEL_SPECS = [
    ("instr",   "emb_instr.pkl",    "hkunlp/instructor-large",         True),
    ("jina",    "emb_jina.pkl",     "jinaai/jina-embeddings-v2-base-en", False),
    ("e5",      "emb_e5.pkl",       "intfloat/e5-base",                False),  # <— changed to e5-base (768 dim)
    ("specter", "emb_specter.pkl",  "allenai-specter",                 False),
    ("mpnet",   "emb_mpnet.pkl",    "sentence-transformers/all-mpnet-base-v2", True),  # no prompt for mpnet
    ("minilm6", "emb_minilm6.pkl",  "sentence-transformers/all-MiniLM-L6-v2",    False),
    ("minilm12","emb_minilm12.pkl", "sentence-transformers/all-MiniLM-L12-v2",   False),
    ("labse",   "emb_labse.pkl",    "sentence-transformers/LaBSE",       False),
]

# ─── 3) Determine device ────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"→ Using device: {DEVICE}\n")

# ─── 4) Helper: save (Q_test, C_train) ───────────────────────────────────────────
def save_test_embedding(name, Q_test, C_train):
    out_fname = f"test_emb_{name}.pkl"
    with open(out_fname, "wb") as f:
        pickle.dump((Q_test.astype(np.float32), C_train.astype(np.float32)), f)
    print(f"    Saved {out_fname}: Q_test = {Q_test.shape}, C_train = {C_train.shape}")

# ─── 5) Iterate and generate ──────────────────────────────────────────────────
for short_name, train_pkl, model_id, needs_prompt in MODEL_SPECS:
    print(f"→ Processing `{short_name}` (train pickle: {train_pkl}, SBERT ID: {model_id})")

    # 5a) Load C_train
    if not os.path.isfile(train_pkl):
        print(f"   {train_pkl} not found. Skipping `{short_name}`.\n")
        continue

    Q_train_dummy, C_train = pickle.load(open(train_pkl, "rb"))
    C_train = np.asarray(C_train, dtype=np.float32)
    print(f"   • Loaded C_train for `{short_name}`, shape = {C_train.shape}")

    # 5b) Load SBERT model
    try:
        model = SentenceTransformer(model_id, device=DEVICE)
    except Exception as e:
        print(f"   ERROR loading `{model_id}`: {e}\n")
        continue

    # 5c) Build inputs
    if needs_prompt:
        # Instructor requires ["Represent the question for retrieval:", q]
        inputs = [[ "Represent the question for retrieval:", q ] for q in questions]
    else:
        inputs = questions

    # 5d) Encode test questions
    batch_size = 16 if needs_prompt else 32
    print(f"   • Encoding {n_test} test questions via `{short_name}` (batch={batch_size}) …")
    Q_test = model.encode(
        inputs,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True
    )
    Q_test = np.asarray(Q_test, dtype=np.float32)
    print(f"   • Produced Q_test shape = {Q_test.shape}")

    # 5e) Save
    save_test_embedding(short_name, Q_test, C_train)
    print()

print("All SBERT test embeddings generated.\n")
