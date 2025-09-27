#!/usr/bin/env python3
"""
ingest_all_embeddings.py

This script recreates the following Qdrant collections:

  • INSTR, JINA, E5, SPECTER, MPNET, MINILM6, MINILM12, LABSE
  • WORD2VEC, FASTTEXT

  • TEST_INSTR, TEST_JINA, TEST_E5, TEST_SPECTER, TEST_MPNET, TEST_MINILM6, TEST_MINILM12, TEST_LABSE
  • TEST_WORD2VEC, TEST_FASTTEXT

For each “<MODEL>” above:

  1) The collection <MODEL> will hold **168,921** training‐context embeddings (C_train).
  2) The collection TEST_<MODEL> will hold **39,209** test‐question embeddings (Q_test).

NOTE:
  - Make sure you have all of these files in your current working directory (~/medqa-wsl):
      • emb_instr.pkl, emb_jina.pkl, emb_e5.pkl, emb_specter.pkl,
        emb_mpnet.pkl, emb_minilm6.pkl, emb_minilm12.pkl, emb_labse.pkl
      • emb_w2v_ft.pkl
      • test_emb_instr.pkl, test_emb_jina.pkl, test_emb_e5.pkl, test_emb_specter.pkl,
        test_emb_mpnet.pkl, test_emb_minilm6.pkl, test_emb_minilm12.pkl, test_emb_labse.pkl
      • test_emb_word2vec.pkl, test_emb_fasttext.pkl

  - This script assumes that each SBERT “emb_<model>.pkl” was saved as a tuple:
        (Q_train, C_train)
    where Q_train ∈ numpy.ndarray (shape = (168921, D)) and
          C_train ∈ numpy.ndarray (shape = (168921, D)).

  - For Word2Vec/FT, “emb_w2v_ft.pkl” is assumed to be a pandas.DataFrame with columns
      ["WORD2VEC_Q_EMB", "WORD2VEC_C_EMB", "FASTTEXT_Q", "FASTTEXT_C"].

  - Each “test_emb_<model>.pkl” is assumed to be a tuple:
        (Q_test, C_train)
    where Q_test ∈ numpy.ndarray (shape = (39209, D)) and
          C_train ∈ numpy.ndarray (shape = (168921, D))
    (The training‐context embeddings can be re‐used from emb_<model>.pkl, but here we load them
     from the “test” pickle as well.)

USAGE:
    cd ~/medqa-wsl
    python3 ingest_all_embeddings.py

"""

import os
import pickle
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ──────────────────────────────────────────────────────────────────────────────
# 1) Initialize Qdrant client (assumes local Qdrant at http://localhost:6333)
# ──────────────────────────────────────────────────────────────────────────────
client = QdrantClient(url="http://localhost:6333")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Define SBERT models (the pickles for train & test)
# ──────────────────────────────────────────────────────────────────────────────
SBERT_MODELS = [
    ("INSTR",   "emb_instr.pkl",    "test_emb_instr.pkl"),
    ("JINA",    "emb_jina.pkl",     "test_emb_jina.pkl"),
    ("E5",      "emb_e5.pkl",       "test_emb_e5.pkl"),
    ("SPECTER", "emb_specter.pkl",  "test_emb_specter.pkl"),
    ("MPNET",   "emb_mpnet.pkl",    "test_emb_mpnet.pkl"),
    ("MINILM6", "emb_minilm6.pkl",  "test_emb_minilm6.pkl"),
    ("MINILM12","emb_minilm12.pkl", "test_emb_minilm12.pkl"),
    ("LABSE",   "emb_labse.pkl",    "test_emb_labse.pkl"),
]

# ──────────────────────────────────────────────────────────────────────────────
# 3) Ingest SBERT context (C_train) into “<MODEL>” and test questions (Q_test) into “TEST_<MODEL>”
# ──────────────────────────────────────────────────────────────────────────────
for (MODEL, train_pickle, test_pickle) in SBERT_MODELS:
    # Load the “train” pickle: this should be a tuple (Q_train, C_train)
    print(f"Ingesting training‐context embeddings for “{MODEL}” → collection `{MODEL}`")
    with open(train_pickle, "rb") as f:
        Q_train, C_train = pickle.load(f)

    # Q_train ∈ np.ndarray of shape (168921, D)  [unused for ingestion]
    # C_train ∈ np.ndarray of shape (168921, D)

    # Determine dimension from C_train
    dim = C_train.shape[1]

    # Recreate (drop + create) the train collection
    client.recreate_collection(
        collection_name=MODEL,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    # Bulk‐upsert C_train in chunks of 1024
    for offset in range(0, C_train.shape[0], 1024):
        chunk = C_train[offset : offset + 1024]
        points = [
            qm.PointStruct(
                id=int(idx), 
                vector=vec.astype(np.float32).tolist(),
                payload={"idx": int(idx)}
            )
            for idx, vec in enumerate(chunk, start=offset)
        ]
        client.upsert(collection_name=MODEL, points=points)

    print(f" → Done `{MODEL}` ({C_train.shape[0]} vectors total)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Now ingest the “test” pickle: (Q_test, C_train) for TEST_<MODEL>
    # ──────────────────────────────────────────────────────────────────────────
    print(f"Ingesting TEST‐question embeddings for “{MODEL}” → collection `TEST_{MODEL}`")
    with open(test_pickle, "rb") as f:
        Q_test, _C_train_dummy = pickle.load(f)

    # Q_test ∈ np.ndarray of shape (39209, D)
    dim_t = Q_test.shape[1]  # should match dim, but we’ll re‐check

    # Recreate (drop + create) the test collection
    client.recreate_collection(
        collection_name=f"TEST_{MODEL}",
        vectors_config=qm.VectorParams(size=dim_t, distance=qm.Distance.COSINE),
    )

    # Bulk‐upsert Q_test in chunks of 1024
    for offset in range(0, Q_test.shape[0], 1024):
        chunk = Q_test[offset : offset + 1024]
        points = [
            qm.PointStruct(
                id=int(idx), 
                vector=vec.astype(np.float32).tolist(),
                payload={"idx": int(idx)}
            )
            for idx, vec in enumerate(chunk, start=offset)
        ]
        client.upsert(collection_name=f"TEST_{MODEL}", points=points)

    print(f" → Done `TEST_{MODEL}` ({Q_test.shape[0]} vectors total)\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4) Handle WORD2VEC + FASTTEXT (train & test)
#     – “emb_w2v_ft.pkl” is a DataFrame with columns:
#         ['WORD2VEC_EMB', 'WORD2VEC_C_EMB', 'FASTTEXT_Q', 'FASTTEXT_C']
#     – “test_emb_word2vec.pkl” is a tuple (Q_test_word2vec, C_train_word2vec)
#     – “test_emb_fasttext.pkl” is (Q_test_fasttext, C_train_fasttext)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading emb_w2v_ft.pkl …")
df_w2v = pickle.load(open("emb_w2v_ft.pkl", "rb"))

# ──────────────────────────────────────────────────────────────────────────────
# 4A) TRAIN WORD2VEC
# ──────────────────────────────────────────────────────────────────────────────
print("Ingesting WORD2VEC training‐context → collection `WORD2VEC`")
Cw = np.vstack(df_w2v["WORD2VEC_C_EMB"].tolist()).astype(np.float32)
dim_w = Cw.shape[1]
client.recreate_collection(
    collection_name="WORD2VEC",
    vectors_config=qm.VectorParams(size=dim_w, distance=qm.Distance.COSINE),
)
for offset in range(0, Cw.shape[0], 1024):
    chunk = Cw[offset : offset + 1024]
    points = [
        qm.PointStruct(id=int(idx), vector=vec.tolist(), payload={"idx": int(idx)})
        for idx, vec in enumerate(chunk, start=offset)
    ]
    client.upsert(collection_name="WORD2VEC", points=points)
print(f" → Done `WORD2VEC` ({Cw.shape[0]} vectors total)\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4B) TEST WORD2VEC
# ──────────────────────────────────────────────────────────────────────────────
print("Ingesting WORD2VEC TEST‐questions → collection `TEST_WORD2VEC`")
Qw_test, _ctw_dummy = pickle.load(open("test_emb_word2vec.pkl", "rb"))
dim_wt = Qw_test.shape[1]
client.recreate_collection(
    collection_name="TEST_WORD2VEC",
    vectors_config=qm.VectorParams(size=dim_wt, distance=qm.Distance.COSINE),
)
for offset in range(0, Qw_test.shape[0], 1024):
    chunk = Qw_test[offset : offset + 1024]
    points = [
        qm.PointStruct(id=int(idx), vector=vec.tolist(), payload={"idx": int(idx)})
        for idx, vec in enumerate(chunk, start=offset)
    ]
    client.upsert(collection_name="TEST_WORD2VEC", points=points)
print(f" → Done `TEST_WORD2VEC` ({Qw_test.shape[0]} vectors total)\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4C) TRAIN FASTTEXT
# ──────────────────────────────────────────────────────────────────────────────
print("Ingesting FASTTEXT training‐context → collection `FASTTEXT`")
Cf = np.vstack(df_w2v["FASTTEXT_C"].tolist()).astype(np.float32)
dim_f = Cf.shape[1]
client.recreate_collection(
    collection_name="FASTTEXT",
    vectors_config=qm.VectorParams(size=dim_f, distance=qm.Distance.COSINE),
)
for offset in range(0, Cf.shape[0], 1024):
    chunk = Cf[offset : offset + 1024]
    points = [
        qm.PointStruct(id=int(idx), vector=vec.tolist(), payload={"idx": int(idx)})
        for idx, vec in enumerate(chunk, start=offset)
    ]
    client.upsert(collection_name="FASTTEXT", points=points)
print(f" → Done `FASTTEXT` ({Cf.shape[0]} vectors total)\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4D) TEST FASTTEXT
# ──────────────────────────────────────────────────────────────────────────────
print("Ingesting FASTTEXT TEST‐questions → collection `TEST_FASTTEXT`")
Qf_test, _ctf_dummy = pickle.load(open("test_emb_fasttext.pkl", "rb"))
dim_ft = Qf_test.shape[1]
client.recreate_collection(
    collection_name="TEST_FASTTEXT",
    vectors_config=qm.VectorParams(size=dim_ft, distance=qm.Distance.COSINE),
)
for offset in range(0, Qf_test.shape[0], 1024):
    chunk = Qf_test[offset : offset + 1024]
    points = [
        qm.PointStruct(id=int(idx), vector=vec.tolist(), payload={"idx": int(idx)})
        for idx, vec in enumerate(chunk, start=offset)
    ]
    client.upsert(collection_name="TEST_FASTTEXT", points=points)
print(f" → Done `TEST_FASTTEXT` ({Qf_test.shape[0]} vectors total)\n")


print(" All collections (train & test) have been (re)created successfully in Qdrant.\n")
