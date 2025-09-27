#!/usr/bin/env python3
import os
import pickle
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# 1) Connect to your local Qdrant
client = QdrantClient(url="http://localhost:6333")

# 2) SBERT pickles you already have
SBERT_FILES = [
    "emb_instr.pkl", "emb_jina.pkl", "emb_e5.pkl", "emb_specter.pkl",
    "emb_mpnet.pkl","emb_minilm6.pkl","emb_minilm12.pkl","emb_labse.pkl",
]

# 3) Handle SBERT collections
for fn in SBERT_FILES:
    name = fn.replace("emb_","").replace(".pkl","").upper()
    print(f"Ingesting SBERT → collection `{name}`")
    Q_all, C_all = pickle.load(open(fn, "rb"))
    dim = C_all.shape[1]

    # (re)create collection
    client.recreate_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    # upsert in batches
    for offset in range(0, len(C_all), 1024):
        chunk = C_all[offset:offset+1024]
        points = [
            qm.PointStruct(id=idx, vector=vec.tolist(), payload={"idx": idx})
            for idx, vec in enumerate(chunk, start=offset)
        ]
        client.upsert(collection_name=name, points=points)
    print(f" → Done {name} ({len(C_all)} vectors)\n")

# 4) Handle Word2Vec + FastText DataFrame
print("Loading emb_w2v_ft.pkl …")
df_w2v = pickle.load(open("emb_w2v_ft.pkl", "rb"))

# WORD2VEC
print("Ingesting WORD2VEC → collection `WORD2VEC`")
Cw = np.vstack(df_w2v["WORD2VEC_C_EMB"].tolist()).astype(np.float32)
dim = Cw.shape[1]
client.recreate_collection(
    collection_name="WORD2VEC",
    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
)
for offset in range(0, len(Cw), 1024):
    chunk = Cw[offset:offset+1024]
    points = [
        qm.PointStruct(id=idx, vector=vec.tolist(), payload={"idx": idx})
        for idx, vec in enumerate(chunk, start=offset)
    ]
    client.upsert(collection_name="WORD2VEC", points=points)
print(f" → Done WORD2VEC ({len(Cw)} vectors)\n")

# FASTTEXT
print("Ingesting FASTTEXT → collection `FASTTEXT`")
Cf = np.vstack(df_w2v["FASTTEXT_C"].tolist()).astype(np.float32)
dim = Cf.shape[1]
client.recreate_collection(
    collection_name="FASTTEXT",
    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
)
for offset in range(0, len(Cf), 1024):
    chunk = Cf[offset:offset+1024]
    points = [
        qm.PointStruct(id=idx, vector=vec.tolist(), payload={"idx": idx})
        for idx, vec in enumerate(chunk, start=offset)
    ]
    client.upsert(collection_name="FASTTEXT", points=points)
print(f" → Done FASTTEXT ({len(Cf)} vectors)\n")
