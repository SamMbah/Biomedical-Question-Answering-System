# push_w2v_ft.py
from qdrant_client import QdrantClient
import pickle

# 1) Connect to local Qdrant
client = QdrantClient(url="http://localhost:6333")

# 2) Load your Word2Vec & FastText embeddings
df = pickle.load(open("emb_w2v_ft.pkl", "rb"))
datasets = {
    "w2v": (df["WORD2VEC_EMB"].tolist(), df["WORD2VEC_C_EMB"].tolist()),
    "ft":  (df["FASTTEXT_Q"].tolist(),    df["FASTTEXT_C"].tolist()),
}

# 3) Check existing collections
existing = {col.name for col in client.get_collections().collections}

# 4) Create collections if missing
for name in datasets:
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vector_size=768,
            distance="Cosine"
        )
        print(f"✔ Created collection `{name}`")

# 5) Upsert in batches of 500, using integer IDs
def upsert_in_batches(name, q_embs, c_embs, batch_size=500):
    total_q = len(q_embs)
    total   = total_q + len(c_embs)

    # questions: 0..total_q-1, contexts: total_q..total-1
    points = []
    for i, vec in enumerate(q_embs):
        points.append({"id": i, "vector": vec, "payload": {"type": "q"}})
    for i, vec in enumerate(c_embs, start=total_q):
        points.append({"id": i, "vector": vec, "payload": {"type": "c"}})

    # send in slices
    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch = points[start:end]
        client.upsert(collection_name=name, points=batch)
        print(f"  → Pushed `{name}` IDs {start:,}–{end-1:,}")

# 6) Push each dataset
for name, (q_embs, c_embs) in datasets.items():
    print(f"\nPushing `{name}` ({len(q_embs)} questions + {len(c_embs)} contexts)…")
    upsert_in_batches(name, q_embs, c_embs)
    print(f"✔ Done with `{name}`")

print("\nAll embeddings pushed successfully.")
