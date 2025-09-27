# load_and_push_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
import pickle
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
QDRANT_URL  = "127.0.0.1:6333"
TIMEOUT_SEC = 120
BATCH_SIZE  = 500

MODELS = {
    "instr":    "emb_instr.pkl",
    "jina":     "emb_jina.pkl",
    "e5":       "emb_e5.pkl",
    "specter":  "emb_specter.pkl",
    "mpnet":    "emb_mpnet.pkl",
    "minilm6":  "emb_minilm6.pkl",
    "minilm12": "emb_minilm12.pkl",
    "labse":    "emb_labse.pkl",
}

# ─── CONNECT ──────────────────────────────────────────────────────────────────
client = QdrantClient(
    url           = QDRANT_URL,
    prefer_grpc   = False,
    timeout       = TIMEOUT_SEC
)

for model_name, pickle_file in MODELS.items():
    # 1) load embeddings
    q_emb, c_emb = pickle.load(open(pickle_file, "rb"))
    dim = len(q_emb[0])
    total = len(q_emb)

    # 2) recreate collection
    if client.collection_exists(model_name):
        client.delete_collection(model_name)

    client.create_collection(
        collection_name = model_name,
        vectors_config  = VectorParams(size=dim, distance="Cosine"),
    )
    print(f"🗄  `{model_name}` collection ready (dim={dim}, N={total})")

    # 3) upsert in batches
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        points = []

        # question vectors: IDs = 0..total-1
        for i, vec in enumerate(q_emb[start:end], start):
            points.append(
                PointStruct(
                    id      = i,
                    vector  = vec.tolist(),
                    payload = {"model": model_name, "kind": "q", "idx": i}
                )
            )

        # context vectors: IDs = total..2*total-1
        for i, vec in enumerate(c_emb[start:end], start):
            points.append(
                PointStruct(
                    id      = total + i,
                    vector  = vec.tolist(),
                    payload = {"model": model_name, "kind": "c", "idx": i}
                )
            )

        client.upsert(
            collection_name = model_name,
            points          = points,
        )
        print(f"  • Pushed {model_name} points {start}–{end}")

        time.sleep(0.1)

    print(f"Done pushing `{model_name}`.\n")
