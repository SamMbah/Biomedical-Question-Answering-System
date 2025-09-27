#!/usr/bin/env python3
"""
prepare_qdrant_hotpot.py

Prepares context and question embeddings for four models (Instructor, BioBERT, E5, BERT-Large-Cased)
on the HotpotQA dataset. Saves Q_test and C_train embeddings as .npy under embeddings/<model>/.
Populates Qdrant with context embeddings for each model.

Usage:
    python prepare_qdrant_hotpot.py
"""

import os
import json
import pickle
import time
import numpy as np
import torch
import spacy
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import ResponseHandlingException

# ─── A) DOWNLOAD NLTK DATA & SETUP ─────────────────────────────────────────────
nltk.download("stopwords")
nltk.download("punkt")

# ─── B) CONFIGURATION ────────────────────────────────────────────────────────────

# 1) Paths
HOTPOT_JSON   = "hotpot_test_fullwiki_v1.json"
EMBEDDINGS_DIR = "embeddings"
QDRANT_URL     = "http://localhost:6333"

# 2) Models to process (name → (embedding_dim, type))
#    type = "sentence_transformer" or "huggingface"
MODELS = {
    "instructor":       (768,  "sentence_transformer", "hkunlp/instructor-large"),
    "biobert":          (768,  "huggingface",          "dmis-lab/biobert-v1.1"),
    "e5":               (1024, "sentence_transformer", "intfloat/e5-large"),
    "bert_large_cased": (1024, "huggingface",          "bert-large-cased")
}

# 3) Qdrant batch size for upserting vectors
QDRANT_BATCH_SIZE = 256

# ─── C) TEXT CLEANING FUNCTION ───────────────────────────────────────────────────

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")

# Prepare NLTK stopwords and stemmer
STOP = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")

def clean_text(raw: str) -> str:
    """
    1) Lowercase
    2) Tokenize with spaCy
    3) Remove any token that is not purely alphabetic
    4) Remove stopwords
    5) Stem with SnowballStemmer
    6) Re-join into a single space-delimited string
    """
    text = raw.lower()
    doc  = nlp(text)
    tokens = []
    for tok in doc:
        tok_text = tok.text.strip()
        # Keep only alphabetic tokens
        if not tok_text.isalpha():
            continue
        # Remove stopwords
        if tok_text in STOP:
            continue
        # Stem
        stemmed = STEMMER.stem(tok_text)
        tokens.append(stemmed)
    return " ".join(tokens)

# ─── D) LOAD HOTPOTQA JSON ───────────────────────────────────────────────────────

print("→ Loading HotpotQA JSON…")
with open(HOTPOT_JSON, "r", encoding="utf8") as f:
    hotpot_data = json.load(f)

num_examples = len(hotpot_data)
print(f"→ HotpotQA contains {num_examples} examples")

# Build raw question list and raw context list
raw_questions = []
raw_contexts  = []

for item in hotpot_data:
    # Question
    q = item.get("question", "") or ""
    raw_questions.append(q)

    # Flatten context paragraphs into one string
    paras = []
    for (_title, sentences) in item["context"]:
        paras.append(" ".join(sentences))
    raw_contexts.append("\n\n".join(paras))

# ─── E) CLEANED TEXT LISTS ───────────────────────────────────────────────────────

print("→ Cleaning all questions and contexts…")
cleaned_questions = [clean_text(q) for q in raw_questions]
cleaned_contexts  = [clean_text(c) for c in raw_contexts]

# ─── F) QDRANT CLIENT ────────────────────────────────────────────────────────────

print("→ Connecting to Qdrant…")
qdrant = QdrantClient(url=QDRANT_URL)

# ─── G) ENSURE embeddings/<model>/ FOLDERS EXIST ─────────────────────────────────

for model_name in MODELS:
    model_dir = os.path.join(EMBEDDINGS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

# ─── H) PROCESS EACH MODEL ──────────────────────────────────────────────────────

for model_name, (dim, model_type, hf_id) in MODELS.items():
    print(f"\n→ Processing model: {model_name} (dim={dim}, type={model_type})")

    # ─── H.1) LOAD OR INITIALIZE ENCODER ─────────────────────────────────────────
    if model_type == "sentence_transformer":
        encoder = SentenceTransformer(hf_id, device=("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        # HuggingFace model: load tokenizer + model, move to DEVICE
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        def hf_encode(texts: list[str]) -> np.ndarray:
            """
            Tokenizes `texts` (list of strings), runs through HuggingFace `model`,
            mean-pools last_hidden_state, returns np.ndarray of shape (len(texts), dim).
            """
            all_embs = []
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch, return_tensors="pt", truncation=True, padding=True, max_length=512
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    outputs = model(**inputs)
                pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embs.append(pooled)
            return np.vstack(all_embs).astype(np.float32)

    # ─── H.2) ENCODE CONTEXTS ─────────────────────────────────────────────────────
    print(f"  • Encoding all {num_examples} contexts…")
    if model_type == "sentence_transformer":
        C_embs = encoder.encode(cleaned_contexts, batch_size=32, show_progress_bar=True, normalize_embeddings=False)
    else:
        C_embs = hf_encode(cleaned_contexts)
    C_embs = np.asarray(C_embs, dtype=np.float32)
    print(f"    → C_embs.shape = {C_embs.shape}")
    np.save(os.path.join(EMBEDDINGS_DIR, model_name, "C_train.npy"), C_embs)

    # ─── H.3) ENCODE QUESTIONS ────────────────────────────────────────────────────
    print(f"  • Encoding all {num_examples} questions…")
    if model_type == "sentence_transformer":
        Q_embs = encoder.encode(cleaned_questions, batch_size=32, show_progress_bar=True, normalize_embeddings=False)
    else:
        Q_embs = hf_encode(cleaned_questions)
    Q_embs = np.asarray(Q_embs, dtype=np.float32)
    print(f"    → Q_embs.shape = {Q_embs.shape}")
    np.save(os.path.join(EMBEDDINGS_DIR, model_name, "Q_test.npy"), Q_embs)

    # ─── H.4) POPULATE QDRANT COLLECTION ─────────────────────────────────────────
    print(f"  • Preparing Qdrant collection: {model_name}…")
    # Delete existing collection if present
    try:
        qdrant.delete_collection(collection_name=model_name)
    except Exception:
        pass

    # Create new collection
    qdrant.recreate_collection(
        collection_name = model_name,
        vectors_config  = qdrant_models.VectorParams(
            size = dim,
            distance = qdrant_models.Distance.COSINE
        )
    )

    # Upsert context vectors in batches
    print(f"  • Upserting {num_examples} context vectors into Qdrant/{model_name}…")
    for start in range(0, num_examples, QDRANT_BATCH_SIZE):
        end = min(start + QDRANT_BATCH_SIZE, num_examples)
        batch_vectors = C_embs[start:end]
        points = []
        for idx, vec in enumerate(batch_vectors, start=start):
            points.append(
                qdrant_models.PointStruct(
                    id = idx,
                    vector = vec.tolist(),
                    payload = {"idx": idx}
                )
            )
        qdrant.upsert(
            collection_name = model_name,
            points = points
        )
        # Brief sleep to avoid overwhelming Qdrant
        time.sleep(0.1)

    print(f"  → Qdrant collection `{model_name}` ready (total points: {num_examples})")

print("\n→ All models processed. Embeddings saved under `embeddings/` and Qdrant collections populated.")
