#!/usr/bin/env python3
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 1) Load test split
test_df = pickle.load(open("testqa.pkl", "rb"))
questions = test_df["QUESTION"].tolist()
print(f"→Loaded testqa.pkl with {len(questions)} questions.")

# 2) Load Instructor context embeddings from train
#    Key is "INSTR", since emb_instr.pkl → "INSTR"
Q_train, C_train = pickle.load(open("emb_instr.pkl", "rb"))
C_train = np.asarray(C_train, dtype=np.float32)
print(f"Loaded INSTR context pool, shape = {C_train.shape}")

# 3) Instantiate Instructor model on GPU (if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device = {DEVICE} for Instructor encoding.")
instr_model = SentenceTransformer("hkunlp/instructor-large", device=DEVICE)

# 4) Prepare inputs for Instructor: a list of [ ["Represent the question for retrieval:", q], … ]
instr_inputs = [[ "Represent the question for retrieval:", q ] for q in questions]

# 5) Encode test questions
print("→ Encoding test questions through INSTR …")
instr_q_test = instr_model.encode(
    instr_inputs,
    normalize_embeddings=True,
    batch_size=16,
    show_progress_bar=True
)
instr_q_test = np.asarray(instr_q_test, dtype=np.float32)
print(f"→ Produced INSTR Q_test shape = {instr_q_test.shape}")

# 6) Save out test_emb_instr.pkl
out_path = "test_emb_instr.pkl"
with open(out_path, "wb") as f:
    pickle.dump((instr_q_test, C_train), f)
print(f" Saved {out_path}: (Q_test: {instr_q_test.shape}, C_train: {C_train.shape})")
