# -*- coding: utf-8 -*-
"""
MedQA evaluation with GPU support (sm_120).
All embedding steps are here, but some are commented out so you
can focus on the models you need.
"""

import os, gc, pickle, random
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util
import spacy, nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1) Seed & device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 2) NLP setup
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
STOP = set(stopwords.words('english'))
STEMMER = SnowballStemmer("english")

# 3) Load & split
df = pd.read_json("ori_pqaa.json", orient="index")
df = df.drop_duplicates("QUESTION").reset_index(drop=True)
df["CONTEXTS"] = df["CONTEXTS"].apply(lambda x: x[0] if isinstance(x, list) and x else x)
train, test = train_test_split(df, test_size=0.2, random_state=SEED)
questions = train["QUESTION"].tolist()
contexts  = train["CONTEXTS"].tolist()

# 4) Tokenization
def tokenize(text):
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in STOP]

train["TOK_Q"] = train["QUESTION"].apply(tokenize)
train["TOK_C"] = train["CONTEXTS"].apply(tokenize)

# helper to free GPU memory
def free_mem():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# 5) WORD2VEC (CPU) — skip
if False:
    print("→ Running Word2Vec…")
    # ... your Word2Vec code here ...
    free_mem()

# 6) FASTTEXT (CPU) — skip or run as needed
if False:
    print("→ Running FastText simulation…")
    # ... your FastText code here ...
    free_mem()

# 7) INSTRUCTOR (GPU) — skip
if False:
    print("→ Running Instructor…")
    # ... your Instructor code here ...
    free_mem()

# 8) JINA (GPU) — skip
if False:
    print("→ Running Jina…")
    # ... your Jina code here ...
    free_mem()

# 9) E5 (GPU via SentenceTransformer)
print("\n→ Running E5 embeddings…")
e5_model = SentenceTransformer('intfloat/e5-base-v2', device=DEVICE)
e5_q = e5_model.encode(questions, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
e5_c = e5_model.encode(contexts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
with open("emb_e5.pkl", "wb") as f:
    pickle.dump((e5_q, e5_c), f)
free_mem()

# 10) SPECTER (GPU via SentenceTransformer)
print("\n→ Running SPECTER embeddings…")
spec_model = SentenceTransformer('sentence-transformers/allenai-specter', device=DEVICE)
spec_q = spec_model.encode(questions, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
spec_c = spec_model.encode(contexts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
with open("emb_specter.pkl", "wb") as f:
    pickle.dump((spec_q, spec_c), f)
free_mem()

# 11) SBERT models (GPU)
def run_st(name, model_id, inputs, batch_size):
    print(f"\n→ Running {name}…")
    m = SentenceTransformer(model_id, device=DEVICE)
    emb = m.encode(inputs, normalize_embeddings=True,
                   batch_size=batch_size, show_progress_bar=True)
    del m; free_mem()
    return emb

sbert_models = [
    ("MPNET",   "all-mpnet-base-v2",        64),
    ("MINILM6", "all-MiniLM-L6-v2",       128),
    ("MINILM12","all-MiniLM-L12-v2",       64),
    ("LaBSE",   "sentence-transformers/LaBSE", 32),
]
for name, mid, bs in sbert_models:
    q_emb = run_st(f"SBERT-{name}-Q", mid, questions, bs)
    c_emb = run_st(f"SBERT-{name}-C", mid, contexts, bs)
    with open(f"emb_{name.lower()}.pkl","wb") as f:
        pickle.dump((q_emb, c_emb), f)

print("\nAll requested embeddings generated.")
