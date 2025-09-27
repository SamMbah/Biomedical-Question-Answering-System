# -*- coding: utf-8 -*-
"""
MedQA evaluation with GPU support (CUDA 12.8 / sm_120).
Embeddings are computed one model at a time to avoid OOM kills:
  1) Word2Vec (CPU)
  2) FastText (CPU)
  3) Instructor (GPU)
  4) Jina (GPU)
  5) USE (CPU)
  6) SBERT all-mpnet-base-v2 (GPU)
  7) SBERT all-MiniLM-L6-v2 (GPU)
  8) SBERT all-MiniLM-L12-v2 (GPU)
  9) SBERT LaBSE (GPU)
"""
import os, gc, pickle, random
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util
import tensorflow_hub as hub
import spacy, nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1) SET SEED & DEVICE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# 2) NLTK / SpaCy
nltk.download('stopwords'); nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
STOP = set(stopwords.words('english'))
STEMMER = SnowballStemmer("english")

# 3) LOAD & SPLIT
df = pd.read_json("ori_pqaa.json", orient="index")
df = df.drop_duplicates("QUESTION").reset_index(drop=True)
df["CONTEXTS"] = df["CONTEXTS"].apply(lambda x: x[0] if isinstance(x,list) and x else x)
train, test = train_test_split(df, test_size=0.2, random_state=SEED)
questions = train["QUESTION"].tolist()
contexts  = train["CONTEXTS"].tolist()

# 4) TOKENIZATION
def tok(txt):
    doc = nlp(txt.lower())
    return [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in STOP]

train["TOK_Q"] = train["QUESTION"].apply(tok)
train["TOK_C"] = train["CONTEXTS"].apply(tok)

# helper to free GPU memory
def free_mem():
    gc.collect()
    if DEVICE=="cuda":
        torch.cuda.empty_cache()

# 5) WORD2VEC & FASTTEXT (CPU)
def cpu_w2v(ft=False):
    vec_q = Word2Vec(sentences=train["TOK_Q"],  vector_size=768, window=2, min_count=2, workers=4)
    vec_c = Word2Vec(sentences=train["TOK_C"],  vector_size=768, window=2, min_count=2, workers=4)
    for col, model in [("WORD2VEC", vec_q), ("WORD2VEC_C", vec_c)]:
        emb = [np.mean([model.wv[t] for t in toks if t in model.wv], axis=0) 
               if any(t in model.wv for t in toks) else np.zeros(768)
               for toks in (train["TOK_Q"] if col=="WORD2VEC" else train["TOK_C"])]
        train[f"{col}_EMB"] = emb
    # FastText simulation reuses Word2Vec API
    if ft:
        train["FASTTEXT_Q"] = train["TOK_Q"].apply(lambda toks: 
            np.mean([vec_q.wv[t] for t in toks if t in vec_q.wv], axis=0) 
            if any(t in vec_q.wv for t in toks) else np.zeros(768))
        train["FASTTEXT_C"] = train["TOK_C"].apply(lambda toks: 
            np.mean([vec_c.wv[t] for t in toks if t in vec_c.wv], axis=0) 
            if any(t in vec_c.wv for t in toks) else np.zeros(768))
    # save & drop
    with open("emb_w2v_ft.pkl", "wb") as f: pickle.dump(train, f)
    del vec_q, vec_c; free_mem()

cpu_w2v(ft=True)


# generic GPU embedder
def run_st(name, model_id, inputs, batch_size, out_prefix):
    print(f"\n→ Running {name}...")
    model = SentenceTransformer(model_id, device=DEVICE, trust_remote_code=True)
    emb = model.encode(inputs, normalize_embeddings=True,
                       batch_size=batch_size, show_progress_bar=True)
    del model; free_mem()
    return emb

# 6) INSTRUCTOR
instr_q = run_st("Instructor-Q",  "hkunlp/instructor-large", 
                 [["Represent the question for retrieval:", q] for q in questions],
                 batch_size=16, out_prefix="INSTR_Q")
instr_c = run_st("Instructor-C",  "hkunlp/instructor-large", 
                 [["Represent the document for retrieval:", c] for c in contexts],
                 batch_size=16, out_prefix="INSTR_C")
with open("emb_instr.pkl","wb") as f: pickle.dump((instr_q, instr_c), f)

# 7) JINA
jina_q = run_st("Jina-Q", "jinaai/jina-embeddings-v2-base-en", 
                questions, batch_size=32, out_prefix="JINA_Q")
jina_c = run_st("Jina-C", "jinaai/jina-embeddings-v2-base-en", 
                contexts, batch_size=32, out_prefix="JINA_C")
with open("emb_jina.pkl","wb") as f: pickle.dump((jina_q, jina_c), f)

# 8) USE (CPU)
print("\n→ Running USE (CPU)...")
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
use_q = use(questions).numpy()
use_c = use(contexts).numpy()
with open("emb_use.pkl","wb") as f: pickle.dump((use_q, use_c), f)
free_mem()

# 9-12) SBERTs
sbert_models = [
    ("MPNET",   "all-mpnet-base-v2", 64),
    ("MINILM6", "all-MiniLM-L6-v2",  64),
    ("MINILM12","all-MiniLM-L12-v2", 64),
    ("LaBSE",   "sentence-transformers/LaBSE", 32),
]
for name, mid, bs in sbert_models:
    q_emb = run_st(f"SBERT-{name}-Q", mid, questions, batch_size=bs, out_prefix=f"{name}_Q")
    c_emb = run_st(f"SBERT-{name}-C", mid, contexts, batch_size=bs, out_prefix=f"{name}_C")
    with open(f"emb_{name.lower()}.pkl","wb") as f: pickle.dump((q_emb, c_emb), f)

# ———————————————————————————————————————————————————————————————
# 13) LOAD ALL EMBEDDINGS back & EVALUATE
print("\n→ Loading all embeddings back in and evaluating…")
with open("emb_w2v_ft.pkl","rb")     as f: df = pickle.load(f)  # has WORD2VEC/_FT columns
with open("emb_instr.pkl","rb")      as f: instr_q, instr_c = pickle.load(f)
with open("emb_jina.pkl","rb")       as f: jina_q, jina_c = pickle.load(f)
with open("emb_use.pkl","rb")        as f: use_q, use_c = pickle.load(f)
emb_store = {"INSTR":(instr_q,instr_c),"JINA":(jina_q,jina_c),"USE":(use_q,use_c)}

for name,_,_ in sbert_models:
    with open(f"emb_{name.lower()}.pkl","rb") as f:
        emb_store[name] = pickle.load(f)

# now assemble for evaluation
results = []
def evaluate_pair(q_emb, c_emb):
    sims  = [util.cos_sim(torch.tensor(q),torch.tensor(c)).item() for q,c in zip(q_emb, c_emb)]
    dists = [distance.euclidean(q, c) for q,c in zip(q_emb, c_emb)]
    return np.mean(sims), np.mean(dists)

# CPU models
results.append(("WORD2VEC", df["WORD2VEC_EMB"].tolist(), df["WORD2VEC_C_EMB"].tolist()))
results.append(("FASTTEXT", df["FASTTEXT_Q"].tolist(), df["FASTTEXT_C"].tolist()))
# others
for name,(q_emb,c_emb) in emb_store.items():
    results.append((name, q_emb, c_emb))

# compute metrics
metrics = []
for name, q_emb, c_emb in results:
    cos, euc = evaluate_pair(q_emb, c_emb)
    metrics.append({"model":name, "mean_cosine":cos, "mean_euclid":euc})

res_df = pd.DataFrame(metrics).sort_values("mean_cosine", ascending=False)
print(res_df)

# 14) VISUALIZE
ax = res_df.plot.barh(x="model", y="mean_cosine", legend=False)
ax2 = ax.twiny()
res_df.plot.barh(x="model", y="mean_euclid", ax=ax2, legend=False)
ax.set_xlabel("Mean Cosine"); ax2.set_xlabel("Mean Euclid"); ax.set_title("MedQA Embedding Comparison")
plt.tight_layout(); plt.show()
