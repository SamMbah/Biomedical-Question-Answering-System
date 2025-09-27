#!/usr/bin/env python3
import pickle
import numpy as np
from gensim.models import Word2Vec, FastText
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Ensure NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("punkt")

nlp     = spacy.load("en_core_web_sm")
STOP    = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")

def tokenize(text: str):
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in STOP]

# Load test split and tokenize
test_df   = pickle.load(open("testqa.pkl", "rb"))
questions = test_df["QUESTION"].tolist()
tokenized_q = [ tokenize(q) for q in questions ]
print(f"Tokenized {len(questions)} test questions for Word2Vec/FT.")

# Load W2V+FT training DF to extract context arrays
df_w2v = pickle.load(open("emb_w2v_ft.pkl", "rb"))
Cw_train = np.vstack(df_w2v["WORD2VEC_C_EMB"].tolist()).astype(np.float32)
Cf_train = np.vstack(df_w2v["FASTTEXT_C"].tolist()).astype(np.float32)

# 1) Train Word2Vec on tokenized_q
print("Training Word2Vec on tokenized test questions…")
w2v_q = Word2Vec(sentences=tokenized_q, vector_size=768, window=2, min_count=2, workers=4)

# Build Q_test for Word2Vec
Qw_list = []
for toks in tokenized_q:
    vecs = [ w2v_q.wv[t] for t in toks if t in w2v_q.wv ]
    if len(vecs) > 0:
        Qw_list.append(np.mean(vecs, axis=0))
    else:
        Qw_list.append(np.zeros(768, dtype=np.float32))
Qw_test = np.vstack(Qw_list).astype(np.float32)
print(f"Word2Vec Q_test shape = {Qw_test.shape}")

with open("test_emb_word2vec.pkl", "wb") as f:
    pickle.dump((Qw_test, Cw_train), f)
print("Saved test_emb_word2vec.pkl")

# 2) Train FastText on tokenized_q
print("→ Training FastText on tokenized test questions…")
ft_q = FastText(sentences=tokenized_q, vector_size=768, window=2, min_count=2, workers=4)

# Build Q_test for FastText
Qf_list = []
for toks in tokenized_q:
    vecs = [ ft_q.wv[t] for t in toks if t in ft_q.wv ]
    if len(vecs) > 0:
        Qf_list.append(np.mean(vecs, axis=0))
    else:
        Qf_list.append(np.zeros(768, dtype=np.float32))
Qf_test = np.vstack(Qf_list).astype(np.float32)
print(f"FastText Q_test shape = {Qf_test.shape}")

with open("test_emb_fasttext.pkl", "wb") as f:
    pickle.dump((Qf_test, Cf_train), f)
print(" Saved test_emb_fasttext.pkl")
