#!/usr/bin/env python3
# make_train_embeddings.py

import os
import pickle
import numpy as np

# your SBERT-style pickles:
SBERT_PKLS = [
    "emb_instr.pkl",
    "emb_jina.pkl",
    "emb_e5.pkl",
    "emb_specter.pkl",
    "emb_mpnet.pkl",
    "emb_minilm6.pkl",
    "emb_minilm12.pkl",
    "emb_labse.pkl",
]

# your Word2Vec+FastText combined pickle:
W2VFT_PKL = "emb_w2v_ft.pkl"

OUTPUT_PICKLE = "train_embeddings.pkl"

def load_and_cast(fname):
    """Load a (Q, C) tuple from pickle, cast to float16, return (name, (Q16, C16))."""
    data = pickle.load(open(fname, "rb"))
    if not (isinstance(data, tuple) and len(data) == 2):
        raise ValueError(f"{fname} is not a (Q_all, C_all) tuple")
    Q, C = data
    Q16 = np.asarray(Q, dtype=np.float16)
    C16 = np.asarray(C, dtype=np.float16)
    if Q16.shape != C16.shape:
        raise ValueError(f"{fname} shape mismatch {Q16.shape} vs {C16.shape}")
    name = os.path.basename(fname).replace("emb_", "").replace(".pkl","").upper()
    return name, (Q16, C16)

def load_w2v_ft_and_cast(fname):
    """Load your emb_w2v_ft DataFrame, extract 4 columns, cast to float16, return dict."""
    df = pickle.load(open(fname, "rb"))
    out = {}
    # Word2Vec
    Qw = np.asarray(df["WORD2VEC_EMB"].tolist(), dtype=np.float16)
    Cw = np.asarray(df["WORD2VEC_C_EMB"].tolist(), dtype=np.float16)
    out["WORD2VEC"] = (Qw, Cw)
    # FastText
    Qf = np.asarray(df["FASTTEXT_Q"].tolist(), dtype=np.float16)
    Cf = np.asarray(df["FASTTEXT_C"].tolist(), dtype=np.float16)
    out["FASTTEXT"] = (Qf, Cf)
    return out

def main():
    store = {}

    # 1) SBERT pickles
    for fn in SBERT_PKLS:
        if not os.path.isfile(fn):
            print(f"  {fn} missing, skipping.")
            continue
        try:
            name, pair = load_and_cast(fn)
            store[name] = pair
            print(f" Loaded {name}: {pair[0].shape[0]}×{pair[0].shape[1]} (float16)")
        except Exception as e:
            print(f" Skipped {fn}: {e}")

    # 2) Word2Vec+FastText
    if os.path.isfile(W2VFT_PKL):
        try:
            w2vft = load_w2v_ft_and_cast(W2VFT_PKL)
            for name, pair in w2vft.items():
                store[name] = pair
                print(f" Loaded {name}: {pair[0].shape[0]}×{pair[0].shape[1]} (float16)")
        except Exception as e:
            print(f"Skipped {W2VFT_PKL}: {e}")
    else:
        print(f" {W2VFT_PKL} not found, skipping word embeddings.")

    if not store:
        print(" No embeddings loaded, aborting.")
        return

    # 3) Dump with highest protocol
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n Wrote {OUTPUT_PICKLE} with models: {', '.join(store.keys())}")

if __name__ == "__main__":
    main()
