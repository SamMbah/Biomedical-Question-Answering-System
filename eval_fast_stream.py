# eval_fast_stream.py
import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sentence_transformers import util
import matplotlib.pyplot as plt

# suppress any TF logs if they leak in
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def mean_cosine_and_euclid(q_np, c_np):
    # vectorized all‐at‐once in PyTorch, using float32
    a = torch.from_numpy(q_np)
    b = torch.from_numpy(c_np)
    a_norm = torch.nn.functional.normalize(a, dim=1)
    b_norm = torch.nn.functional.normalize(b, dim=1)
    cos = (a_norm * b_norm).sum(dim=1).mean().item()
    euc = (a - b).norm(dim=1).mean().item()
    return cos, euc

def main():
    # 1) Load CPU embeddings (downcast to float32)
    df = load_pkl("emb_w2v_ft.pkl")
    w2v_q = np.stack(df["WORD2VEC_EMB"].tolist()).astype(np.float32)
    w2v_c = np.stack(df["WORD2VEC_C_EMB"].tolist()).astype(np.float32)
    ft_q  = np.stack(df["FASTTEXT_Q"].tolist()).astype(np.float32)
    ft_c  = np.stack(df["FASTTEXT_C"].tolist()).astype(np.float32)
    del df; gc.collect()

    # 2) Load ST‐style embeddings (float32)
    emb_store = {}
    for key, fname in [
        ("INSTR",    "emb_instr.pkl"),
        ("JINA",     "emb_jina.pkl"),
        ("E5",       "emb_e5.pkl"),
        ("SPECTER",  "emb_specter.pkl"),
    ]:
        q, c = load_pkl(fname)
        emb_store[key] = (np.array(q, dtype=np.float32),
                          np.array(c, dtype=np.float32))

    for name in ["mpnet", "minilm6", "minilm12", "labse"]:
        q, c = load_pkl(f"emb_{name}.pkl")
        emb_store[name.upper()] = (np.array(q, dtype=np.float32),
                                   np.array(c, dtype=np.float32))

    # 3) Group sentence models by embedding dimension
    by_dim = {}
    for k, (q, c) in emb_store.items():
        dim = q.shape[1]
        by_dim.setdefault(dim, {})[k] = (q, c)

    records = []

    # 4) Helper to evaluate & record
    def eval_and_record(name, q, c):
        cos, euc = mean_cosine_and_euclid(q, c)
        records.append((name, cos, euc))
        # free immediately
        del q, c
        gc.collect()

    # 5) Evaluate base CPU models
    eval_and_record("WORD2VEC", w2v_q, w2v_c)
    eval_and_record("FASTTEXT", ft_q, ft_c)

    # 6) Evaluate pure ST models
    for k, (q, c) in emb_store.items():
        eval_and_record(k, q, c)

    # 7) Pairwise average within each dim‐group
    for dim, group in by_dim.items():
        names = list(group.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                m1, m2 = names[i], names[j]
                q1, c1 = group[m1]
                q2, c2 = group[m2]
                avg_q = ((q1 + q2) / 2.0).astype(np.float32)
                avg_c = ((c1 + c2) / 2.0).astype(np.float32)
                eval_and_record(f"{m1}+{m2}", avg_q, avg_c)

    # 8) Fuse W2V/FT with any ST model matching dims
    α, β = 0.3, 0.7
    for base_name, (bq, bc) in [("W2V", (w2v_q, w2v_c)), ("FT", (ft_q, ft_c))]:
        dim = bq.shape[1]
        if dim not in by_dim:
            continue
        for m, (sq, sc) in by_dim[dim].items():
            fq = (α * bq + β * sq).astype(np.float32)
            fc = (α * bc + β * sc).astype(np.float32)
            eval_and_record(f"{base_name}×{α:.1f}+{m}×{β:.1f}", fq, fc)

    # 9) Build DataFrame and show
    df_res = pd.DataFrame(records, columns=["model","mean_cosine","mean_euclid"])
    df_res = df_res.sort_values("mean_cosine", ascending=False).reset_index(drop=True)
    print(df_res)

    # 10) Plot top 10
    top10 = df_res.head(10).set_index("model")
    ax = top10["mean_cosine"].plot.barh(figsize=(8,6), legend=False)
    ax2 = ax.twiny()
    top10["mean_euclid"].plot.barh(ax=ax2, legend=False, color="gray", alpha=0.5)
    ax.set_xlabel("Mean Cosine"); ax2.set_xlabel("Mean Euclid")
    ax.set_title("Fast Streaming Ensemble Comparison")
    plt.tight_layout()
    plt.savefig("Ensemble_Embedding_Comparison.png", dpi=300)
    print("Ensemble_Embedding_Comparison.png")

if __name__=="__main__":
    main()
