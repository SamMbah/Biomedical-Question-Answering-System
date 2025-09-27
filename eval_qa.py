# eval_qa.py
import os
import random
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sentence_transformers import util
from qdrant_client import QdrantClient
from openai import OpenAI
from rouge_score import rouge_scorer, scoring

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration
SAMPLE_SIZE = 50    # how many QA pairs to evaluate
TOP_K       = 3     # how many contexts to retrieve per question
TEMP        = 0.0   # deterministic GPT output
MAX_TOKENS  = 256

# 2) Load & sample only the approved QA pairs
df = pd.read_json("ori_pqaa.json", orient="index")
print("Columns in your dataset:", df.columns.tolist())
df = df[df["final_decision"] == "yes"].reset_index(drop=True)

random.seed(42)
sample_idxs = random.sample(range(len(df)), SAMPLE_SIZE)
samples     = df.loc[sample_idxs].reset_index(drop=True)

# 3) Load all emb_*.pkl that unpack to (q_embs, c_embs)
embeddings = {}
for fn in os.listdir():
    if fn.startswith("emb_") and fn.endswith(".pkl"):
        name = fn[len("emb_"):-len(".pkl")].upper()
        obj  = pickle.load(open(fn, "rb"))
        if isinstance(obj, tuple) and len(obj) == 2:
            embeddings[name] = obj
        else:
            print(f"Skipping {fn}: not a (q,c) tuple")
print("Loaded embeddings for:", list(embeddings.keys()))

# 4) Prepare clients
#   Qdrant (not used in this script, but left here for future)
qdrant = QdrantClient(url="http://localhost:6333")

#   OpenAI: strip newline/spaces from your API key
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing or empty (after strip)!")
openai = OpenAI(api_key=api_key)

# … the imports and setup remain the same …

from rouge_score import rouge_scorer

# set up once globally
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

def evaluate_rouge(hyps, refs):
    """
    Compute average precision, recall, and F1 for
    rouge1, rouge2 and rougeL across all (hyp,ref) pairs.
    """
    metrics = ["rouge1", "rouge2", "rougeL"]
    # sums for P/R/F
    sums = {m: {"P":0.0, "R":0.0, "F":0.0} for m in metrics}
    n = len(hyps)

    for hyp, ref in zip(hyps, refs):
        scores = scorer.score(ref, hyp)
        for m in metrics:
            sc = scores[m]
            sums[m]["P"] += sc.precision
            sums[m]["R"] += sc.recall
            sums[m]["F"] += sc.fmeasure

    # now average
    out = {}
    for m in metrics:
        out[f"{m}_P"] = sums[m]["P"] / n
        out[f"{m}_R"] = sums[m]["R"] / n
        out[f"{m}_F"] = sums[m]["F"] / n

    return out



# … the rest of your eval loop and final reporting unchanged …

# 6) Main evaluation loop
records = []
for model_name, (q_all, c_all) in embeddings.items():
    print(f"\n→ Evaluating {model_name}…")
    cos_vals, euc_vals = [], []
    hyps, refs = [], []

    for i, row in samples.iterrows():
        # a) retrieve top-K by cosine
        qv = torch.tensor(q_all[i], dtype=torch.float32)
        sims = util.cos_sim(qv, torch.tensor(c_all, dtype=torch.float32)).cpu().tolist()[0]
        topk = sorted(range(len(sims)), key=lambda j: sims[j], reverse=True)[:TOP_K]

        # build a single context string
        ctxs = " ".join([ row["CONTEXTS"] if isinstance(row["CONTEXTS"], str) 
                          else row["CONTEXTS"][0] 
                          for _ in topk ])

        # b) prompt GPT-3.5
        prompt = (
            f"Use the following context to answer the question succinctly.\n\n"
            f"Context:\n{ctxs}\n\n"
            f"Question: {row['QUESTION']}\nAnswer:"
        )
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=TEMP,
            max_tokens=MAX_TOKENS,
        )
        pred = resp.choices[0].message.content.strip()

        # c) accumulate sim/distance stats
        #    (mean over the TOP_K contexts)
        cos_vals.append( np.mean([ sims[j] for j in topk ]) )
        euc_vals.append( np.mean([
            distance.euclidean(q_all[i], c_all[j]) for j in topk
        ]) )

        hyps.append(pred)
        refs.append(row["LONG_ANSWER"])

    # aggregate
    rouge_stats = evaluate_rouge(hyps, refs)
    records.append({
        "model":       model_name,
        "mean_cosine": np.mean(cos_vals),
        "mean_euclid": np.mean(euc_vals),
        **rouge_stats
    })

# 7) Report & save
res_df = pd.DataFrame(records).sort_values("mean_cosine", ascending=False)
print("\n=== Final Evaluation ===")
# … after you build res_df …

print("\n=== Final evaluation ===")
try:
    # pretty Markdown table if tabulate is installed
    print(res_df.to_markdown(index=False))
except ImportError:
    # fallback
    print(res_df.to_string(index=False))


res_df.to_csv("llm_eval_results.csv", index=False)
print("\nResults written to llm_eval_results.csv")
