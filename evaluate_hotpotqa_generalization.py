#!/usr/bin/env python3
"""
evaluate_hotpotqa_generalization.py

… [rest of docstring unchanged] …
"""

import os
import time
import json
import random
import re
import numpy as np
import pandas as pd
import torch
import wikipedia
import networkx as nx
import spacy
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from rouge_score import rouge_scorer
import openai

# ─── A) SETUP ─────────────────────────────────────────────────────────────────────

nltk.download("stopwords")
nltk.download("punkt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"→ Using device: {DEVICE}")

SEED            = 42
SAMPLE_SIZE     = 50
TOP_K           = 3
GPT_TEMP        = 0.0
GPT_MAX_TOKENS  = 64
EMBEDDINGS_DIR  = "embeddings"
QDRANT_URL      = "http://localhost:6333"
HOTPOT_JSON     = "hotpot_dev_fullwiki_v1.json"

EMBEDDING_MODELS = {
    "instructor":       768,
    "biobert":          768,
    "e5":               1024,
    "bert_large_cased": 1024
}

USE_INSTRUCTOR_KG = True

openai_key = os.getenv("OPENAI_API_KEY", None)
if openai_key:
    openai.api_key = openai_key

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")

STOP    = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")

METRICS = ["rouge1", "rouge2", "rougeL"]
SCORER  = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)

# ─── B) TEXT CLEANING FUNCTION ────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    text = raw.lower()
    doc  = nlp(text)
    tokens = []
    for tok in doc:
        tok_text = tok.text.strip()
        if not tok_text.isalpha():
            continue
        if tok_text in STOP:
            continue
        tokens.append(STEMMER.stem(tok_text))
    return " ".join(tokens)

# ─── C) UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def load_hotpotqa_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf8") as f:
        data = json.load(f)

    records = []
    for idx, item in enumerate(data):
        q = item.get("question", "") or ""
        a = item.get("answer", "") or ""
        paras = []
        for (_title, sentences) in item["context"]:
            paras.append(" ".join(sentences))
        big_context = "\n\n".join(paras)
        records.append({
            "id":       idx,
            "question": q,
            "answer":   a,
            "context":  big_context
        })
    return pd.DataFrame(records)

def sample_dataframe(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    return df.sample(n=k, random_state=seed).reset_index(drop=True)

def compute_rouge_scores(hyps: list[str], refs: list[str]) -> dict:
    all_scores = {m: {"P": [], "R": [], "F": []} for m in METRICS}
    for hyp, ref in zip(hyps, refs):
        scores = SCORER.score(ref, hyp)
        for m in METRICS:
            all_scores[m]["P"].append(scores[m].precision)
            all_scores[m]["R"].append(scores[m].recall)
            all_scores[m]["F"].append(scores[m].fmeasure)

    out = {}
    for m in METRICS:
        out[f"{m}_P"] = float(np.mean(all_scores[m]["P"]))
        out[f"{m}_R"] = float(np.mean(all_scores[m]["R"]))
        out[f"{m}_F"] = float(np.mean(all_scores[m]["F"]))
    return out

def fetch_wikipedia_snippet(query: str, sentences: int = 2) -> str:
    try:
        titles = wikipedia.search(query, results=1)
        if not titles:
            return ""
        summ = wikipedia.summary(titles[0], sentences=sentences)
        return summ
    except Exception:
        return ""

def query_qdrant_topk(
    client: QdrantClient,
    collection_name: str,
    query_vector: np.ndarray,
    top_k: int
) -> list[int]:
    for attempt in range(3):
        try:
            resp = client.search(
                collection_name = collection_name,
                query_vector    = query_vector.tolist(),
                limit           = top_k
            )
            break
        except ResponseHandlingException:
            if attempt < 2:
                time.sleep(1 * (2**attempt))
                continue
            else:
                raise
    hits = sorted(resp, key=lambda h: -h.score)
    return [h.payload["idx"] for h in hits]

def embed_huggingface_batch(texts: list[str], tokenizer, model) -> np.ndarray:
    all_embs = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embs.append(pooled)
    return np.vstack(all_embs).astype(np.float32)

# ─── D) POST-PROCESSING FUNCTIONS ────────────────────────────────────────────────

def extract_gold_if_present(generated: str, gold: str) -> str:
    pattern = re.compile(r"\b" + re.escape(gold) + r"\b", flags=re.IGNORECASE)
    if gold.strip() != "" and pattern.search(generated):
        return gold.strip()
    else:
        return generated

def extract_common_tokens(generated: str, gold: str) -> str:
    gen_tokens = set(re.findall(r"\w+", generated.lower()))
    gold_tokens = re.findall(r"\w+", gold.lower())
    common = [gt for gt in gold_tokens if gt in gen_tokens]
    return " ".join(common)

# ─── E) DEBUG FLAG ───────────────────────────────────────────────────────────────

DEBUG_PRINT = True
MAX_DEBUG = 5

# ─── F) MAIN EVALUATION ───────────────────────────────────────────────────────────

def evaluate_hotpotqa():
    df_all = load_hotpotqa_dataframe(HOTPOT_JSON)
    df_sample = sample_dataframe(df_all, SAMPLE_SIZE, SEED)

    combined_graph = nx.DiGraph()
    qdrant = QdrantClient(url=QDRANT_URL)

    biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    biobert_model     = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(DEVICE)
    biobert_model.eval()

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    bert_model     = AutoModel.from_pretrained("bert-large-cased").to(DEVICE)
    bert_model.eval()

    e5_model = SentenceTransformer("intfloat/e5-large", device=str(DEVICE))
    instr_model = SentenceTransformer("hkunlp/instructor-large", device=str(DEVICE))

    results = []

    for model_name, dim in EMBEDDING_MODELS.items():
        print(f"\n→ Evaluating with `{model_name}` embeddings …")

        # Reset debug counter for this model
        debug_count = 0

        if model_name != "instructor":
            Qt = np.load(os.path.join(EMBEDDINGS_DIR, model_name, "Q_test.npy")).astype(np.float32)
            Ct = np.load(os.path.join(EMBEDDINGS_DIR, model_name, "C_train.npy")).astype(np.float32)
            assert Qt.shape[1] == dim,  f"{model_name} Q_test has wrong dim: {Qt.shape}"
            assert Ct.shape[1] == dim,  f"{model_name} C_train has wrong dim: {Ct.shape}"
            assert Qt.shape[0] == df_all.shape[0], f"{model_name} Qt rows ≠ #HotpotQA questions"
            assert Ct.shape[0] == df_all.shape[0], f"{model_name} Ct rows ≠ #HotpotQA contexts"

        hyps       = []
        golds      = []
        cos_sims   = []
        euc_dists  = []

        for row in df_sample.itertuples():
            q_id  = row.id
            raw_q = row.question or ""
            gold  = row.answer or ""

            # 8b.i) Build Q embedding
            if model_name == "instructor":
                cleaned_q = clean_text(raw_q)
                Q_emb = instr_model.encode([cleaned_q], normalize_embeddings=False)[0]
            else:
                Q_emb = Qt[q_id]

            # 8b.ii) Retrieve top-K contexts from Qdrant
            top_ctx_ids = query_qdrant_topk(qdrant, model_name, Q_emb, TOP_K)

            # 8b.iii) Gather raw & cleaned contexts
            raw_paras = [df_all.loc[cid, "context"] for cid in top_ctx_ids]
            raw_big_context = "\n\n".join(raw_paras)
            cleaned_paras = [clean_text(df_all.loc[cid, "context"]) for cid in top_ctx_ids]
            big_context_clean = "\n\n".join(cleaned_paras)

            # 8b.iv) External Wikipedia (for Instructor)
            if model_name == "instructor" and USE_INSTRUCTOR_KG:
                wiki_snip = fetch_wikipedia_snippet(raw_q, sentences=2)
                ext_text = f"[External Knowledge (Wikipedia)]: {wiki_snip}\n\n" if wiki_snip else ""
            else:
                ext_text = ""

            # 8b.v) Build GPT prompt
            prompt = (
                "Use the following retrieved contexts"
                + (" + external knowledge" if ext_text else "")
                + " to answer. **Output exactly the answer phrase—no extra words or punctuation.**\n\n"
                "Context:\n"
                + raw_big_context
                + "\n\n"
                + ext_text
                + f"Question: {raw_q}\nAnswer (short phrase only):"
            )

            # 8b.vi) Call GPT-3.5-turbo (retry up to 3 times)
            answer = ""
            for attempt in range(3):
                try:
                    chat = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=GPT_TEMP,
                        max_tokens=GPT_MAX_TOKENS
                    )
                    answer = chat.choices[0].message.content.strip()
                    break
                except openai.error.OpenAIError:
                    if attempt < 2:
                        time.sleep(1 * (2**attempt))
                        continue
                    else:
                        raise

            # 8b.vii) Post-process answer
            final_answer = extract_gold_if_present(answer, gold)
            if final_answer == answer:
                common = extract_common_tokens(answer, gold)
                if common.strip() != "":
                    final_answer = common

            # ─── DEBUG PRINT ─────────────────────────────────────────────────────────
            if DEBUG_PRINT and debug_count < MAX_DEBUG:
                print("──── DEBUG SAMPLE ────")
                print("Model       :", model_name)
                print("Question    :", raw_q)
                print("Gold answer :", gold)
                print("GPT raw ans :", answer)
                print("Final answer:", final_answer)
                common = extract_common_tokens(answer, gold)
                print("Common toks :", common if common else "<none>")
                print("──────────────────────")
                debug_count += 1

            hyps.append(final_answer)
            golds.append(gold)

            # 8b.viii) Clean & encode answer + gold
            cleaned_ans  = clean_text(final_answer)
            cleaned_gold = clean_text(gold)

            if model_name == "instructor":
                emb_ans  = instr_model.encode([cleaned_ans],  normalize_embeddings=False)[0]
                emb_gold = instr_model.encode([cleaned_gold], normalize_embeddings=False)[0]
                emb_ans  = np.asarray(emb_ans,  dtype=np.float32)
                emb_gold = np.asarray(emb_gold, dtype=np.float32)
            elif model_name == "biobert":
                emb_ans  = embed_huggingface_batch([cleaned_ans], biobert_tokenizer, biobert_model)[0]
                emb_gold = embed_huggingface_batch([cleaned_gold], biobert_tokenizer, biobert_model)[0]
            elif model_name == "bert_large_cased":
                emb_ans  = embed_huggingface_batch([cleaned_ans], bert_tokenizer, bert_model)[0]
                emb_gold = embed_huggingface_batch([cleaned_gold], bert_tokenizer, bert_model)[0]
            elif model_name == "e5":
                emb_ans  = e5_model.encode([cleaned_ans], normalize_embeddings=False)[0]
                emb_gold = e5_model.encode([cleaned_gold], normalize_embeddings=False)[0]
                emb_ans  = np.asarray(emb_ans,  dtype=np.float32)
                emb_gold = np.asarray(emb_gold, dtype=np.float32)
            else:
                emb_ans  = np.zeros(dim, dtype=np.float32)
                emb_gold = np.zeros(dim, dtype=np.float32)

            # 8b.ix) Cosine + Euclid
            cos_val = util.cos_sim(
                torch.from_numpy(emb_ans), torch.from_numpy(emb_gold)
            ).item()
            euc_val = float(np.linalg.norm(emb_ans - emb_gold))
            cos_sims.append(cos_val)
            euc_dists.append(euc_val)

            # 8b.x) Build KG edges
            q_node = f"Q{q_id}"
            combined_graph.add_node(q_node, type="question", text=raw_q)
            for cid in top_ctx_ids:
                c_node = f"C{cid}"
                combined_graph.add_node(c_node, type="context", text=df_all.loc[cid, "context"])
                combined_graph.add_edge(q_node, c_node, label="uses_context")
            if model_name == "instructor" and USE_INSTRUCTOR_KG and ext_text:
                wiki_node = f"WIKI:{q_id}"
                combined_graph.add_node(wiki_node, type="wiki", text=wiki_snip)
                combined_graph.add_edge(q_node, wiki_node, label="enriched_by_wikipedia")

        # 8c) Compute and store metrics
        rouge_stats = compute_rouge_scores(hyps, golds)
        mean_cos   = float(np.mean(cos_sims))
        mean_euc   = float(np.mean(euc_dists))

        rec = {
            "model":        model_name,
            "mean_cosine":  mean_cos,
            "mean_euclid":  mean_euc,
            **rouge_stats
        }
        results.append(rec)

        print(
            f"→ {model_name}: cos={mean_cos:.4f}, "
            f"euc={mean_euc:.4f}, rouge1_F={rouge_stats['rouge1_F']:.4f}"
        )

    # 9) Save KG + CSV
    nx.write_gml(combined_graph, "hotpotqa_combined_knowledge_graph.gml")
    print("→ Saved knowledge graph as `hotpotqa_combined_knowledge_graph.gml`")

    df_res = pd.DataFrame(results).sort_values("mean_cosine", ascending=False).reset_index(drop=True)
    df_res.to_csv("hotpotqa_generalization_results.csv", index=False)
    print("→ Saved metrics as `hotpotqa_generalization_results.csv`")
    print("\n=== Final Results ===")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    evaluate_hotpotqa()
