# ICONQUER – Medical QA with Embeddings, RAG, and Qdrant

ICONQUER is a **medical question-answering** pipeline that combines:
- **Text embeddings** (Instructor, E5, MiniLM, etc.)
- A **vector DB (Qdrant)** for semantic retrieval
- An **LLM generator** for final answers
- Lightweight eval scripts for MedQA / HotPotQA

## What’s inside
- `make_*_emb.py`, `ingest_*_embeddings.py` — build and load train/test embeddings  
- `load_and_push_qdrant.py`, `prepare_qdrant_hotpot.py` — create Qdrant collections and upsert vectors  
- `eval_*.py`, `run_eval.sh` — run offline evaluations (cosine, distance, ROUGE)  
- `eval_qa.py`, `eval_with_qdrant*.py` — interactive/demo QA with retrieval  
- `UmlsClient.py`, `test_umls.py` — UMLS/BioPortal hooks  
- `examples*.csv/json`, `hotpotqa.pkl`, `medqa.pkl` — sample data/artifacts  
- `requirements.txt`, `.env` — deps and config

## Prerequisites
- Python 3.9+ (3.10 recommended)
- (Optional) NVIDIA GPU + CUDA/cuDNN
- **Qdrant** (Docker or Cloud)
- (Optional) OpenAI API key (or swap in a different LLM)

## Quick start

1) **Clone & env**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Configure `.env`** (create if missing)
```
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# LLM (optional)
OPENAI_API_KEY=sk-...
```

3) **Run Qdrant**
```bash
# Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

4) **Build embeddings & load**
```bash
python make_train_embeddings.py
python ingest_embeddings.py
python load_and_push_qdrant.py
```

5) **Evaluate**
```bash
bash run_eval.sh
python eval_test_only.py
python eval_with_qdrant.py
```

6) **Try QA**
```bash
python eval_qa.py
python eval_with_qdrant_enriched.py
```

---

## 📂 Datasets

You’ll need to download the main datasets used in ICONQUER:

- **PubMedQA (ori_pqaa.json)** — the core dataset used in experimentation  
  🔗 [PubMedQA GitHub](https://github.com/pubmedqa/pubmedqa) (PQA-A and PQA-U splits are provided there)  

- **HotpotQA full wiki (hotpotqa_full_wiki)** — used for generalization  
  🔗 [HotpotQA official website / GitHub](https://hotpotqa.github.io/)  

After downloading, place the files in the project root (or adjust paths in `eval_*.py` and ingestion scripts).

---

## Tips & Troubleshooting
- **GPU check**: `python gpu_check.py` or `python gpu_test.py`  
- **Qdrant**: If searches return empty, confirm collection name, vector size, and distance.  
- **LLM keys**: Ensure `OPENAI_API_KEY` is set if using OpenAI.  

---

## Repo structure (minimal)
```
.
├─ embeddings/               # (optional) saved vectors
├─ pytorch/, test_emb/       # model helpers / tests
├─ *_emb*.pkl                # embedding artifacts
├─ eval_*.py                 # evaluation & QA scripts
├─ load_and_push_qdrant.py   # Qdrant loader
├─ requirements.txt
└─ .env                      # config variables
```

## License
MIT License.

## Citation
If you use this project, please cite the ICONQUER work (add BibTeX once finalized).
