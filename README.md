# ICONQUER – Biomedical QA with Embeddings, RAG, and Qdrant

ICONQUER is a **biomedical question-answering** pipeline that combines:
- **Text embeddings** (Instructor, E5, MiniLM, SBERT, word2vec/fastText, etc.)
- A **vector DB (Qdrant)** for semantic retrieval
- An **LLM generator** for final answers
- Lightweight eval scripts for **PubMedQA (ori_pqaa.json)** and **HotpotQA (full wiki)**

---

## 📂 Repo Structure & Key Files

This repo is organized into scripts for **embedding**, **retrieval**, **evaluation**, and **utilities**. Below is a guide for reviewers:

### 1) Embeddings & Ingestion
- `make_train_embeddings.py` → Build embeddings from PubMedQA (`ori_pqaa.json`)
- `make_instr_test_emb.py` / `make_sbert_test_emb.py` / `make_w2v_ft_test_emb.py` → Generate test embeddings with different models
- `ingest_embeddings.py` / `ingest_test_embeddings.py` → Load embeddings into memory for pushing
- `push_w2v_ft.py` → Special handling for word2vec/fastText embeddings

### 2) Qdrant Vector DB & Retrieval
- `load_and_push_qdrant.py` → Create Qdrant collections & upsert vectors
- `prepare_qdrant_hotpot.py` → Load HotpotQA dataset into Qdrant
- `eval_with_qdrant.py` → Main retrieval + QA pipeline
- `eval_with_qdrant_enriched_with_graph.py` → Retrieval with enriched context
- `run_eval.sh` → Shell script to restart Qdrant and run evaluation

### 3) Evaluation & Experiments
- `eval_test_only.py` → Evaluate retrieval/QA on MedQA test set
- `eval_test_enriched.py` / `eval_test_with_enriched_knowledge.py` → Test with enriched knowledge
- `eval_fast_stream.py` → Faster evaluation loop for development
- `evaluate_hotpotqa_generalization.py` → Generalization experiments on HotpotQA
- `eval_qa.py` → Interactive QA from console
- `run_full_medqa_test_pipeline.py` → End-to-end MedQA pipeline (embedding → retrieval → evaluation)

### 4) Data & Examples
- `examples.json` / `examples_test*.json` → Small sample inputs
- `eval_sample_ids.json` → Fixed test IDs for reproducibility
- `print_examples.py` → Pretty-print examples

### 5) Knowledge Graph (optional)
*(kept for experiments; Qdrant-only pipeline does not require these)*
- `import_to_neo4j.py`, `view_kg.py`
- `knowledge_graph.gml`, `knowledge_graph.json`, `knowledge_graph.svg`, `hotpotqa_combined_knowledge_graph.gml`

### 6) Utilities & Environment
- `gpu_check.py`, `gpu_test.py` → Check CUDA/GPU availability
- `requirements.txt` → Python dependencies
- `.gitignore` → Ignore large models/datasets
- `UmlsClient.py`, `test_umls.py`, `umls_test.py` → UMLS integration (optional)
- `notebook.py`, `notebook_1.py`, `updatednotebook.py` → Exploratory notebooks as `.py`
- Images: `Ensemble_Embedding_Comparison.png`, `medqa_individual_embedding_comparison.png`

---

##  Quick Start

1. **Setup environment**
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Run Qdrant (Docker)**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

3. **Build & push embeddings**
```bash
python make_train_embeddings.py
python ingest_embeddings.py
python load_and_push_qdrant.py
```

4. **Evaluate**
```bash
bash run_eval.sh
python eval_test_only.py
```

5. **Try QA**
```bash
python eval_qa.py
```

---

##  Datasets

- **PubMedQA (ori_pqaa.json)** – Core dataset for experimentation  
  🔗 [PubMedQA GitHub](https://github.com/pubmedqa/pubmedqa)  

- **HotpotQA full wiki** – Used for generalization  
  🔗 [HotpotQA official site](https://hotpotqa.github.io/)  

> Place datasets in the repo root or update paths in scripts.

---

##  Tips & Troubleshooting
- **GPU check**: `python gpu_check.py`  
- **Qdrant empty results**: check collection name, vector size, and distance metric  
- **Secrets**: set `OPENAI_API_KEY` in `.env` or your shell, never in code  

---

##  Suggested Reading Order for Reviewers
1. **README.md** → overview + setup  
2. **make_*_emb*.py** → how embeddings are built  
3. **load_and_push_qdrant.py** → how vectors are stored  
4. **eval_with_qdrant.py** → retrieval + QA logic  
5. **eval_test_only.py** → evaluation flow  
6. **examples.json** → test input/output  
7. (Optional) **Knowledge graph scripts** if interested in KG augmentation  

---

##  License
MIT (recommended; update LICENSE file accordingly)

---

##  Citation
```bibtex
@misc{mbah2025iconquer,
  title={ICONQUER: Biomedical QA with Embeddings and Qdrant},
  author={Mbah, Samuel and Temitayo, Fagbola},
  year={2025},
  url={https://github.com/SamMbah/Biomedical-Question-Answering-System}
}

```

