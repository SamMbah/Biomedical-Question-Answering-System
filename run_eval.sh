#!/usr/bin/env bash
set -euo pipefail

# 0) cd into project root (this script's dir)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) sanity-check we're in the right conda env
if [[ "${CONDA_DEFAULT_ENV-}" != "medqa-gpu" ]]; then
  echo "❗ Please: conda activate medqa-gpu" >&2
  exit 1
fi

# 2) Load secrets from your environment (or optional .env)
#    Never hardcode keys in this file.
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your shell or .env before running}"

# 3) restart Qdrant
docker rm -f qdrant >/dev/null 2>&1 || true
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# 4) inspect emb_w2v_ft.pkl (optional sanity check)
echo "=== Inspecting emb_w2v_ft.pkl ==="
python - << 'PYCODE'
import pickle, pandas as pd
df = pickle.load(open("emb_w2v_ft.pkl","rb"))
print("Type:", type(df))
if isinstance(df, dict):
    print("Dict keys:", list(df.keys())[:10])
elif hasattr(df, "columns"):
    print("DF columns:", list(df.columns))
else:
    print("Repr:", repr(df)[:500])
PYCODE
echo "=== End inspect ==="

# 5) run the eval
echo "=== Running eval_with_qdrant.py ==="
python eval_with_qdrant.py
