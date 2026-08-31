#!/usr/bin/env bash
# SIH26017 bootstrap: fresh clone -> working demo in one command.
#
# Usage:
#   bash scripts/bootstrap.sh             # full setup, then launch the dashboard
#   bash scripts/bootstrap.sh --no-launch # setup only (headless / CI / e2e test)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAUNCH=1
if [[ "${1:-}" == "--no-launch" ]]; then
  LAUNCH=0
fi

# resolve uv (the only system-level dependency we assume; everything else is in .venv)
if command -v uv >/dev/null 2>&1; then
  UV="uv"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV="$HOME/.local/bin/uv"
else
  echo "ERROR: 'uv' not found. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# 1. virtualenv
if [[ ! -d .venv ]]; then
  echo "[1/5] Creating virtualenv ..."
  "$UV" venv .venv
fi

# 2. dependencies
echo "[2/5] Installing dependencies ..."
"$UV" pip install --python .venv/bin/python -r requirements.txt

# 3. synthetic data
echo "[3/5] Generating data ..."
.venv/bin/python src/data_generator.py

# 4. models (+ LODO cold-start + SHAP + metrics report)
echo "[4/5] Training models ..."
.venv/bin/python src/train.py

# 5. live portfolio cache
echo "[5/5] Scoring live portfolio ..."
.venv/bin/python src/predict.py --refresh-portfolio

echo "Bootstrap complete. Models + portfolio ready."

if [[ "$LAUNCH" == "1" ]]; then
  echo "Launching dashboard at http://localhost:8501"
  exec .venv/bin/streamlit run app/streamlit_app.py
fi
