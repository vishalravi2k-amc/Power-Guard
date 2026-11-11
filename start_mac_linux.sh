#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
if [ ! -f .venv/power_guard_ready ]; then
  if [ "$(uname -s)" = "Linux" ]; then
    .venv/bin/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
  else
    .venv/bin/python -m pip install torch==2.15.0
  fi
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/power_guard_ready
fi
exec .venv/bin/python -m streamlit run app.py --server.address localhost
