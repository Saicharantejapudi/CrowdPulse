#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Creating venv + installing deps for local (CPU) use"
[ -d venv ] || "$PYTHON" -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install torch -r requirements.txt

echo "==> Done. Model weights + patched PET are already vendored (via git-lfs)."
echo "    Run: ./venv/bin/python count.py path/to/image.jpg"
