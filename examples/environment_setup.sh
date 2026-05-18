#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-unit}"
# Default: PyPI. Set PIP_INDEX_URL to a regional mirror if downloads are slow.
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not on PATH. Install Miniconda/Anaconda or source conda.sh first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

env_exists() {
  conda env list | awk -v n="$CONDA_ENV_NAME" '$1 == n { found=1 } END { exit !found }'
}

if env_exists; then
  echo "Conda env '$CONDA_ENV_NAME' already exists; skip conda create."
else
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

conda activate "$CONDA_ENV_NAME"

pip install -i "$PIP_INDEX_URL" --upgrade setuptools
cd "$PROJECT_ROOT"
pip install -i "$PIP_INDEX_URL" einx
pip install -i "$PIP_INDEX_URL" -e ".[base]"
pip install -i "$PIP_INDEX_URL" --no-build-isolation "flash-attn==2.7.1.post4"
# pip install -i "$PIP_INDEX_URL" transformers==4.52.0
pip install -i "$PIP_INDEX_URL" "qwen-vl-utils[decord]==0.0.8"
pip install -i "$PIP_INDEX_URL" lpips

echo "Done. Activate with: conda activate $CONDA_ENV_NAME"
