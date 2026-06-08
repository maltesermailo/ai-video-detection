#!/usr/bin/env bash
#
# setup_gend.sh — one-time environment setup for running GenD + face cropping.
# Idempotent: safe to re-run (won't recreate the env, pip installs are no-ops
# if already satisfied, repo is only cloned once).
#
# Override defaults via env vars, e.g.:  GEND_DIR=~/code/GenD ./scripts/setup_gend.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-GenD}"
GEND_DIR="${GEND_DIR:-$HOME/GenD}"          # where to clone the GenD repo
GEND_URL="https://github.com/yermandy/GenD"

# --- conda activation (works in a non-interactive script) ------------------
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- 1. create the env if it doesn't exist ---------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' already exists"
else
  echo "creating conda env '$ENV_NAME' (python 3.12)"
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

# --- 2. python dependencies ------------------------------------------------
# Plain pip (not uv) to avoid the 'uv not on defaults channel' problem.
python -m pip install --upgrade pip
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 transformers==4.56.2 \
  retinaface-pytorch opencv-python \
  pandas scikit-learn numpy requests
# NOTE: if pip can't resolve those exact torch pins on Apple Silicon, drop the
# '==' versions and let pip pick the latest compatible torch/torchvision —
# GenD does not require those exact versions.

# --- 3. GenD repo (needed for `import src.hf.modeling_gend`) ---------------
if [ -d "$GEND_DIR/.git" ]; then
  echo "GenD repo already present at $GEND_DIR"
else
  echo "cloning GenD into $GEND_DIR"
  git clone "$GEND_URL" "$GEND_DIR"
fi

echo
echo "setup complete."
echo "  env:       $ENV_NAME"
echo "  GenD repo: $GEND_DIR"
echo "Next:  conda activate $ENV_NAME  &&  ./scripts/run_faces.sh"
echo "Tip: run_gend.sh defaults to GEND_DIR=$GEND_DIR — keep them consistent."
