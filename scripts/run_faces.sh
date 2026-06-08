#!/usr/bin/env bash
#
# run_faces.sh — crop faces from every source's normalized clips into faces/<source>/
# (input to GenD). Skips any source whose processed/ dir is missing.
#
#   ./scripts/run_faces.sh              # device auto-detect (MPS on Mac)
#   DEVICE=cpu ./scripts/run_faces.sh   # force CPU if RetinaFace errors on MPS
set -euo pipefail

ENV_NAME="${ENV_NAME:-GenD}"
DEVICE="${DEVICE:-auto}"                      # auto|mps|cpu|cuda
SOURCES="${SOURCES:-veo31 omniflash ltx23 real}"

# project root = parent of this script's directory (so cwd doesn't matter)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

for src in $SOURCES; do
  in_dir="processed/$src"
  if [ ! -d "$in_dir" ]; then
    echo "skip $src (no $in_dir/)"
    continue
  fi
  echo "=== cropping faces: $src (device=$DEVICE) ==="
  python scripts/crop_faces.py --videos "$in_dir" --out "faces/$src" --device "$DEVICE"
done

echo
echo "done. face crops are in $ROOT/faces/<source>/"
echo "Next:  ./scripts/run_gend.sh"
