#!/usr/bin/env bash
#
# run_gend.sh — score every face-crop clip with GenD, then compute metrics.
#
#   ./scripts/run_gend.sh             # score faces/ -> results/gend_scores.csv -> metrics
#   ./scripts/run_gend.sh calibrate   # just verify which logit index is 'fake'
#
# fake_index defaults to 1 (confirmed earlier via --calibrate: index 1 = fake).
set -euo pipefail

ENV_NAME="${ENV_NAME:-GenD}"
GEND_DIR="${GEND_DIR:-$HOME/GenD}"            # must match setup_gend.sh
MODEL_ID="${MODEL_ID:-yermandy/GenD_CLIP_L_14}"
FAKE_INDEX="${FAKE_INDEX:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ ! -d "$GEND_DIR" ]; then
  echo "ERROR: GenD repo not found at $GEND_DIR (run setup_gend.sh or set GEND_DIR)" >&2
  exit 1
fi

# Optional calibration pass: confirms which output column means 'fake'.
if [ "${1:-}" = "calibrate" ]; then
  python scripts/gend_score.py --repo "$GEND_DIR" --model_id "$MODEL_ID" --calibrate
  exit 0
fi

mkdir -p results

echo "=== scoring with GenD (model=$MODEL_ID, fake_index=$FAKE_INDEX) ==="
python scripts/gend_score.py \
  --repo "$GEND_DIR" --model_id "$MODEL_ID" \
  --frames_root faces --manifest metadata/manifest.csv \
  --out results/gend_scores.csv --fake_index "$FAKE_INDEX"

echo
echo "=== computing detection metrics ==="
python scripts/detection_metrics.py --input results/gend_scores.csv --score_col gend_fake_score

echo
echo "done."
echo "  per-clip scores: results/gend_scores.csv"
echo "  metrics:         results/gend_scores_metrics.csv"
echo "  distribution:    results/gend_scores_distribution.csv"
