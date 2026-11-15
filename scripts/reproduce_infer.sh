#!/usr/bin/env bash
set -euo pipefail

# Dynamic Multi-GPU Panoptic Inference & Pixel-Disjointness Verification
# Solar Filament Segmentation Challenge 2026
#
# Usage:
#   BUNDLE_DIR=./outputs/v4_models_bundle TEST_DIR=/path/to/test_images OUT_CSV=./submission.csv bash scripts/reproduce_infer.sh

BUNDLE_DIR="${BUNDLE_DIR:-./weights}"
TEST_DIR="${TEST_DIR:-/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images}"
OUT_CSV="${OUT_CSV:-./submission.csv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${BUNDLE_DIR}/ensemble_v4_config.json" ]]; then
  CFG_PATH="${BUNDLE_DIR}/ensemble_v4_config.json"
else
  CFG_PATH="${BUNDLE_DIR}/ensemble_config.json"
fi

echo "[1/2] Running dynamic panoptic inference using ${CFG_PATH}..."
SF_ENS_CONFIG="${CFG_PATH}" \
SF_TEST_DIR="${TEST_DIR}" \
SF_OUT_CSV="${OUT_CSV}" \
  python3 "${REPO_ROOT}/src/inference_v4_panoptic.py"

echo "[2/2] Verifying strict pixel-disjointness (0 overlap pixels)..."
python3 - "${OUT_CSV}" <<'PY'
import sys
import pandas as pd
from pycocotools import mask as mu

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
assert list(df.columns) == ["filament_id", "segmentation_rle"], f"Unexpected columns: {list(df.columns)}"
df["stem"] = df["filament_id"].map(lambda s: str(s).rsplit("_", 1)[0])
overlap_px = 0
for _, grp in df.groupby("stem", sort=False):
    rles = [{"size": [2048, 2048], "counts": str(c).encode("ascii")} for c in grp["segmentation_rle"]]
    if rles:
        overlap_px += int(mu.area(rles).sum()) - int(mu.area(mu.merge(rles)))
assert overlap_px == 0, f"Found {overlap_px} overlapping pixels!"
print(f"[VERIFIED] {csv_path}: {len(df)} instances across {df['stem'].nunique()} solar disks, 0 overlapping pixels.")
PY
