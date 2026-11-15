#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="${TEST_DIR:-${1:-./samples}}"
OUT_CSV="${OUT_CSV:-${2:-./submission_v3.csv}}"
BUNDLE_DIR="${BUNDLE_DIR:-${3:-./weights}}"

if [ -f "$BUNDLE_DIR/v3_yolo_and_lgbm_bundle.tar.gz" ] && [ ! -f "$BUNDLE_DIR/v3_selector_fold0.txt" ]; then
  tar -xzf "$BUNDLE_DIR/v3_yolo_and_lgbm_bundle.tar.gz" -C "$BUNDLE_DIR"
fi

export SF_TEST_DIR="$TEST_DIR"
export SF_OUT_CSV="$OUT_CSV"
export SF_BUNDLE_DIR="$BUNDLE_DIR"
export SF_ENS_CONFIG="${SF_ENS_CONFIG:-./configs/ensemble_v3_config.json}"

python3 src/inference_v3_panoptic.py
