#!/usr/bin/env bash
set -euo pipefail

# End-to-End Clean Reproducible Training Pipeline (v4.6)
# Solar Filament Segmentation Challenge 2026
#
# Usage:
#   DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026 WORK_DIR=./outputs bash scripts/reproduce_train.sh

DATA_ROOT="${DATA_ROOT:-/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026}"
WORK_DIR="${WORK_DIR:-./outputs}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${WORK_DIR}"

echo "[1/6] Generating 4-fold Stage-1 segmentation (v4 4-ch + v3 3-ch) and 3-ch YOLO training scripts..."
python3 "${REPO_ROOT}/src/make_v4_kernels.py"

echo "[2/6] Training 4-fold 4-Channel Dinkelbach Compromise & Repulsion-Seam Backbones (v4)..."
for SLUG in solar-filament-v4-convnext-small solar-filament-v4-hrnet-w32 solar-filament-v4-regnety-064-fpn; do
  OUT_SUB="${WORK_DIR}/${SLUG}"
  mkdir -p "${OUT_SUB}"
  SF_DATA_DIR="${DATA_ROOT}" SF_OUT_DIR="${OUT_SUB}" \
    python3 "${REPO_ROOT}/src/${SLUG}/train.py"
done

echo "[3/6] Training 4-fold 3-Channel Consensus Support Backbones (v3)..."
for SLUG in solar-filament-v3-effnet-b4 solar-filament-v3-densenet169-upp solar-filament-v3-resnext50; do
  OUT_SUB="${WORK_DIR}/${SLUG}"
  mkdir -p "${OUT_SUB}"
  SF_DATA_DIR="${DATA_ROOT}" SF_OUT_DIR="${OUT_SUB}" \
    python3 "${REPO_ROOT}/src/${SLUG}/train.py"
done

echo "[4/6] Training 4-fold Physical-3Ch Dinkelbach Compromise YOLO11m-Seg (1024px)..."
mkdir -p "${WORK_DIR}/v4_yolo"
SF_DATA_DIR="${DATA_ROOT}" SF_YOLO_OUT="${WORK_DIR}/v4_yolo" \
  python3 "${REPO_ROOT}/src/solar-filament-v4-yolo11m-1024/train.py"

echo "[5/6] Fusing 6-Backbone Channel-Aware OOF Maps + Topological Proposals + Dual-Anchored LightGBM..."
python3 "${REPO_ROOT}/src/fuse_v4_cross_arch.py" \
  --arch-dirs \
    "${WORK_DIR}/solar-filament-v4-convnext-small" \
    "${WORK_DIR}/solar-filament-v4-hrnet-w32" \
    "${WORK_DIR}/solar-filament-v4-regnety-064-fpn" \
    "${WORK_DIR}/solar-filament-v3-effnet-b4" \
    "${WORK_DIR}/solar-filament-v3-densenet169-upp" \
    "${WORK_DIR}/solar-filament-v3-resnext50" \
  --weights 3.2 1.8 1.8 1.2 1.2 0.9 \
  --p0-weights 3.6 2.0 1.8 0.25 0.25 0.20 \
  --yolo-oof "${WORK_DIR}/v4_yolo/oof_yolo.pkl" \
  --coco-json "${DATA_ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json" \
  --fold-json "${REPO_ROOT}/configs/clean_4fold_split.json" \
  --val141-json "${REPO_ROOT}/configs/clean_val_split.json" \
  --out-dir "${WORK_DIR}/v4_stage1_fused"

echo "[6/6] Staging Final Dynamic Offline Inference Bundle..."
python3 "${REPO_ROOT}/src/stage_v4_dataset.py" \
  --model-dirs \
    "${WORK_DIR}/solar-filament-v4-convnext-small" \
    "${WORK_DIR}/solar-filament-v4-hrnet-w32" \
    "${WORK_DIR}/solar-filament-v4-regnety-064-fpn" \
    "${WORK_DIR}/solar-filament-v3-effnet-b4" \
    "${WORK_DIR}/solar-filament-v3-densenet169-upp" \
    "${WORK_DIR}/solar-filament-v3-resnext50" \
  --fuse-dir "${WORK_DIR}/v4_stage1_fused" \
  --yolo-dir "${WORK_DIR}/v4_yolo" \
  --tta-views 4 --v3-tta-views 2 \
  --stage-dir "${WORK_DIR}/v4_models_bundle"

echo "[DONE] Complete reproducible model bundle staged at: ${WORK_DIR}/v4_models_bundle"
