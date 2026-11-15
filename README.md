# Solar Filament Segmentation Challenge 2026 — Reproducible Panoptic Pipeline

**Author:** Amit Gunjan (`daydreamerthinks`)  
**Affiliation:** Nalanda University, Rajgir, India (`amit.202614029@nalandauniv.edu.in`)

This repository contains the reproducible machine learning pipeline and stored model weights for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). The pipeline trains strictly on the official `MAGFiLO_1.0_Kaggle_2026` training split (`707` stems). The dynamic inference engine runs offline (`enable_internet: false`) and produces pixel-disjoint filament masks (`0` overlap pixels).

> [!IMPORTANT]
> Always keep observations from the same solar calendar day inside the same fold (`configs/clean_4fold_split.json`) to prevent temporal leakage across observatories.

---

## 1. Quick Start: Interactive Notebook and Pretrained Weights

Open and run `pipeline_reproduction.ipynb` to reproduce predictions directly from stored model weights without re-training:

1. **Automatic Stored Weight Loader:** Cell 1 reads `configs/weights_manifest.json` and downloads `28` `.pt` checkpoints plus `48` boosters from [GitHub Release `v1.0.0`](https://github.com/DayDreamer-commits/solar-filament-segmentation-2026/releases/tag/v1.0.0). Next, the loader verifies every `SHA-256` digest inside `./weights`.
2. **Out-of-the-Box Sample Execution:** Cells 2–5 normalize `samples/20140406195854Bh.jpeg` and build Dinkelbach targets from `samples/sample_annotations_20140406195854Bh.json`. Next, `src/inference_v4_panoptic.py` generates disjoint masks (`0` overlap) and computes official Panoptic Quality (`PQ = SQ * RQ`).

---

## 2. Architecture Summary

1. **Physical 3-Channel Solar Disk Normalization:** The preprocessor converts `2048 x 2048` H-alpha images into disk-percentile intensity, `16 x 16` CLAHE contrast, and `sigma = 18 px` limb-flattened absorption maps.
2. **Dinkelbach Compromise Targets and Repulsion Seams:** The target builder computes signed-distance-field (`SDF`) median contours across `K` human annotators (`vals >= 0.35 * K`) and `5 px` repulsion seams (`p3`).
3. **Channel-Aware 6-Backbone Streaming Ensemble:** The engine streams three 4-channel `v4` backbones (`ConvNeXt-Small`, `HRNet-W32`, `RegNetY-064`) and three 3-channel `v3` backbones (`EfficientNet-B4`, `DenseNet-169`, `SE-ResNeXt50`).
4. **3-Channel Dinkelbach `YOLO11m-Seg` (`1024 px`):** Four fold checkpoints run on normalized 3-channel solar disks (`0.4031` standalone Out-Of-Fold `PQ`) and supply calibrated consensus proposals.
5. **Unshaved Topological Proposals and Dual-Anchored LightGBM:** The proposal engine extracts crisp seam-cut, watershed, hysteresis, YOLO-bridged, and tip-tangent candidates (`[0, 1, 2, 3, 6, 7, 10]`). Two bagged 24-booster LightGBM ensembles (`0.60 * YOLO_GBDT + 0.40 * SEG_GBDT`) score `60` geometric, neural-ambiguity, and relational features without single-feature over-reliance.
6. **Exact Connected-Component `MWIS` Packing:** A branch-and-bound Maximum-Weight Independent Set (`MWIS`) solver packs conflict components (`cut = 0.22`, `split_cost = 0.24`) and guarantees `0` overlapping pixels.

---

## 3. Repository Structure

| Path | Purpose |
|---|---|
| `pipeline_reproduction.ipynb` | Executed Jupyter notebook illustrating all pipeline stages with stored model weights |
| `requirements.txt` | Pinned Python package versions for reproducible training and inference |
| `configs/weights_manifest.json` | Release `v1.0.0` URLs and `SHA-256` checksums for all `28` `.pt` weights and `48` boosters |
| `configs/ensemble_v4_config.json` | Channel-aware ensemble weights, topological features, and `MWIS` thresholds |
| `configs/clean_4fold_split.json` | Deterministic solar-day-grouped 4-fold split across all `707` training stems |
| `configs/clean_val_split.json` | Deterministic `141`-stem holdout validation subset |
| `samples/` | Bundled GONG H-alpha disk `20140406195854Bh.jpeg` and `K = 3` human annotation subset |
| `src/make_v4_kernels.py` | Generates all 4-fold `v4`, `v3`, and 3-channel `YOLO11m-Seg` training scripts |
| `src/train_v4_panoptic_template.py` | 4-fold 4-channel Dinkelbach compromise and repulsion-seam segmentation trainer |
| `src/train_v3_panoptic_template.py` | 4-fold 3-channel consensus support segmentation trainer |
| `src/train_v4_dinkelbach_yolo.py` | 4-fold 3-channel Dinkelbach `YOLO11m-Seg` (`1024 px`) instance trainer |
| `src/fuse_v4_cross_arch.py` | 6-backbone Out-Of-Fold fusion, topological proposals, Dual-Anchored LightGBM, and `MWIS` |
| `src/stage_v4_dataset.py` | Packages checkpoints, `48` LightGBM boosters, and JSON metadata into an offline bundle |
| `src/inference_v4_panoptic.py` | Multi-GPU and CPU dynamic inference engine with process-isolated scoring and `0`-overlap check |
| `scripts/reproduce_infer.sh` | Single-command dynamic test inference using stored weights and pixel-disjointness verifier |
| `scripts/reproduce_train.sh` | Single-command end-to-end 4-fold training and bundle staging script |
| `scripts/evaluate_official_pq.py` | Official Panoptic Quality (`PQ = SQ * RQ`) evaluator and diagnostic plotter |
| `scripts/plot_pipeline_architecture.py` | Generates the vector architecture flowchart (`Figure 1`) for the technical report |
| `scripts/plot_progression_ablation.py` | Generates the vector ablation progression figure for the technical report |
| `technical_report/` | Official ACM `sigconf` LaTeX source (`main.tex`), BibTeX (`main.bib`), figures, and 4-page PDF |

---

## 4. Environment Setup

1. Create a clean Python `3.10+` virtual environment.
2. Activate the virtual environment.
3. Install the pinned packages from `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

* **Hardware Requirements:**
  * **Training:** `1x` or `2x` NVIDIA GPU with `>= 16 GB` VRAM (`Tesla P100`, `T4`, `L4`, or `A100`) and `32 GB` system RAM.
  * **Inference:** `2x` NVIDIA `T4` GPU (`54.2 min` for all `28` checkpoints) or `4`-vCPU host (`< 56 min` in YOLO-ROI CPU mode) with `< 3.5 GB` RAM.

---

## 5. Reproduce Dynamic Test Inference Without Re-Training

1. Run Cell 1 of `pipeline_reproduction.ipynb` to download and verify stored weights in `./weights`.
2. Run `scripts/reproduce_infer.sh` to reproduce the `v4.6` 28-model ensemble (`Ref 58364478`).
3. Run `scripts/reproduce_infer_v3.sh` to reproduce the `v3` 4-family ensemble (`Ref 58250373`).

```bash
# Reproduce v4.6 28-model submission (Ref 58364478)
BUNDLE_DIR=./weights \
TEST_DIR=/path/to/MAGFiLO_1.0_Kaggle_2026/test/test_images \
OUT_CSV=./submission.csv \
bash scripts/reproduce_infer.sh

# Reproduce v3 4-family submission (Ref 58250373)
BUNDLE_DIR=./weights \
TEST_DIR=/path/to/MAGFiLO_1.0_Kaggle_2026/test/test_images \
OUT_CSV=./submission_v3.csv \
bash scripts/reproduce_infer_v3.sh
```

Both inference scripts detect `1` or `2` GPUs automatically and stream checkpoints across chunks of `24` images. Next, each post-processor selects pixel-disjoint filament masks and verifies `0` overlapping pixels in the output CSV.

---

## 6. Reproduce Full 4-Fold Training End-to-End

1. Download the official competition dataset `MAGFiLO_1.0_Kaggle_2026`.
2. Run `scripts/reproduce_train.sh` with `DATA_ROOT` set to `MAGFiLO_1.0_Kaggle_2026`.

```bash
DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026 \
WORK_DIR=./outputs \
bash scripts/reproduce_train.sh
```

The training script builds all 4-fold `v4`, `v3`, and `YOLO11m-Seg` checkpoints. Next, `src/fuse_v4_cross_arch.py` fuses Out-Of-Fold probability maps and trains the Dual-Anchored LightGBM rankers. Finally, `src/stage_v4_dataset.py` writes the self-contained model bundle to `./outputs/v4_models_bundle`.

---

## 7. Official Metric Evaluation

1. Run `scripts/evaluate_official_pq.py` to compute official Panoptic Quality (`IoU > 0.50`) and export diagnostic plots.

```bash
python3 scripts/evaluate_official_pq.py \
  --coco-json /path/to/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --pred-csv ./oof_submission.csv \
  --out-dir ./eval_diagnostics
```
