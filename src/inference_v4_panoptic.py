"""
Solar Filament Segmentation 2026 - v4.6 Cross-Generation + 3-Ch Dinkelbach YOLO + Exact MWIS Panoptic Inference.

100% dynamic runtime neural inference (`enable_gpu: true`, `enable_internet: false`):
  1. Multi-GPU Process Isolation: when `nvidia-smi -L` detects `>= 2` GPUs (2x T4), spawns 2
     isolated OS subprocesses (`CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`),
     halving wall-clock runtime with zero shared CUDA/cuDNN state.
  2. O(1)-Memory Channel-Aware Welford Streaming (`CHUNK_SIZE = 24`):
     - `p0` (Dinkelbach silhouette) weighted by `p0_weight` to preserve crisp boundaries (`SQ ~ 0.688`).
     - `p1, p2` (consensus & interior distance spine) weighted across all 6 diverse backbones (`weight`).
     - `p3` (5px repulsion seam) weighted across 4-ch `v4` backbones.
     - `p0_v3` (Tversky recall map) tracked from 3-ch `v3` families to expose neural multi-annotator
       ambiguity features (`p0_v3_mean`, `p0_v3_gap`).
  3. Calibrated 4-Fold 3-Ch `YOLO11m-Seg` Consensus (`ymat >= 0.35`, `w_tot / sqrt(k * M)` calibration).
  4. Unshaved Topological Proposals (`src = 0..10`) + 60 Relational Features + Bagged LightGBM +
     Exact Dinkelbach `MWIS` Disjoint Set Packing (`cut` + `split_cost`, `overlap_total == 0`).
"""

from __future__ import annotations

import gc
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

H = W = 2048
CHUNK_SIZE = 24
T0 = time.time()


def log(*args) -> None:
    sh = os.environ.get("SF_WORKER_SHARD", "main")
    print(f"[{(time.time() - T0) / 60:6.2f}m][{sh}]", *args, flush=True)


def install_offline_wheels() -> None:
    wheels = glob.glob("/kaggle/input/**/segmentation_models_pytorch-*.whl", recursive=True)
    if wheels and importlib.util.find_spec("segmentation_models_pytorch") is None:
        wdir = str(Path(wheels[0]).parent)
        if importlib.util.find_spec("timm") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", wdir, "timm"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", wdir, "segmentation-models-pytorch"])
    uwheels = glob.glob("/kaggle/input/**/ultralytics-*.whl", recursive=True)
    if uwheels and importlib.util.find_spec("ultralytics") is None:
        uwdir = str(Path(uwheels[0]).parent)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", uwdir, "ultralytics"])


def detect_hw_gpus() -> int:
    if os.environ.get("SF_FORCE_2SHARD_TEST") == "1":
        return 2
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return len([ln for ln in smi.stdout.strip().splitlines() if ln.strip()]) if smi.returncode == 0 else 0
    except FileNotFoundError:
        return 0


def cpu_has_fast_bf16() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
        return ("amx_bf16" in flags) or ("avx512_bf16" in flags)
    except OSError:
        return False


def locate_test_images() -> list[Path]:
    env_test = os.environ.get("SF_TEST_DIR")
    if env_test:
        d = Path(env_test)
        return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    for d in sorted(Path("/kaggle/input").rglob("test_images")):
        if d.is_dir():
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
            if len(imgs) > 0:
                return imgs
    raise FileNotFoundError("Could not locate test_images directory")


def run_lgb_score_subproc(in_npz: str, out_npy: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import lightgbm as lgb
    import numpy as np

    d = np.load(in_npz, allow_pickle=False)
    X_s1 = d["X_s1"]
    X_seg = d["X_seg"]
    sel_paths = [str(p) for p in d["sel_paths"]]
    seg_sel_paths = [str(p) for p in d["seg_sel_paths"]]
    alpha_g = float(d["alpha_g"][0])

    p_y = sum(lgb.Booster(model_file=p).predict(X_s1, num_threads=1) for p in sel_paths) / float(len(sel_paths))
    if seg_sel_paths and X_seg.shape[1] > 0:
        p_s = sum(lgb.Booster(model_file=p).predict(X_seg, num_threads=1) for p in seg_sel_paths) / float(len(seg_sel_paths))
        preds = alpha_g * p_y + (1.0 - alpha_g) * p_s
    else:
        preds = p_y
    np.save(out_npy, np.asarray(preds, dtype=np.float64))


def run_cpu_calibrate_subproc(out_cfg_json: str) -> None:
    n_cal_thr = "1" if sys.platform == "darwin" else "2"
    os.environ["OMP_NUM_THREADS"] = n_cal_thr
    os.environ["MKL_NUM_THREADS"] = n_cal_thr
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import cv2
    import numpy as np
    import torch
    import segmentation_models_pytorch as smp

    cv2.setNumThreads(1)
    torch.set_num_threads(int(n_cal_thr))
    cpu_bf16_ok = cpu_has_fast_bf16()

    cfg_path = os.environ.get("SF_ENS_CONFIG")
    if not cfg_path:
        hits = sorted(
            glob.glob("/kaggle/input/**/ensemble_v4_config.json", recursive=True)
            + glob.glob("/kaggle/input/**/ensemble_config.json", recursive=True),
            key=lambda p: (0 if "ensemble_v4_config.json" in p else (1 if "v4-panoptic-models" in p else 2), p),
        )
        if not hits:
            raise FileNotFoundError("Could not locate ensemble_config.json in /kaggle/input")
        cfg_path = hits[0]
    ens_cfg = json.loads(Path(cfg_path).read_text())
    bundle_dir = Path(ens_cfg.get("orig_bundle_dir") or Path(cfg_path).parent)

    asset_index: dict[str, Path] = {}
    in_root = Path("/kaggle/input")
    if in_root.exists():
        for p_in in sorted(in_root.rglob("*")):
            if p_in.is_file():
                if p_in.name not in asset_index or "v4" in str(p_in):
                    asset_index[p_in.name] = p_in

    def resolve_asset(rel_name: str) -> Path:
        cand = bundle_dir / rel_name
        if cand.exists():
            return cand
        if rel_name in asset_index:
            return asset_index[rel_name]
        raise FileNotFoundError(f"Missing model asset {rel_name} in {bundle_dir} and /kaggle/input")

    test_files = locate_test_images()
    n_sh_tot = 2 if (len(test_files) >= 2 and (sys.platform != "darwin" or os.environ.get("SF_FORCE_CPU_2SHARD_TEST") == "1")) else 1
    n_shard = max(1, math.ceil(len(test_files) / float(n_sh_tot)))
    smoke_fast = os.environ.get("SF_SMOKE_FAST") == "1"

    fold_prio_map = {
        "convnext": [0, 1, 2, 3],
        "hrnet": [0, 1, 3, 2],
        "regnety": [1, 0, 2, 3],
        "efficientnet": [2, 0, 1, 3],
        "resnext": [3, 1, 0, 2],
    }

    def ordered_ckpts(backbone_str: str, ck_list: list[str]) -> list[str]:
        bb_low = backbone_str.lower()
        for key, idxs in fold_prio_map.items():
            if key in bb_low:
                return [ck_list[idx] for idx in idxs if idx < len(ck_list)]
        return list(ck_list)

    raw_groups = [g for g in ens_cfg["arch_groups"] if g["arch"] != "UnetPlusPlus"]
    yolo_rels = [yrel for yrel in ens_cfg.get("yolo_ckpts", []) if (bundle_dir / yrel).exists() or yrel in asset_index]
    yolo_imgsz = int(ens_cfg.get("yolo_imgsz", 1024))

    if smoke_fast or len(test_files) <= 4:
        cal_groups = []
        target_keys = ("regnety", "resnext", "hrnet") if smoke_fast else ("convnext", "hrnet", "regnety", "efficientnet")
        for g in raw_groups:
            if any(k in g["backbone"].lower() for k in target_keys):
                g_copy = dict(g)
                g_copy["ckpts"] = ordered_ckpts(g["backbone"], g["ckpts"])[:1]
                g_copy["tta_views"] = 1
                cal_groups.append(g_copy)
        out_cfg = dict(ens_cfg)
        out_cfg["orig_bundle_dir"] = str(bundle_dir)
        out_cfg["arch_groups"] = cal_groups
        out_cfg["yolo_ckpts"] = yolo_rels[: (1 if smoke_fast else min(2, len(yolo_rels)))]
        out_cfg["tta_views"] = 1
        out_cfg["best_cut"] = 0.22
        out_cfg["split_cost"] = 0.24
        Path(out_cfg_json).write_text(json.dumps(out_cfg, indent=2))
        log(f"CPU ROI fast calibration (n_imgs={len(test_files)}) selected {len(cal_groups)} families -> {out_cfg_json}")
        return

    img0 = cv2.imread(str(test_files[0]), cv2.IMREAD_GRAYSCALE)
    if img0 is None or img0.shape != (H, W):
        img0 = np.full((H, W), 128, dtype=np.uint8)
    x_warm = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    x_bench = torch.zeros((1, 3, 768, 1024), dtype=torch.float32)
    contention_factor = 1.78 if n_sh_tot >= 2 else 1.05
    roi_scale_vs_bench = 1.25  # (768*1024)/(2048*2048) = 0.1875 DRAM-realistic ROI budget + 25% loop/feather overhead
    max_cpu_ckpts = int(os.environ.get("SF_MAX_CPU_CKPTS", "4"))

    dt_yolo = 0.0
    n_yolo = 0
    if yolo_rels and importlib.util.find_spec("ultralytics") is not None:
        from ultralytics import YOLO
        ym = YOLO(str(resolve_asset(yolo_rels[0])))
        y_dummy = np.stack([img0, img0, img0], axis=-1)
        _ = ym.predict(y_dummy, imgsz=yolo_imgsz, conf=0.05, device="cpu", verbose=False)
        t_y0 = time.time()
        _ = ym.predict(y_dummy, imgsz=yolo_imgsz, conf=0.05, device="cpu", verbose=False)
        dt_yolo = max(0.05, time.time() - t_y0)
        del ym
        gc.collect()
        if n_shard * 3 * dt_yolo * contention_factor <= 500.0:
            n_yolo = min(3, len(yolo_rels))
        elif n_shard * 2 * dt_yolo * contention_factor <= 500.0:
            n_yolo = min(2, len(yolo_rels))
        else:
            n_yolo = 1

    def fam_quality_rank(backbone_str: str) -> int:
        bb_l = backbone_str.lower()
        if "convnext" in bb_l:
            return 0
        if "hrnet" in bb_l:
            return 1
        if "regnety" in bb_l:
            return 2
        if "efficientnet" in bb_l:
            return 3
        if "resnext" in bb_l:
            return 4
        return 9

    fam_infos = []
    for idx_g, g in enumerate(raw_groups):
        ck0 = resolve_asset(g["ckpts"][0])
        sd = torch.load(str(ck0), map_location="cpu", weights_only=True)
        n_out_ch = int(sd["segmentation_head.0.weight"].shape[0])
        m = getattr(smp, g["arch"])(encoder_name=g["backbone"], encoder_weights=None, in_channels=3, classes=n_out_ch, activation=None).eval()
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
        del sd
        with torch.no_grad():
            if cpu_bf16_ok:
                with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                    _ = m(x_warm)
                t_m0 = time.time()
                with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                    _ = m(x_bench)
            else:
                _ = m(x_warm)
                t_m0 = time.time()
                _ = m(x_bench)
            dt_640 = max(0.02, time.time() - t_m0)
        del m
        gc.collect()
        est_ckpt_sec = n_shard * dt_640 * roi_scale_vs_bench * contention_factor
        fam_infos.append({
            "idx": idx_g,
            "qrank": fam_quality_rank(g["backbone"]),
            "grp": g,
            "is_v4": (n_out_ch >= 4),
            "dt": dt_640,
            "cost": est_ckpt_sec,
            "ordered_ckpts": ordered_ckpts(g["backbone"], g["ckpts"]),
            "picked": [],
        })
        log(f"CPU ROI bench {g['arch']}/{g['backbone']} (768x1024={dt_640:.2f}s, bf16={cpu_bf16_ok}) -> {est_ckpt_sec / 60.0:.2f}m/ckpt")

    target_sec = float(os.environ.get("SF_CPU_BUDGET_MIN", "47.0")) * 60.0
    fixed_sec = 90.0 + n_shard * (n_yolo * dt_yolo * contention_factor + 0.30)
    seg_budget_sec = max(600.0, target_sec - fixed_sec)
    used_sec = 0.0
    picked_tot = 0

    v4_fams = sorted([fi for fi in fam_infos if fi["is_v4"]], key=lambda d: d["qrank"])
    v3_fams = sorted([fi for fi in fam_infos if not fi["is_v4"]], key=lambda d: d["qrank"])
    if v4_fams and picked_tot < max_cpu_ckpts:
        v4_fams[0]["picked"].append(v4_fams[0]["ordered_ckpts"][0])
        used_sec += v4_fams[0]["cost"]
        picked_tot += 1
    if v3_fams and picked_tot < max_cpu_ckpts:
        v3_fams[0]["picked"].append(v3_fams[0]["ordered_ckpts"][0])
        used_sec += v3_fams[0]["cost"]
        picked_tot += 1

    fam_by_q = sorted(fam_infos, key=lambda d: d["qrank"])
    for r_fold in range(4):
        for fi in fam_by_q:
            if picked_tot >= max_cpu_ckpts:
                break
            if len(fi["picked"]) > r_fold or r_fold >= len(fi["ordered_ckpts"]):
                continue
            cand_ck = fi["ordered_ckpts"][r_fold]
            if cand_ck in fi["picked"]:
                continue
            if used_sec + fi["cost"] <= seg_budget_sec:
                fi["picked"].append(cand_ck)
                used_sec += fi["cost"]
                picked_tot += 1

    active_fams = [fi for fi in fam_infos if fi["picked"]]
    if len(active_fams) < 2 and len(fam_infos) >= 2:
        rem = sorted([fi for fi in fam_infos if not fi["picked"]], key=lambda d: d["cost"])
        if rem:
            rem[0]["picked"].append(rem[0]["ordered_ckpts"][0])
            used_sec += rem[0]["cost"]

    cal_groups = []
    for fi in fam_infos:
        if fi["picked"]:
            gc_item = dict(fi["grp"])
            gc_item["ckpts"] = list(fi["picked"])
            gc_item["tta_views"] = 1
            cal_groups.append(gc_item)

    out_cfg = dict(ens_cfg)
    out_cfg["orig_bundle_dir"] = str(bundle_dir)
    out_cfg["arch_groups"] = cal_groups
    out_cfg["yolo_ckpts"] = yolo_rels[:n_yolo]
    out_cfg["tta_views"] = 1
    out_cfg["best_cut"] = 0.22
    out_cfg["split_cost"] = 0.24
    Path(out_cfg_json).write_text(json.dumps(out_cfg, indent=2))
    tot_ck = sum(len(g["ckpts"]) for g in cal_groups)
    log(
        f"CPU ROI calibration selected {len(cal_groups)} families ({tot_ck} seg ckpts + {n_yolo} YOLO ckpts, cut=0.22/split=0.24); "
        f"projected shard wall-clock = {(fixed_sec + used_sec) / 60.0:.1f}m (shards={n_sh_tot})"
    )


def run_worker_shard() -> None:
    n_hw_g = detect_hw_gpus()
    sh_env = os.environ.get("SF_WORKER_SHARD", "")
    if n_hw_g >= 1 or sys.platform == "darwin":
        n_thr_str = "1"
    elif sh_env and "/" in sh_env:
        n_thr_str = "2"
    else:
        n_thr_str = str(max(1, os.cpu_count() or 4))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OMP_NUM_THREADS", n_thr_str)
    os.environ.setdefault("MKL_NUM_THREADS", n_thr_str)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import cv2
    import numpy as np
    import pandas as pd
    from pycocotools import mask as mu
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp

    cv2.setNumThreads(1)

    def compute_disk_stats(img_u8: np.ndarray, disk_mask: np.ndarray) -> list[float]:
        vals = img_u8[disk_mask]
        if vals.size == 0:
            return [64.0, 128.0, 192.0]
        p10, p50, p90 = np.percentile(vals, (10, 50, 90))
        return [float(p10), float(p50), float(p90)]

    def encode_3ch_u8(img_u8: np.ndarray, stats: list[float]) -> np.ndarray:
        p10, p50, p90 = stats
        span = max(p90 - p10, 16.0)
        img_f = img_u8.astype(np.float32)
        ch0 = np.clip((img_f - p50) / span * 0.5 + 0.5, 0.0, 1.0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(img_u8)
        ch1 = clahe.astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(img_u8, (0, 0), 18.0).astype(np.float32)
        ch2 = np.clip((blur - img_f) / span + 0.5, 0.0, 1.0)
        return np.rint(255.0 * np.stack([ch0, ch1, ch2], axis=0)).clip(0, 255).astype(np.uint8)

    def encode_crop_rle(crop_mask: np.ndarray, x0: int, y0: int):
        yy, xx = np.nonzero(crop_mask.T)
        if len(xx) == 0:
            return None
        idx = (yy.astype(np.int64) + x0) * H + (xx.astype(np.int64) + y0)
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks] + 1, idx[-1] + 1]
        counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
        counts[0] = starts[0]
        counts[1:-1:2] = ends - starts
        counts[2:-1:2] = starts[1:] - ends[:-1]
        counts[-1] = H * W - ends[-1]
        return mu.frPyObjects({"size": [H, W], "counts": counts.tolist()}, H, W)

    def fill_holes_crop(crop_mask: np.ndarray) -> np.ndarray:
        if not np.any(crop_mask):
            return crop_mask.copy()
        padded = np.pad(np.ascontiguousarray(crop_mask, dtype=np.uint8), 1, mode="constant", constant_values=0)
        cnts, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(padded)
        cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
        return filled[1:-1, 1:-1]

    def calibrate_crop_mask(mc: np.ndarray, p1_crop: np.ndarray, thr: float = 0.44, min_ret: float = 0.72) -> np.ndarray:
        a0 = int(mc.sum())
        if a0 < 60:
            return mc
        sub = ((mc > 0) & (p1_crop >= thr)).astype(np.uint8)
        sub = fill_holes_crop(sub) & mc
        ncc, lab, st_c, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        clean = np.zeros_like(sub)
        for ck in range(1, ncc):
            if st_c[ck, cv2.CC_STAT_AREA] >= 25:
                clean[lab == ck] = 1
        a1 = int(clean.sum())
        if a1 >= 60 and a1 >= min_ret * a0:
            return clean
        return mc

    def extract_component_tips(mc: np.ndarray, bx: int, by: int) -> tuple[list[tuple[float, float, float, float]], float] | None:
        ys, xs = np.nonzero(mc)
        n_px = len(xs)
        if n_px < 70:
            return None
        xf = xs.astype(np.float64) + bx
        yf = ys.astype(np.float64) + by
        cx, cy = float(xf.mean()), float(yf.mean())
        dx, dy = xf - cx, yf - cy
        cov_xx = float(np.mean(dx * dx)) + 1e-6
        cov_yy = float(np.mean(dy * dy)) + 1e-6
        cov_xy = float(np.mean(dx * dy))
        tr = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        disc = math.sqrt(max(0.0, 0.25 * tr * tr - det))
        lam1 = 0.5 * tr + disc
        lam2 = max(1.0, 0.5 * tr - disc)
        if lam1 / lam2 < 2.2:
            return None
        if abs(cov_xy) > 1e-6:
            ux, uy = lam1 - cov_yy, cov_xy
        elif cov_xx >= cov_yy:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = 0.0, 1.0
        unorm = math.hypot(ux, uy)
        ux, uy = ux / unorm, uy / unorm

        proj = dx * ux + dy * uy
        p_min, p_max = float(proj.min()), float(proj.max())
        span = max(p_max - p_min, 1.0)
        stroke_w = float(np.clip(n_px / span, 3.0, 14.0))

        tips = []
        for sign, thr_tip, thr_seg in [
            (-1.0, p_min + 0.12 * span, p_min + 0.32 * span),
            (1.0, p_max - 0.12 * span, p_max - 0.32 * span),
        ]:
            if sign < 0:
                m_tip = proj <= thr_tip
                m_seg = proj <= thr_seg
            else:
                m_tip = proj >= thr_tip
                m_seg = proj >= thr_seg
            if int(m_tip.sum()) < 3 or int(m_seg.sum()) < 8:
                continue
            tx_pt, ty_pt = float(xf[m_tip].mean()), float(yf[m_tip].mean())
            sx_m, sy_m = float(xf[m_seg].mean()), float(yf[m_seg].mean())
            vx, vy = tx_pt - sx_m, ty_pt - sy_m
            vnorm = math.hypot(vx, vy)
            if vnorm > 1.5:
                tx_dir, ty_dir = vx / vnorm, vy / vnorm
            else:
                tx_dir, ty_dir = sign * ux, sign * uy
            tips.append((tx_pt, ty_pt, tx_dir, ty_dir))
        if len(tips) < 2:
            return None
        return tips, stroke_w

    def compute_stem_relational_features(rows: list[dict]) -> None:
        m = len(rows)
        if m == 0:
            return
        rles = [r["rle"] for r in rows]
        areas = np.asarray([r["area"] for r in rows], dtype=np.float64)
        p0s = np.asarray([r["p0_mean"] for r in rows], dtype=np.float64)
        p1s = np.asarray([r["p1_mean"] for r in rows], dtype=np.float64)
        srcs = np.asarray([r["source"] for r in rows], dtype=np.int32)
        bbs = np.asarray([mu.toBbox(r) for r in rles], dtype=np.float64)
        x0, y0 = bbs[:, 0], bbs[:, 1]
        x1, y1 = bbs[:, 0] + bbs[:, 2], bbs[:, 1] + bbs[:, 3]

        iou_mat = np.asarray(mu.iou(rles, rles, [0] * m), dtype=np.float64)
        np.fill_diagonal(iou_mat, 0.0)
        inter_mat = iou_mat * (areas[:, None] + areas[None, :]) / np.maximum(1.0 + iou_mat, 1e-9)
        cov_j_by_i = inter_mat / np.maximum(areas[None, :], 1.0)
        cov_i_by_j = inter_mat / np.maximum(areas[:, None], 1.0)

        dx = np.maximum(0.0, np.maximum(x0[:, None] - x1[None, :], x0[None, :] - x1[:, None]))
        dy = np.maximum(0.0, np.maximum(y0[:, None] - y1[None, :], y0[None, :] - y1[:, None]))
        bgap = np.hypot(dx, dy)
        np.fill_diagonal(bgap, 1e6)

        src0_area_sum = max(float(areas[srcs == 0].sum()), 1.0)
        order_p1 = np.argsort(p1s, kind="mergesort")
        ranks = np.empty(m, dtype=np.float64)
        ranks[order_p1] = (np.arange(m, dtype=np.float64) + 1.0) / float(m)

        for i, row in enumerate(rows):
            s_id = int(srcs[i])
            row["stem_prop_count"] = float(m)
            row["stem_p1_rank_pct"] = float(ranks[i])
            row["stem_area_share"] = float(areas[i] / src0_area_sum)
            row["n_overlap_gt15"] = float((iou_mat[i] > 0.15).sum())

            child_mask = (cov_j_by_i[i] >= 0.65) & (areas <= 0.78 * areas[i]) & np.isin(srcs, [0, 7, 8])
            if np.any(child_mask):
                row["n_strict_children"] = float(child_mask.sum())
                row["children_area_sum_ratio"] = float(areas[child_mask].sum() / max(areas[i], 1.0))
                row["children_p1_wmean"] = float(np.sum(areas[child_mask] * p1s[child_mask]) / max(areas[child_mask].sum(), 1.0))
            else:
                row["n_strict_children"] = 0.0
                row["children_area_sum_ratio"] = 0.0
                row["children_p1_wmean"] = 0.0

            parent_mask = (cov_i_by_j[i] >= 0.65) & (areas >= 1.28 * areas[i])
            if np.any(parent_mask):
                row["has_strict_parent"] = 1.0
                row["max_parent_area_ratio"] = float(np.max(areas[parent_mask] / max(areas[i], 1.0)))
                row["max_parent_p1"] = float(np.max(p1s[parent_mask]))
            else:
                row["has_strict_parent"] = 0.0
                row["max_parent_area_ratio"] = 1.0
                row["max_parent_p1"] = 0.0

            alt_mask = iou_mat[i] >= 0.35
            if np.any(alt_mask):
                bj = int(np.where(alt_mask)[0][np.argmax(iou_mat[i, alt_mask])])
                row["best_alt_iou"] = float(iou_mat[i, bj])
                row["p0_vs_best_alt"] = float(p0s[i] - p0s[bj])
                row["p1_vs_best_alt"] = float(p1s[i] - p1s[bj])
            else:
                row["best_alt_iou"] = 0.0
                row["p0_vs_best_alt"] = 0.0
                row["p1_vs_best_alt"] = 0.0

            disj = inter_mat[i] <= 0.02 * areas[i]
            disj[i] = False
            row["nearest_disjoint_bbox_gap"] = float(np.min(bgap[i, disj])) if np.any(disj) else 500.0

            for target_src in (0, 1, 4, 6, 7, 8, 9, 10):
                row[f"src_is_{target_src}"] = 1.0 if s_id == target_src else 0.0

    cpu_bf16_ok = cpu_has_fast_bf16()

    @torch.no_grad()
    def forward_single_image_batched(model, n_out_ch: int, dev: torch.device, x_cpu: torch.Tensor, tta_views: int, smoke_fast: bool) -> np.ndarray:
        h_in, w_in = int(x_cpu.shape[2]), int(x_cpu.shape[3])
        x0 = x_cpu.to(dev, non_blocking=(dev.type == "cuda"))
        if smoke_fast or tta_views <= 1:
            views_meta = [(False, [])]
        elif tta_views == 2:
            views_meta = [(False, []), (False, [3])]
        elif tta_views <= 4 or h_in != w_in:
            views_meta = [(False, fl) for fl in ([], [2], [3], [2, 3])]
        else:
            views_meta = [(tr_flag, fl) for tr_flag in (False, True) for fl in ([], [2], [3], [2, 3])]

        acc = torch.zeros((1, n_out_ch, h_in, w_in), device=dev, dtype=torch.float32)
        view_bs = 1

        for v0 in range(0, len(views_meta), view_bs):
            sub_meta = views_meta[v0:v0 + view_bs]
            v_list = []
            for tr_flag, fl in sub_meta:
                vt = x0.transpose(2, 3) if tr_flag else x0
                if fl:
                    vt = torch.flip(vt, fl)
                v_list.append(vt[0])
            xb = torch.stack(v_list, dim=0).contiguous()
            del v_list

            if smoke_fast and dev.type == "cpu":
                sh_h = max(128, (h_in // 2 // 32) * 32)
                sh_w = max(128, (w_in // 2 // 32) * 32)
                bx_s = F.interpolate(xb, size=(sh_h, sh_w), mode="bilinear", align_corners=False)
                pb = F.interpolate(torch.sigmoid(model(bx_s).float()), size=(h_in, w_in), mode="bilinear", align_corners=False)
            elif dev.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    pb = torch.sigmoid(model(xb).float())
            elif cpu_bf16_ok:
                with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                    pb = torch.sigmoid(model(xb).float())
            else:
                pb = torch.sigmoid(model(xb).float())

            for k_v, (tr_flag, fl) in enumerate(sub_meta):
                pk = pb[k_v:k_v + 1]
                if fl:
                    pk = torch.flip(pk, fl)
                if tr_flag:
                    pk = pk.transpose(2, 3)
                acc += pk
            del xb, pb

        out = (acc[0] / float(len(views_meta))).clamp(0.0, 1.0).cpu().numpy()
        del acc, x0
        return np.transpose(out, (1, 2, 0))

    def pack_rois(
        boxes: list[tuple[float, float, float, float]],
        pad: int = 64,
        min_sz: int = 224,
        merge_ratio: float = 1.12,
    ) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []
        rects = []
        for bx, by, bw, bh in boxes:
            cx, cy = bx + 0.5 * bw, by + 0.5 * bh
            rw = min(W, max(min_sz, int(math.ceil((bw + 2 * pad) / 32.0)) * 32))
            rh = min(H, max(min_sz, int(math.ceil((bh + 2 * pad) / 32.0)) * 32))
            nx0 = min(W - rw, max(0, int(round((cx - 0.5 * rw) / 32.0)) * 32))
            ny0 = min(H - rh, max(0, int(round((cy - 0.5 * rh) / 32.0)) * 32))
            rects.append([nx0, ny0, nx0 + rw, ny0 + rh])
        changed = True
        while changed:
            changed = False
            n_r = len(rects)
            for i_r in range(n_r):
                for j_r in range(i_r + 1, n_r):
                    ax0, ay0, ax1, ay1 = rects[i_r]
                    bx0, by0, bx1, by1 = rects[j_r]
                    ux0, uy0, ux1, uy1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
                    ua = (ux1 - ux0) * (uy1 - uy0)
                    sa = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0)
                    ov_x = max(0, min(ax1, bx1) - max(ax0, bx0))
                    ov_y = max(0, min(ay1, by1) - max(ay0, by0))
                    if (ov_x > 0 and ov_y > 0 and ua <= merge_ratio * sa) or ua <= sa:
                        rects[i_r] = [ux0, uy0, ux1, uy1]
                        rects.pop(j_r)
                        changed = True
                        break
                if changed:
                    break
        return [(r[0], r[1], r[2], r[3]) for r in rects]

    def make_feather_weight(rh: int, rw: int, rx0: int, ry0: int, rx1: int, ry1: int, ramp: int = 24) -> np.ndarray:
        wy = np.ones(rh, dtype=np.float32)
        wx = np.ones(rw, dtype=np.float32)
        if ry0 > 0 and rh > 2 * ramp:
            wy[:ramp] = np.linspace(0.15, 1.0, ramp, dtype=np.float32)
        if ry1 < H and rh > 2 * ramp:
            wy[-ramp:] = np.minimum(wy[-ramp:], np.linspace(1.0, 0.15, ramp, dtype=np.float32))
        if rx0 > 0 and rw > 2 * ramp:
            wx[:ramp] = np.linspace(0.15, 1.0, ramp, dtype=np.float32)
        if rx1 < W and rw > 2 * ramp:
            wx[-ramp:] = np.minimum(wx[-ramp:], np.linspace(1.0, 0.15, ramp, dtype=np.float32))
        return wy[:, None] * wx[None, :]

    def forward_rois_image(
        model,
        n_out_ch: int,
        dev: torch.device,
        x_cpu: torch.Tensor,
        rois: list[tuple[int, int, int, int]],
        tta_views: int,
        smoke_fast: bool,
    ) -> np.ndarray:
        out_sum = np.zeros((H, W, n_out_ch), dtype=np.float32)
        if not rois:
            return out_sum
        if len(rois) == 1 and rois[0] == (0, 0, W, H):
            return forward_single_image_batched(model, n_out_ch, dev, x_cpu, tta_views, smoke_fast)
        w_acc = np.zeros((H, W), dtype=np.float32)
        for rx0, ry0, rx1, ry1 in rois:
            rh, rw = ry1 - ry0, rx1 - rx0
            p_c = forward_single_image_batched(model, n_out_ch, dev, x_cpu[:, :, ry0:ry1, rx0:rx1], tta_views, smoke_fast)
            w2d = make_feather_weight(rh, rw, rx0, ry0, rx1, ry1)
            out_sum[ry0:ry1, rx0:rx1] += p_c * w2d[:, :, None]
            w_acc[ry0:ry1, rx0:rx1] += w2d
        valid_m = w_acc > 0
        out_sum[valid_m] /= w_acc[valid_m, None]
        return out_sum

    def extract_stem_candidates(job) -> tuple[str, list[dict]]:
        (
            stem, prob_mean_f16, p3_f16, p0_v3_f16, seam_w_sum,
            p0_std_f16, p1_min_f16, raw_y, n_yolo_models,
        ) = job

        prob_mean = prob_mean_f16.astype(np.float32)
        p3 = p3_f16.astype(np.float32)
        p0_v3 = p0_v3_f16.astype(np.float32)
        p0_std_map = p0_std_f16.astype(np.float32)
        p1_min_map = p1_min_f16.astype(np.float32)

        p0, p1, p2 = prob_mean[:, :, 0], prob_mean[:, :, 1], prob_mean[:, :, 2]

        proposals: list[tuple[int, dict, int, int, np.ndarray, float]] = []
        fg_seam_cut = ((p0 >= 0.50) & (p3 < 0.45 if seam_w_sum > 0 else True)).astype(np.uint8)
        ncc0, lab0, st0, _ = cv2.connectedComponentsWithStats(fg_seam_cut, connectivity=8)
        cc0_items: list[tuple[dict, int]] = []
        elong_comps = []

        for k in range(1, ncc0):
            bx, by, bw, bh, area = map(int, st0[k])
            if area < 60:
                continue
            mc_raw = fill_holes_crop((lab0[by:by + bh, bx:bx + bw] == k).astype(np.uint8))
            r_raw = encode_crop_rle(mc_raw, bx, by)
            if r_raw is None:
                continue
            cc0_items.append((r_raw, int(mc_raw.sum())))
            # Family 0: Unshaved Dinkelbach seam-cut component
            proposals.append((0, r_raw, bx, by, mc_raw, 0.0))

            p1_crop = p1[by:by + bh, bx:bx + bw]
            p2_crop = p2[by:by + bh, bx:bx + bw]
            p3_crop = p3[by:by + bh, bx:bx + bw]

            # Family 7: Softer Seam-Split Proposals (`(mc_raw > 0) & (p3 < 0.36)`) - unshaved
            if seam_w_sum > 0 and area >= 150 and float(p3_crop[mc_raw > 0].max()) >= 0.36:
                sub_s = ((mc_raw > 0) & (p3_crop < 0.36)).astype(np.uint8)
                nc_s, lab_s, st_s, _ = cv2.connectedComponentsWithStats(sub_s, connectivity=8)
                valid_s = [sk for sk in range(1, nc_s) if st_s[sk, cv2.CC_STAT_AREA] >= 60]
                if len(valid_s) >= 2:
                    for sk in valid_s:
                        sp = fill_holes_crop((lab_s == sk).astype(np.uint8))
                        rs = encode_crop_rle(sp, bx, by)
                        if rs is not None:
                            proposals.append((7, rs, bx, by, sp, 0.0))

            # Family 8: Optional Calibrated / Anti-Bloat Tight Consensus
            mc_cal = calibrate_crop_mask(mc_raw, p1_crop, thr=0.44, min_ret=0.72)
            if int(mc_cal.sum()) < int(0.93 * area):
                r_cal = encode_crop_rle(mc_cal, bx, by)
                if r_cal is not None:
                    proposals.append((8, r_cal, bx, by, mc_cal, 0.0))

            if area >= 160:
                for sub_bin in (
                    ((mc_raw > 0) & (p1_crop >= 0.64) & (p3_crop < 0.42 if seam_w_sum > 0 else True)).astype(np.uint8),
                    ((mc_raw > 0) & (p2_crop >= 0.23) & (p3_crop < 0.42 if seam_w_sum > 0 else True)).astype(np.uint8),
                ):
                    n_t, l_t, s_t, _ = cv2.connectedComponentsWithStats(sub_bin, connectivity=8)
                    valid_t = [tk for tk in range(1, n_t) if s_t[tk, cv2.CC_STAT_AREA] >= 65]
                    for tk in valid_t:
                        mt = fill_holes_crop((l_t == tk).astype(np.uint8))
                        if len(valid_t) >= 2 or (0.32 * area <= int(mt.sum()) <= 0.86 * area):
                            rt = encode_crop_rle(mt, bx, by)
                            if rt is not None:
                                proposals.append((8, rt, bx, by, mt, 0.0))

            # Family 2: Seam-aware Spine-Core Voronoi Watershed Splits - unshaved
            if area >= 220:
                core_bin = (((p2_crop >= 0.42) | (p1_crop >= 0.48)) & (mc_raw > 0) & (p3_crop < 0.36 if seam_w_sum > 0 else True)).astype(np.uint8)
                nc_core, lab_core, st_core, cen_core = cv2.connectedComponentsWithStats(core_bin, connectivity=8)
                valid_cores = [ck for ck in range(1, nc_core) if st_core[ck, cv2.CC_STAT_AREA] >= 25]
                if len(valid_cores) >= 2:
                    kept_cores = []
                    for ck in sorted(valid_cores, key=lambda c: -st_core[c, cv2.CC_STAT_AREA]):
                        cy_c, cx_c = cen_core[ck][1], cen_core[ck][0]
                        if all(math.hypot(cx_c - cen_core[ok][0], cy_c - cen_core[ok][1]) >= 12.0 for ok in kept_cores):
                            kept_cores.append(ck)
                    if len(kept_cores) >= 2:
                        dists = [cv2.distanceTransform(np.where(lab_core == ck, 0, 1).astype(np.uint8), cv2.DIST_L2, 3) for ck in kept_cores]
                        assign = np.argmin(np.stack(dists, axis=0), axis=0)
                        for idx_c in range(len(kept_cores)):
                            part = fill_holes_crop(((assign == idx_c) & (mc_raw > 0)).astype(np.uint8))
                            if int(part.sum()) >= 60:
                                r_part = encode_crop_rle(part, bx, by)
                                if r_part is not None:
                                    proposals.append((2, r_part, bx, by, part, 0.0))

            tip_res = extract_component_tips(mc_raw, bx, by)
            if tip_res is not None:
                elong_comps.append((k, bx, by, bw, bh, mc_raw, tip_res[0], tip_res[1]))

        cc_count_050 = len(cc0_items)

        # Family 1: Unsplit `p0 >= 0.50` components where `p3 >= 0.45` cut a seam
        if seam_w_sum > 0:
            ncc_u, lab_u, st_u, _ = cv2.connectedComponentsWithStats((p0 >= 0.50).astype(np.uint8), connectivity=8)
            for k in range(1, ncc_u):
                bx, by, bw, bh, area = map(int, st_u[k])
                if area < 90:
                    continue
                mc_u = (lab_u[by:by + bh, bx:bx + bw] == k).astype(np.uint8)
                spanned_0 = [cid for cid in np.unique(lab0[by:by + bh, bx:bx + bw][mc_u > 0]) if cid > 0 and st0[cid, cv2.CC_STAT_AREA] >= 60]
                if len(spanned_0) >= 2:
                    mc_u = fill_holes_crop(mc_u)
                    ru = encode_crop_rle(mc_u, bx, by)
                    if ru is not None:
                        proposals.append((1, ru, bx, by, mc_u, 0.0))

        # Family 5: Direct consensus p1 >= 0.45 connected components
        ncc1, lab1, st1, _ = cv2.connectedComponentsWithStats((p1 >= 0.45).astype(np.uint8), connectivity=8)
        for k in range(1, ncc1):
            bx, by, bw, bh, area = map(int, st1[k])
            if area < 60:
                continue
            mc1 = fill_holes_crop((lab1[by:by + bh, bx:bx + bw] == k).astype(np.uint8))
            r1 = encode_crop_rle(mc1, bx, by)
            if r1 is not None:
                proposals.append((5, r1, bx, by, mc1, 0.0))

        # Family 9: Diffuse / Faint Recall Hysteresis
        p0_rec = 0.70 * p0 + 0.30 * p0_v3
        seam_ok_lo = (p3 < 0.40) if seam_w_sum > 0 else True
        ncc_lo, lab_lo, st_lo, _ = cv2.connectedComponentsWithStats(((p0_rec >= 0.25) & seam_ok_lo).astype(np.uint8), connectivity=8)
        for k in range(1, ncc_lo):
            bx, by, bw, bh, area = map(int, st_lo[k])
            if area < 120 or area > 22000:
                continue
            mc_lo = lab_lo[by:by + bh, bx:bx + bw] == k
            if float(p0_rec[by:by + bh, bx:bx + bw][mc_lo].max()) < 0.36 or float(p1[by:by + bh, bx:bx + bw][mc_lo].max()) < 0.25:
                continue
            spanned_050 = [cid for cid in np.unique(lab0[by:by + bh, bx:bx + bw][mc_lo]) if cid > 0 and st0[cid, cv2.CC_STAT_AREA] >= 60]
            if len(spanned_050) <= 3:
                mf = fill_holes_crop(mc_lo.astype(np.uint8))
                rf = encode_crop_rle(mf, bx, by)
                if rf is not None:
                    proposals.append((9, rf, bx, by, mf, 0.0))

        # Family 10: Oriented Principal-Tip-Tangent Linker across 4..46px gaps
        n_el = len(elong_comps)
        if n_el >= 2:
            adj_links: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
            for i in range(n_el):
                _, bxi, byi, bwi, bhi, mci, tips_i, sw_i = elong_comps[i]
                for j in range(i + 1, n_el):
                    _, bxj, byj, bwj, bhj, mcj, tips_j, sw_j = elong_comps[j]
                    gap_x = max(0, max(bxi - (bxj + bwj), bxj - (bxi + bwi)))
                    gap_y = max(0, max(byi - (byj + bhj), byj - (byi + bhi)))
                    if math.hypot(gap_x, gap_y) > 48.0:
                        continue
                    best_pair = None
                    best_dist = 1e9
                    for tx1, ty1, dx1, dy1 in tips_i:
                        for tx2, ty2, dx2, dy2 in tips_j:
                            dist = math.hypot(tx2 - tx1, ty2 - ty1)
                            if dist < 2.0 or dist > 48.0:
                                continue
                            cx_dir, cy_dir = (tx2 - tx1) / dist, (ty2 - ty1) / dist
                            align1 = dx1 * cx_dir + dy1 * cy_dir
                            align2 = -(dx2 * cx_dir + dy2 * cy_dir)
                            if (dist <= 16.0 and align1 >= 0.15 and align2 >= 0.15) or (align1 >= 0.45 and align2 >= 0.45):
                                if dist < best_dist:
                                    best_dist = dist
                                    best_pair = (tx1, ty1, tx2, ty2)
                    if best_pair is not None:
                        tx1, ty1, tx2, ty2 = best_pair
                        n_pts = max(5, int(round(best_dist)))
                        xs_s = np.clip(np.rint(np.linspace(tx1, tx2, n_pts)).astype(np.int32), 0, W - 1)
                        ys_s = np.clip(np.rint(np.linspace(ty1, ty2, n_pts)).astype(np.int32), 0, H - 1)
                        if seam_w_sum > 0 and float(p3[ys_s, xs_s].max()) >= 0.42:
                            continue
                        if float(p0_rec[ys_s, xs_s].mean()) < 0.06 and best_dist > 18.0:
                            continue
                        adj_links[i].append((j, tx1, ty1, tx2, ty2))
                        adj_links[j].append((i, tx2, ty2, tx1, ty1))

                        ux0 = max(0, min(bxi, bxj) - 4)
                        uy0 = max(0, min(byi, byj) - 4)
                        ux1 = min(W, max(bxi + bwi, bxj + bwj) + 4)
                        uy1 = min(H, max(byi + bhi, byj + bhj) + 4)
                        canvas = np.zeros((uy1 - uy0, ux1 - ux0), dtype=np.uint8)
                        canvas[byi - uy0:byi - uy0 + bhi, bxi - ux0:bxi - ux0 + bwi] |= mci
                        canvas[byj - uy0:byj - uy0 + bhj, bxj - ux0:bxj - ux0 + bwj] |= mcj
                        corridor = np.zeros_like(canvas)
                        thick = max(3, int(round(0.65 * min(sw_i, sw_j))))
                        cv2.line(corridor, (int(round(tx1 - ux0)), int(round(ty1 - uy0))), (int(round(tx2 - ux0)), int(round(ty2 - uy0))), 1, thickness=thick)
                        thin_c = np.zeros_like(canvas)
                        cv2.line(thin_c, (int(round(tx1 - ux0)), int(round(ty1 - uy0))), (int(round(tx2 - ux0)), int(round(ty2 - uy0))), 1, thickness=max(2, thick // 2))
                        seam_guard_c = (p3[uy0:uy1, ux0:ux1] < 0.42) if seam_w_sum > 0 else True
                        bridged_m = canvas | (((corridor > 0) & (p0_rec[uy0:uy1, ux0:ux1] >= 0.14) | (thin_c > 0)) & seam_guard_c)
                        bridged_m = fill_holes_crop(bridged_m.astype(np.uint8))
                        rl = encode_crop_rle(bridged_m, ux0, uy0)
                        if rl is not None:
                            proposals.append((10, rl, ux0, uy0, bridged_m, 0.0))

            for j_node, links_j in adj_links.items():
                if len(links_j) >= 2:
                    for a_idx in range(len(links_j)):
                        for b_idx in range(a_idx + 1, len(links_j)):
                            i_node, tx1a, ty1a, tx2a, ty2a = links_j[a_idx]
                            k_node, tx1b, ty1b, tx2b, ty2b = links_j[b_idx]
                            if math.hypot(tx1a - tx1b, ty1a - ty1b) < 12.0:
                                continue
                            nodes = [i_node, j_node, k_node]
                            ux0 = max(0, min(elong_comps[n][1] for n in nodes) - 4)
                            uy0 = max(0, min(elong_comps[n][2] for n in nodes) - 4)
                            ux1 = min(W, max(elong_comps[n][1] + elong_comps[n][3] for n in nodes) + 4)
                            uy1 = min(H, max(elong_comps[n][2] + elong_comps[n][4] for n in nodes) + 4)
                            canvas = np.zeros((uy1 - uy0, ux1 - ux0), dtype=np.uint8)
                            for n in nodes:
                                _, bxn, byn, bwn, bhn, mcn, _, _ = elong_comps[n]
                                canvas[byn - uy0:byn - uy0 + bhn, bxn - ux0:bxn - ux0 + bwn] |= mcn
                            for x_s, y_s, x_e, y_e in [(tx1a, ty1a, tx2a, ty2a), (tx1b, ty1b, tx2b, ty2b)]:
                                cv2.line(canvas, (int(round(x_s - ux0)), int(round(y_s - uy0))), (int(round(x_e - ux0)), int(round(y_e - uy0))), 1, thickness=4)
                            canvas = fill_holes_crop(canvas)
                            rl3 = encode_crop_rle(canvas, ux0, uy0)
                            if rl3 is not None:
                                proposals.append((10, rl3, ux0, uy0, canvas, 0.0))

        # Family 3: Seam-Guarded Hysteresis Gap-Bridging - unshaved
        k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        seam_ok_b = (p3 < 0.40) if seam_w_sum > 0 else True
        bridged = ((cv2.morphologyEx(fg_seam_cut, cv2.MORPH_CLOSE, k_bridge) > 0) & (p0 >= 0.28) & seam_ok_b).astype(np.uint8) | fg_seam_cut
        ncc_b, lab_b, st_b, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
        for k in range(1, ncc_b):
            bx, by, bw, bh, area = map(int, st_b[k])
            if area < 120:
                continue
            mc = (lab_b[by:by + bh, bx:bx + bw] == k).astype(np.uint8)
            orig_ids = np.unique(lab0[by:by + bh, bx:bx + bw][mc > 0])
            if len(orig_ids[orig_ids > 0]) >= 2:
                mc = fill_holes_crop(mc)
                rle = encode_crop_rle(mc, bx, by)
                if rle is not None:
                    proposals.append((3, rle, bx, by, mc, 0.0))

        # Calibrated 4-Fold Consensus YOLO Clustering (`ymat >= 0.35`, `w_tot / sqrt(k * M)`)
        yolo_rles, yolo_confs, yolo_areas = [], [], []
        if raw_y and n_yolo_models > 0:
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            k3_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            raw_y_sorted = sorted(raw_y, key=lambda x: -x[0])
            used_y = [False] * len(raw_y_sorted)
            yr_all = [x[2] for x in raw_y_sorted]
            ymat = np.asarray(mu.iou(yr_all, yr_all, [0] * len(yr_all)))
            denom_y = float(max(n_yolo_models, 1))
            for yi in range(len(raw_y_sorted)):
                if used_y[yi]:
                    continue
                grp_idx = [yi]
                seen_folds = {raw_y_sorted[yi][1]}
                used_y[yi] = True
                for yj in range(yi + 1, len(raw_y_sorted)):
                    if not used_y[yj] and raw_y_sorted[yj][1] not in seen_folds and ymat[yi, yj] >= 0.35:
                        grp_idx.append(yj)
                        seen_folds.add(raw_y_sorted[yj][1])
                        used_y[yj] = True
                w_tot_y = sum(raw_y_sorted[yj][0] for yj in grp_idx)
                cal_sc = w_tot_y / math.sqrt(float(len(grp_idx)) * denom_y)
                if cal_sc < 0.03:
                    continue
                if len(grp_idx) >= 2:
                    bboxes = [list(map(int, mu.toBbox(raw_y_sorted[yj][2]))) for yj in grp_idx]
                    ux0 = max(0, min(b[0] for b in bboxes))
                    uy0 = max(0, min(b[1] for b in bboxes))
                    ux1 = min(W, max(b[0] + b[2] for b in bboxes))
                    uy1 = min(H, max(b[1] + b[3] for b in bboxes))
                    acc_y = sum(
                        raw_y_sorted[yj][0] * mu.decode(raw_y_sorted[yj][2])[uy0:uy1, ux0:ux1].astype(np.float32)
                        for yj in grp_idx
                    )
                    thr_y = 0.5001 * w_tot_y if len(grp_idx) == 2 else 0.48 * w_tot_y
                    cons_y = (acc_y >= thr_y).astype(np.uint8)
                    yr = encode_crop_rle(cons_y, ux0, uy0) if int(cons_y.sum()) >= 60 else raw_y_sorted[yi][2]
                    if yr is None:
                        yr = raw_y_sorted[yi][2]
                else:
                    yr = raw_y_sorted[yi][2]

                ya = float(mu.area(yr))
                if ya < 60:
                    continue
                yolo_rles.append(yr)
                yolo_confs.append(float(cal_sc))
                yolo_areas.append(ya)

                bx, by, bw, bh = map(int, mu.toBbox(yr))
                if bw > 0 and bh > 0 and cal_sc >= 0.12:
                    x0_y, y0_y = max(0, bx - 4), max(0, by - 4)
                    x1_y, y1_y = min(W, bx + bw + 4), min(H, by + bh + 4)
                    ym_crop = mu.decode(yr)[y0_y:y1_y, x0_y:x1_y]
                    ref = ((cv2.erode(ym_crop, k5) > 0) | ((cv2.dilate(ym_crop, k5) > 0) & ((p0[y0_y:y1_y, x0_y:x1_y] >= 0.45) | (p1[y0_y:y1_y, x0_y:x1_y] >= 0.42)))).astype(np.uint8)
                    ref = fill_holes_crop(ref)
                    ra = int(ref.sum())
                    if ra >= 60 and 0.55 <= ra / max(ya, 1.0) <= 1.45:
                        rr = encode_crop_rle(ref, x0_y, y0_y)
                        if rr is not None:
                            proposals.append((4, rr, x0_y, y0_y, ref, float(cal_sc)))

                    if cal_sc >= 0.16 and cc0_items:
                        spanned_ids = []
                        lab0_crop = lab0[y0_y:y1_y, x0_y:x1_y]
                        for cid in np.unique(lab0_crop[ym_crop > 0]):
                            if cid == 0:
                                continue
                            c_tot = int(st0[cid, cv2.CC_STAT_AREA])
                            if c_tot >= 60:
                                c_in = int(((lab0_crop == cid) & (ym_crop > 0)).sum())
                                if c_in >= 0.55 * c_tot:
                                    spanned_ids.append(int(cid))
                        if len(spanned_ids) >= 2:
                            ux0_u = min([x0_y] + [int(st0[cid, cv2.CC_STAT_LEFT]) for cid in spanned_ids])
                            uy0_u = min([y0_y] + [int(st0[cid, cv2.CC_STAT_TOP]) for cid in spanned_ids])
                            ux1_u = max([x1_y] + [int(st0[cid, cv2.CC_STAT_LEFT] + st0[cid, cv2.CC_STAT_WIDTH]) for cid in spanned_ids])
                            uy1_u = max([y1_y] + [int(st0[cid, cv2.CC_STAT_TOP] + st0[cid, cv2.CC_STAT_HEIGHT]) for cid in spanned_ids])
                            u_mask = np.isin(lab0[uy0_u:uy1_u, ux0_u:ux1_u], spanned_ids)
                            ym_u = mu.decode(yr)[uy0_u:uy1_u, ux0_u:ux1_u]
                            s_guard = (p3[uy0_u:uy1_u, ux0_u:ux1_u] < 0.42) if seam_w_sum > 0 else True
                            bridge_spine = (cv2.erode(ym_u, k3_e) > 0) & (p0_rec[uy0_u:uy1_u, ux0_u:ux1_u] >= 0.15) & s_guard
                            u_cal = fill_holes_crop((u_mask | bridge_spine).astype(np.uint8))
                            if int(u_cal.sum()) >= 80:
                                ru = encode_crop_rle(u_cal, ux0_u, uy0_u)
                                if ru is not None:
                                    proposals.append((6, ru, ux0_u, uy0_u, u_cal, float(cal_sc)))

        cand_rows: list[dict] = []
        if proposals:
            cc0_rles = [it[0] for it in cc0_items]
            cc0_areas = np.asarray([it[1] for it in cc0_items], dtype=np.float64) if cc0_items else np.zeros(0)

            dedup, dedup_rles = [], []
            for item in proposals:
                if dedup_rles:
                    ious = mu.iou(dedup_rles, [item[1]], [0]).ravel()
                    if float(ious.max()) > 0.93:
                        continue
                dedup.append(item)
                dedup_rles.append(item[1])

            ci_mat = np.asarray(mu.iou(dedup_rles, cc0_rles, [0] * len(cc0_rles))) if cc0_rles else np.zeros((len(dedup_rles), 0))
            yi_mat = np.asarray(mu.iou(dedup_rles, yolo_rles, [0] * len(yolo_rles))) if yolo_rles else np.zeros((len(dedup_rles), 0))

            ya_arr = np.asarray(yolo_areas, dtype=np.float64)
            yc_arr = np.asarray(yolo_confs, dtype=np.float64)
            pa_arr = np.asarray([float(mu.area(r)) for r in dedup_rles], dtype=np.float64)

            if yolo_rles:
                inter_py = yi_mat * (pa_arr[:, None] + ya_arr[None, :]) / np.maximum(1.0 + yi_mat, 1e-6)
                if cc0_rles:
                    cy_iou = np.asarray(mu.iou(cc0_rles, yolo_rles, [0] * len(yolo_rles)))
                    inter_cy = cy_iou * (cc0_areas[:, None] + ya_arr[None, :]) / np.maximum(1.0 + cy_iou, 1e-6)
                    yolo_spanned_cc_count = ((inter_cy / np.maximum(cc0_areas[:, None], 1.0)) >= 0.55).sum(axis=0)
                else:
                    yolo_spanned_cc_count = np.zeros(len(yolo_rles), dtype=np.int32)
            else:
                inter_py = np.zeros((len(dedup_rles), 0))
                yolo_spanned_cc_count = np.zeros(0, dtype=np.int32)

            k3 = np.ones((3, 3), dtype=np.uint8)
            for idx_p, (src_id, rle, bx, by, mc, orig_yconf) in enumerate(dedup):
                bh, bw = mc.shape
                fg = mc > 0
                area = int(fg.sum())
                if area < 60:
                    continue
                v0 = p0[by:by + bh, bx:bx + bw][fg]
                v0_v3 = p0_v3[by:by + bh, bx:bx + bw][fg]
                v1 = p1[by:by + bh, bx:bx + bw][fg]
                v2 = p2[by:by + bh, bx:bx + bw][fg]
                v3 = p3[by:by + bh, bx:bx + bw][fg]
                vs = p0_std_map[by:by + bh, bx:bx + bw][fg]
                vm1 = p1_min_map[by:by + bh, bx:bx + bw][fg]

                pad_x0, pad_y0 = max(0, bx - 2), max(0, by - 2)
                pad_x1, pad_y1 = min(W, bx + bw + 2), min(H, by + bh + 2)
                m_pad = np.zeros((pad_y1 - pad_y0, pad_x1 - pad_x0), dtype=np.uint8)
                m_pad[by - pad_y0:by - pad_y0 + bh, bx - pad_x0:bx - pad_x0 + bw] = mc
                ring = (cv2.dilate(m_pad, k3, iterations=2) > 0) & (m_pad == 0)
                ring_mean = float(p0[pad_y0:pad_y1, pad_x0:pad_x1][ring].mean()) if np.any(ring) else 0.0
                p3_ring_max = float(p3[pad_y0:pad_y1, pad_x0:pad_x1][ring].max()) if np.any(ring) else 0.0

                core_sub = ((p2[by:by + bh, bx:bx + bw] >= 0.45) & fg).astype(np.uint8)
                nc_sub, _, st_sub, _ = cv2.connectedComponentsWithStats(core_sub, connectivity=8)
                n_cores = sum(1 for ck in range(1, nc_sub) if st_sub[ck, cv2.CC_STAT_AREA] >= 15)

                p1_sub = ((p1[by:by + bh, bx:bx + bw] >= 0.48) & fg).astype(np.uint8)
                nc_p1, _, st_p1, _ = cv2.connectedComponentsWithStats(p1_sub, connectivity=8)
                n_p1_cores = sum(1 for ck in range(1, nc_p1) if st_p1[ck, cv2.CC_STAT_AREA] >= 35)

                c_iou = float(ci_mat[idx_p].max()) if ci_mat.shape[1] > 0 else 0.0
                if yi_mat.shape[1] > 0:
                    y_iou = float(yi_mat[idx_p].max())
                    y_conf = max(
                        float(orig_yconf),
                        float(max((sc for sc, jv in zip(yolo_confs, yi_mat[idx_p]) if jv >= 0.45), default=0.0)),
                    )
                    valid_y = yc_arr >= 0.16
                    n_yolo_sub = int(((inter_py[idx_p, valid_y] / np.maximum(ya_arr[valid_y], 1.0)) >= 0.55).sum()) if np.any(valid_y) else 0
                    cov_by_y = inter_py[idx_p] / float(max(area, 1))
                    valid_parent = (yc_arr >= 0.16) & (cov_by_y >= 0.55)
                    if np.any(valid_parent):
                        best_yj = int(np.where(valid_parent)[0][np.argmax(yc_arr[valid_parent])])
                        yolo_contain = float(cov_by_y[best_yj])
                        yolo_span_ccs = int(yolo_spanned_cc_count[best_yj])
                        yolo_area_ratio = float(area / max(ya_arr[best_yj], 1.0))
                    else:
                        yolo_contain = float(cov_by_y.max())
                        yolo_span_ccs = 0
                        yolo_area_ratio = 1.0
                else:
                    y_iou, y_conf, n_yolo_sub, yolo_contain, yolo_span_ccs, yolo_area_ratio = 0.0, float(orig_yconf), 0, 0.0, 0, 1.0

                p0_m = float(v0.mean())
                p0_v3_m = float(v0_v3.mean())
                p1_m = float(v1.mean())
                diag = max(math.hypot(bw, bh), 1.0)

                cand_rows.append(dict(
                    stem=stem,
                    rle=rle,
                    source=src_id,
                    area=area,
                    log_area=math.log1p(area),
                    bw=bw,
                    bh=bh,
                    fill=area / max(bw * bh, 1),
                    aspect=max(bw, bh) / max(1, min(bw, bh)),
                    stroke_width=area / diag,
                    disk_radius=float(math.hypot(bx + bw / 2.0 - 1024.0, by + bh / 2.0 - 1024.0) / 1000.0),
                    p0_mean=p0_m,
                    p0_q25=float(np.quantile(v0, 0.25)),
                    p0_q75=float(np.quantile(v0, 0.75)),
                    p0_max=float(v0.max()),
                    p0_border_drop=p0_m - ring_mean,
                    p0_v3_mean=p0_v3_m,
                    p0_v3_gap=p0_v3_m - p0_m,
                    p1_mean=p1_m,
                    p1_q25=float(np.quantile(v1, 0.25)),
                    p1_q75=float(np.quantile(v1, 0.75)),
                    p1_max=float(v1.max()),
                    p1_p0_ratio=p1_m / max(p0_m, 1e-4),
                    p2_mean=float(v2.mean()),
                    p2_max=float(v2.max()),
                    p3_mean=float(v3.mean()),
                    p3_max=float(v3.max()),
                    p3_ring_max=p3_ring_max,
                    n_cores=n_cores,
                    n_p1_cores=n_p1_cores,
                    cc_iou=c_iou,
                    image_cc_count=cc_count_050,
                    p0_arch_std=float(vs.mean()),
                    p1_arch_min=float(vm1.mean()),
                    yolo_iou=y_iou,
                    yolo_conf=y_conf,
                    n_yolo_sub=n_yolo_sub,
                    yolo_contain=yolo_contain,
                    yolo_span_ccs=yolo_span_ccs,
                    yolo_area_ratio=yolo_area_ratio,
                ))

            compute_stem_relational_features(cand_rows)

        return stem, cand_rows

    def _mwis_component(weights: np.ndarray, adj_bits: list[int]) -> list[int]:
        k = len(weights)
        order = np.argsort(-weights)
        w_ord = weights[order]
        adj_ord = [0] * k
        for idx_u, orig_u in enumerate(order):
            bits = adj_bits[orig_u]
            new_b = 0
            for idx_v, orig_v in enumerate(order):
                if (bits >> orig_v) & 1:
                    new_b |= (1 << idx_v)
            adj_ord[idx_u] = new_b
        suffix_sum = np.zeros(k + 1, dtype=np.float64)
        for i in range(k - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + max(0.0, w_ord[i])
        best_val = 0.0
        best_mask = 0

        def dfs(idx: int, cur_val: float, cur_mask: int, forbidden: int) -> None:
            nonlocal best_val, best_mask
            if cur_val > best_val:
                best_val = cur_val
                best_mask = cur_mask
            if idx >= k or cur_val + suffix_sum[idx] <= best_val + 1e-9:
                return
            if not ((forbidden >> idx) & 1) and w_ord[idx] > 0.0:
                dfs(idx + 1, cur_val + w_ord[idx], cur_mask | (1 << idx), forbidden | adj_ord[idx])
            dfs(idx + 1, cur_val, cur_mask, forbidden)

        dfs(0, 0.0, 0, 0)
        return [int(order[i]) for i in range(k) if (best_mask >> i) & 1]

    def mwis_select_stem(
        cand_rows: list[dict],
        cut: float,
        allowed_sources: set[int],
        split_cost: float = 0.24,
        max_overlap_frac: float = 0.12,
    ) -> list[dict]:
        sub_rows = [r for r in cand_rows if float(r["score"]) >= cut and int(r["source"]) in allowed_sources]
        if not sub_rows:
            return []
        sub_rows.sort(key=lambda r: -float(r["score"]))
        rles = [r["rle"] for r in sub_rows]
        areas = np.asarray([r["area"] for r in sub_rows], dtype=np.float64)
        scores = np.asarray([r["score"] for r in sub_rows], dtype=np.float64)
        m = len(rles)

        iou_m = np.asarray(mu.iou(rles, rles, [0] * m), dtype=np.float64)
        np.fill_diagonal(iou_m, 0.0)
        inter_m = iou_m * (areas[:, None] + areas[None, :]) / np.maximum(1.0 + iou_m, 1e-9)
        conflict = inter_m > max_overlap_frac * np.minimum(areas[:, None], areas[None, :])

        visited = [False] * m
        picked_indices = []
        for start in range(m):
            if visited[start]:
                continue
            comp = []
            q = [start]
            visited[start] = True
            while q:
                u = q.pop()
                comp.append(u)
                for v in np.flatnonzero(conflict[u]):
                    if not visited[int(v)]:
                        visited[int(v)] = True
                        q.append(int(v))
            if len(comp) == 1:
                picked_indices.append(comp[0])
            elif len(comp) <= 22:
                w_c = scores[comp] - split_cost
                adj_b = []
                for ui_c, u in enumerate(comp):
                    b = 0
                    for vi_c, v in enumerate(comp):
                        if ui_c != vi_c and conflict[u, v]:
                            b |= (1 << vi_c)
                    adj_b.append(b)
                chosen_local = _mwis_component(w_c, adj_b)
                if chosen_local:
                    picked_indices.extend(comp[idx_l] for idx_l in chosen_local)
                else:
                    picked_indices.append(max(comp, key=lambda idx: scores[idx]))
            else:
                comp_sorted = sorted(comp, key=lambda idx: -scores[idx])
                sel = [comp_sorted[0]]
                for u in comp_sorted[1:]:
                    if scores[u] > split_cost and all(not conflict[u, v] for v in sel):
                        sel.append(u)
                picked_indices.extend(sel)

        picked_indices.sort(key=lambda idx: -scores[idx])
        chosen: list[dict] = []
        for idx in picked_indices:
            r = rles[idx]
            if chosen:
                occ = mu.merge(chosen)
                ov = float(mu.area(mu.merge([r, occ], intersect=True)))
                if ov > max_overlap_frac * areas[idx]:
                    continue
                if ov > 0:
                    bx, by, bw, bh = map(int, mu.toBbox(r))
                    m_crop = mu.decode(r)[by:by + bh, bx:bx + bw].astype(bool) & (
                        ~mu.decode(occ)[by:by + bh, bx:bx + bw].astype(bool)
                    )
                    if int(m_crop.sum()) < 60:
                        continue
                    r = encode_crop_rle(m_crop.astype(np.uint8), bx, by)
                    if r is None:
                        continue
            chosen.append(r)
        return chosen

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
    elif sys.platform == "darwin":
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    dev = torch.device("cuda:0" if has_cuda else "cpu")

    cfg_path = os.environ.get("SF_ENS_CONFIG")
    if not cfg_path:
        hits = sorted(
            glob.glob("/kaggle/input/**/ensemble_v4_config.json", recursive=True)
            + glob.glob("/kaggle/input/**/ensemble_config.json", recursive=True),
            key=lambda p: (0 if "ensemble_v4_config.json" in p else (1 if "v4-panoptic-models" in p else 2), p),
        )
        if not hits:
            raise FileNotFoundError("Could not locate ensemble_config.json in /kaggle/input")
        cfg_path = hits[0]
    ens_cfg = json.loads(Path(cfg_path).read_text())
    bundle_dir = Path(ens_cfg.get("orig_bundle_dir") or Path(cfg_path).parent)

    asset_index: dict[str, Path] = {}
    in_root = Path("/kaggle/input")
    if in_root.exists():
        for p_in in sorted(in_root.rglob("*")):
            if p_in.is_file():
                if p_in.name not in asset_index or "v4" in str(p_in):
                    asset_index[p_in.name] = p_in

    def resolve_asset(rel_name: str) -> Path:
        cand = bundle_dir / rel_name
        if cand.exists():
            return cand
        if rel_name in asset_index:
            return asset_index[rel_name]
        raise FileNotFoundError(f"Missing model asset {rel_name} in {bundle_dir} and /kaggle/input")

    arch_specs = ens_cfg["arch_groups"]
    weights = [float(g["weight"]) for g in arch_specs]
    p0_weights = [float(g.get("p0_weight", g["weight"])) for g in arch_specs]
    w_sum = sum(weights)
    w0_sum = sum(p0_weights)
    n_arch = len(arch_specs)
    yolo_rels = [yrel for yrel in ens_cfg.get("yolo_ckpts", []) if (bundle_dir / yrel).exists() or yrel in asset_index]
    yolo_imgsz = int(ens_cfg.get("yolo_imgsz", 1024))
    yolo_use_3ch = bool(ens_cfg.get("yolo_use_3ch", False))
    has_ultralytics = bool(yolo_rels) and (importlib.util.find_spec("ultralytics") is not None)

    selector_files = ens_cfg.get("selector_files") or [f"selector_fold{f}.txt" for f in range(4)]
    selector_paths = [str(resolve_asset(fn)) for fn in selector_files]
    seg_selector_files = ens_cfg.get("seg_selector_files") or []
    seg_selector_paths = [str(resolve_asset(fn)) for fn in seg_selector_files]
    features = ens_cfg["features"]
    seg_features = ens_cfg.get("seg_features") or []
    yolo_gbdt_weight = float(ens_cfg.get("yolo_gbdt_weight", 1.0))
    best_sources_set = {int(s) for s in ens_cfg["best_sources"]}
    best_cut = float(ens_cfg["best_cut"])
    split_cost = float(ens_cfg.get("split_cost", 0.24))
    default_views = int(ens_cfg.get("tta_views", 4))
    smoke_fast = os.environ.get("SF_SMOKE_FAST") == "1"

    all_test_files = locate_test_images()
    shard_spec = os.environ.get("SF_WORKER_SHARD", "")
    is_sharded = bool(shard_spec and "/" in shard_spec)
    if not has_cuda:
        default_views = 1
    elif not is_sharded and n_arch >= 4:
        default_views = min(default_views, 2)
    if is_sharded:
        sh_id, sh_tot = map(int, shard_spec.split("/"))
        test_files = all_test_files[sh_id::sh_tot]
    else:
        test_files = all_test_files

    log(
        f"Worker initialized ({n_arch} seg families, {len(selector_paths)}+{len(seg_selector_paths)} Stage-1 GBDTs, "
        f"{len(yolo_rels)} YOLO ckpts [3ch={yolo_use_3ch}], cut={best_cut:.2f}, split={split_cost:.2f}) "
        f"on {len(test_files)}/{len(all_test_files)} test images"
    )

    yy, xx = np.ogrid[:H, :W]
    disk_mask = (yy - H / 2.0) ** 2 + (xx - W / 2.0) ** 2 <= 940.0 ** 2

    def build_smp_model(arch_name: str, backbone: str, ck_path: Path):
        sd = torch.load(str(ck_path), map_location="cpu", weights_only=True)
        n_out_ch = int(sd["segmentation_head.0.weight"].shape[0])
        m = getattr(smp, arch_name)(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=n_out_ch, activation=None)
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
        del sd
        return m.eval().to(dev), n_out_ch

    sub_rows: list[dict] = []
    overlap_total = 0
    cur_chunk_size = 12 if not has_cuda else CHUNK_SIZE

    for c_start in range(0, len(test_files), cur_chunk_size):
        chunk_paths = test_files[c_start:c_start + cur_chunk_size]
        B = len(chunk_paths)

        elapsed_m = (time.time() - T0) / 60.0
        proj_m = elapsed_m * (len(test_files) / float(max(c_start, 1))) if c_start >= cur_chunk_size else 0.0
        budget_ok = (elapsed_m < 46.0) and (c_start < cur_chunk_size or proj_m <= 53.0)

        chunk_ch01_u8, chunk_tensors, chunk_halpha_boxes = [], [], []
        inner_disk = (yy - H / 2.0) ** 2 + (xx - W / 2.0) ** 2 <= 915.0 ** 2
        for img_p in chunk_paths:
            img_u8 = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img_u8 is None:
                raise RuntimeError(f"Failed to decode test image: {img_p}")
            if img_u8.shape != (H, W):
                img_u8 = cv2.resize(img_u8, (W, H), interpolation=cv2.INTER_LINEAR)
            st = compute_disk_stats(img_u8, disk_mask)
            ch01_u8 = encode_3ch_u8(img_u8, st)
            ch_01 = ch01_u8.astype(np.float32) / 255.0
            x_t = torch.from_numpy(np.ascontiguousarray((ch_01 - 0.5) / 0.25, dtype=np.float32))[None]
            chunk_ch01_u8.append(ch01_u8)
            chunk_tensors.append(x_t)

            # Fast H-alpha high-absorption backup boxes (used only when YOLO detects 0 boxes on a disk)
            dark_m = ((ch01_u8[2] >= 188) & inner_disk).astype(np.uint8)
            n_d, _, st_d, _ = cv2.connectedComponentsWithStats(dark_m, connectivity=8)
            h_boxes = [
                (float(st_d[kd, 0]), float(st_d[kd, 1]), float(st_d[kd, 2]), float(st_d[kd, 3]))
                for kd in range(1, n_d)
                if st_d[kd, 4] >= 80
            ]
            chunk_halpha_boxes.append(h_boxes)

        active_yolo_rels = yolo_rels[:1] if smoke_fast else yolo_rels
        chunk_raw_y: list[list[tuple[float, int, dict]]] = [[] for _ in range(B)]
        if has_ultralytics:
            from ultralytics import YOLO
            yolo_dev = 0 if has_cuda else "cpu"
            for y_idx, yrel in enumerate(active_yolo_rels):
                ym = YOLO(str(resolve_asset(yrel)))
                for i, img_p in enumerate(chunk_paths):
                    y_in = np.ascontiguousarray(np.transpose(chunk_ch01_u8[i][::-1], (1, 2, 0))) if yolo_use_3ch else str(img_p)
                    res = ym.predict(y_in, imgsz=yolo_imgsz, conf=0.03, iou=0.60, max_det=150, retina_masks=True, device=yolo_dev, verbose=False)[0]
                    if res.masks is not None and len(res.boxes) > 0:
                        ms = res.masks.data.cpu().numpy() > 0.5
                        cs = res.boxes.conf.cpu().numpy()
                        for sc, mk in zip(cs, ms):
                            if mk.shape != (H, W):
                                mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                            if int(mk.sum()) >= 25:
                                yr = mu.encode(np.asfortranarray(mk.astype(np.uint8)))
                                chunk_raw_y[i].append((float(sc), y_idx, yr))
                del ym
                gc.collect()
                if has_cuda:
                    torch.cuda.empty_cache()

        n_yolo_active = len(active_yolo_rels) if has_ultralytics else 0
        use_cpu_roi = (not has_cuda) and (n_yolo_active > 0)
        chunk_rois: list[list[tuple[int, int, int, int]]] = []
        if use_cpu_roi:
            for i in range(B):
                y_boxes = [tuple(map(float, mu.toBbox(yr))) for sc, _, yr in chunk_raw_y[i] if sc >= 0.05]
                chunk_rois.append(pack_rois(y_boxes if y_boxes else chunk_halpha_boxes[i], pad=64, min_sz=224, merge_ratio=1.12))

        prob_mean_acc = np.zeros((B, H, W, 3), dtype=np.float32)
        seam_acc = np.zeros((B, H, W), dtype=np.float32)
        seam_w_sum = 0.0
        v3_p0_acc = np.zeros((B, H, W), dtype=np.float32)
        v3_w_sum = 0.0
        p0_unw_sum = np.zeros((B, H, W), dtype=np.float32)
        p0_unw_sq = np.zeros((B, H, W), dtype=np.float32)
        p1_unw_min = np.full((B, H, W), 2.0, dtype=np.float32)

        for grp, w, w0 in zip(arch_specs, weights, p0_weights):
            arch_name = grp["arch"]
            backbone = grp["backbone"]
            ckpts = grp["ckpts"][:1] if (smoke_fast or (not has_cuda and not budget_ok)) else grp["ckpts"]
            raw_views = min(default_views, int(grp.get("tta_views", default_views)))
            grp_views = raw_views if budget_ok else min(2, raw_views)
            if arch_name == "UnetPlusPlus":
                grp_views = min(grp_views, 2)
            grp_acc = None

            for ck_rel in ckpts:
                m0, n_out_ch = build_smp_model(arch_name, backbone, resolve_asset(ck_rel))
                if grp_acc is None:
                    grp_acc = np.zeros((B, H, W, n_out_ch), dtype=np.float32)
                for i in range(B):
                    if use_cpu_roi:
                        grp_acc[i] += forward_rois_image(m0, n_out_ch, dev, chunk_tensors[i], chunk_rois[i], grp_views, smoke_fast)
                    else:
                        grp_acc[i] += forward_single_image_batched(m0, n_out_ch, dev, chunk_tensors[i], grp_views, smoke_fast)
                del m0
                gc.collect()
                if has_cuda:
                    torch.cuda.empty_cache()

            grp_acc /= float(len(ckpts))
            p0_g = grp_acc[:, :, :, 0]
            prob_mean_acc[:, :, :, 0] += p0_g * (w0 / w0_sum)
            prob_mean_acc[:, :, :, 1:3] += grp_acc[:, :, :, 1:3] * (w / w_sum)
            p0_unw_sum += p0_g
            p0_unw_sq += p0_g * p0_g
            np.minimum(p1_unw_min, grp_acc[:, :, :, 1], out=p1_unw_min)
            if grp_acc.shape[3] >= 4:
                seam_acc += grp_acc[:, :, :, 3] * w0
                seam_w_sum += w0
            else:
                v3_p0_acc += p0_g * w
                v3_w_sum += w
            del grp_acc, p0_g
            gc.collect()

        del chunk_tensors, chunk_halpha_boxes
        prob_mean_f16 = prob_mean_acc.astype(np.float16)
        p3_f16 = ((seam_acc / seam_w_sum) if seam_w_sum > 0 else seam_acc).astype(np.float16)
        p0_v3_f16 = ((v3_p0_acc / v3_w_sum) if v3_w_sum > 0 else prob_mean_acc[:, :, :, 0]).astype(np.float16)
        if n_arch >= 2:
            mean_u = p0_unw_sum / float(n_arch)
            p0_std_f16 = np.sqrt(np.maximum(0.0, p0_unw_sq / float(n_arch) - mean_u * mean_u)).astype(np.float16)
            p1_min_f16 = p1_unw_min.astype(np.float16)
            del mean_u
        else:
            p0_std_f16 = np.zeros((B, H, W), dtype=np.float16)
            p1_min_f16 = prob_mean_f16[:, :, :, 1]
        del prob_mean_acc, seam_acc, v3_p0_acc, p0_unw_sum, p0_unw_sq, p1_unw_min
        gc.collect()
        jobs = [
            (
                chunk_paths[i].stem,
                prob_mean_f16[i],
                p3_f16[i],
                p0_v3_f16[i],
                seam_w_sum,
                p0_std_f16[i],
                p1_min_f16[i],
                chunk_raw_y[i],
                n_yolo_active,
            )
            for i in range(B)
        ]
        extracted = [extract_stem_candidates(j) for j in jobs]

        flat_rows = [r_obj for _, c_rows in extracted for r_obj in c_rows]
        if flat_rows:
            df_s1 = pd.DataFrame(flat_rows)
            X_s1 = np.ascontiguousarray(df_s1[features].to_numpy(dtype=np.float64))
            X_seg = (
                np.ascontiguousarray(df_s1[seg_features].to_numpy(dtype=np.float64))
                if (seg_selector_paths and seg_features)
                else np.zeros((len(flat_rows), 0), dtype=np.float64)
            )
            alpha_g = yolo_gbdt_weight if n_yolo_active > 0 else 0.0
            pid_tag = f"{os.getpid()}_{c_start}"
            in_npz = Path(f"/tmp/sf_lgb_in_{pid_tag}.npz")
            out_npy = Path(f"/tmp/sf_lgb_out_{pid_tag}.npy")
            try:
                np.savez(
                    in_npz,
                    X_s1=X_s1,
                    X_seg=X_seg,
                    sel_paths=np.asarray(selector_paths, dtype=str),
                    seg_sel_paths=np.asarray(seg_selector_paths, dtype=str),
                    alpha_g=np.asarray([alpha_g], dtype=np.float64),
                )
                lgb_env = os.environ.copy()
                lgb_env["OMP_NUM_THREADS"] = "1"
                lgb_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                subprocess.check_call(
                    [sys.executable, str(Path(__file__).resolve()), "--lgb-score-subproc", str(in_npz), str(out_npy)],
                    env=lgb_env,
                )
                s1_preds = np.load(out_npy, allow_pickle=False)
            finally:
                in_npz.unlink(missing_ok=True)
                out_npy.unlink(missing_ok=True)
            for r_obj, sv in zip(flat_rows, s1_preds):
                r_obj["stage1_score"] = float(sv)
                r_obj["score"] = float(sv)

        for stem, rows_img in extracted:
            chosen_rles = mwis_select_stem(
                rows_img,
                cut=best_cut,
                allowed_sources=best_sources_set,
                split_cost=split_cost,
                max_overlap_frac=0.12,
            )
            if chosen_rles:
                areas = mu.area(chosen_rles)
                overlap_total += int(areas.sum()) - int(mu.area(mu.merge(chosen_rles)))
            for idx_k, r in enumerate(chosen_rles, 1):
                c = r["counts"]
                c_str = c.decode("ascii") if isinstance(c, bytes) else str(c)
                sub_rows.append({"filament_id": f"{stem}_{idx_k}", "segmentation_rle": c_str})

        del prob_mean_f16, p3_f16, p0_v3_f16, p0_std_f16, p1_min_f16, chunk_ch01_u8, chunk_raw_y, jobs, extracted
        gc.collect()
        log(f"Processed {c_start + B}/{len(test_files)} shard images (instances: {len(sub_rows)})")

    assert overlap_total == 0, f"Strict pixel-disjointness violated: {overlap_total} overlapping pixels!"
    out_csv = Path(os.environ.get("SF_OUT_CSV", "/kaggle/working/submission.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame(sub_rows, columns=["filament_id", "segmentation_rle"])
    sub_df.to_csv(out_csv, index=False)
    log(f"Wrote {out_csv}: {len(sub_df)} instances, 0 overlap pixels")


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "--lgb-score-subproc":
        run_lgb_score_subproc(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--cpu-calibrate-subproc":
        run_cpu_calibrate_subproc(sys.argv[2])
        return
    if os.environ.get("SF_WORKER_SHARD"):
        run_worker_shard()
        return

    install_offline_wheels()
    n_hw_gpus = detect_hw_gpus()
    test_files = locate_test_images()

    cpu_cal_cfg: Path | None = None
    if n_hw_gpus == 0:
        cpu_cal_cfg = Path(f"/tmp/sf_cpu_cal_cfg_{os.getpid()}.json")
        cal_env = os.environ.copy()
        subprocess.check_call(
            [sys.executable, str(Path(__file__).resolve()), "--cpu-calibrate-subproc", str(cpu_cal_cfg)],
            env=cal_env,
        )
        os.environ["SF_ENS_CONFIG"] = str(cpu_cal_cfg)

    use_2shards = len(test_files) >= 2 and (
        n_hw_gpus >= 2
        or (n_hw_gpus == 0 and (sys.platform != "darwin" or os.environ.get("SF_FORCE_CPU_2SHARD_TEST") == "1"))
    )

    try:
        if use_2shards:
            import pandas as pd

            mode_lbl = f"{n_hw_gpus} hardware GPUs" if n_hw_gpus >= 2 else "CPU 2-process sharding"
            log(f"Detected {mode_lbl} -> launching 2 isolated OS worker processes")
            procs = []
            shard_csvs = []
            for g in range(2):
                scsv = Path(f"/tmp/sf_v4_shard_{os.getpid()}_{g}.csv")
                scsv.unlink(missing_ok=True)
                shard_csvs.append(scsv)
                env = os.environ.copy()
                if n_hw_gpus >= 2:
                    env["CUDA_VISIBLE_DEVICES"] = "0" if os.environ.get("SF_FORCE_2SHARD_TEST") == "1" else str(g)
                else:
                    n_thr = "1" if sys.platform == "darwin" else "2"
                    env["OMP_NUM_THREADS"] = n_thr
                    env["MKL_NUM_THREADS"] = n_thr
                env["SF_WORKER_SHARD"] = f"{g}/2"
                env["SF_OUT_CSV"] = str(scsv)
                procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve())], env=env))

            codes = [p.wait() for p in procs]
            assert all(c == 0 for c in codes), f"Worker shard failed with exit codes: {codes}"

            dfs = [pd.read_csv(scsv) for scsv in shard_csvs]
            for scsv in shard_csvs:
                scsv.unlink(missing_ok=True)
            merged = pd.concat(dfs, ignore_index=True)
            if len(merged) > 0:
                merged["_stem"] = merged["filament_id"].map(lambda s: s.rsplit("_", 1)[0])
                merged["_idx"] = merged["filament_id"].map(lambda s: int(s.rsplit("_", 1)[1]))
                merged = merged.sort_values(["_stem", "_idx"], kind="stable").drop(columns=["_stem", "_idx"])
            out_csv = Path(os.environ.get("SF_OUT_CSV", "/kaggle/working/submission.csv"))
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(out_csv, index=False)
            log(f"Merged 2 shards -> {out_csv}: {len(merged)} total instances across {len(test_files)} test images")
        else:
            run_worker_shard()
    finally:
        if cpu_cal_cfg is not None:
            cpu_cal_cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
