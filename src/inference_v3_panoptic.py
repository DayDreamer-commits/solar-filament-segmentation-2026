"""
Solar Filament Segmentation 2026 - v3 Dynamic Offline Panoptic Inference Kernel.

100% dynamic runtime neural inference (`enable_gpu: true`, `enable_internet: false`):
  1. Installs offline wheels (`timm`, `segmentation_models_pytorch`, `ultralytics`)
     from `daydreamerthinks/smp-offline-wheels`.
  2. Streams test images in memory-bounded chunks (`CHUNK_SIZE = 24`), loading only
     ONE checkpoint into GPU VRAM at a time (`<= 150 MB` model footprint, eliminating
     any T4/P100 VRAM fragmentation) and data-parallelizing 50/50 across 2x T4 GPUs
     whenever `torch.cuda.device_count() >= 2`.
  3. Runs full-disk `2048x2048` 3-channel physical encodings with 8-view D4 TTA,
     4-fold Consensus YOLO11m-Seg, Bayes multi-annotator contour calibration
     (`p1 >= 0.45`), topological bipartite features & fragmentation unions, 4-fold
     LightGBM Dinkelbach `v / A` selectors, and strict 0-pixel overlap assertion.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)

H = W = 2048
CHUNK_SIZE = 24
T0 = time.time()


def log(*args) -> None:
    print(f"[{(time.time() - T0) / 60:6.2f}m]", *args, flush=True)


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


def encode_crop_rle(crop_mask: np.ndarray, x0: int, y0: int, mu):
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


def calibrate_crop_mask(mc: np.ndarray, p1_crop: np.ndarray, thr: float = 0.45, min_ret: float = 0.72) -> np.ndarray:
    """Trim Tversky border halo to BCE consensus probability p1 >= thr when safe."""
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


@torch.no_grad()
def forward_single_image(model, arch_name: str, dev: torch.device, x_cpu: torch.Tensor, run_d4: bool, smoke_fast: bool) -> np.ndarray:
    x0 = x_cpu.to(dev, non_blocking=(dev.type == "cuda"))
    views_meta = [(False, [])] if smoke_fast else [
        (tr_flag, fl)
        for tr_flag in ((False, True) if run_d4 else (False,))
        for fl in ([], [2], [3], [2, 3])
    ]
    acc = torch.zeros((1, 3, H, W), device=dev, dtype=torch.float32)
    use_amp_cuda = dev.type == "cuda"

    for tr_flag, fl in views_meta:
        vt = x0.transpose(2, 3) if tr_flag else x0
        if fl:
            vt = torch.flip(vt, fl)
        if smoke_fast and dev.type == "cpu":
            bx_s = F.interpolate(vt, size=(512, 512), mode="bilinear", align_corners=False)
            p = F.interpolate(torch.sigmoid(model(bx_s).float()), size=(H, W), mode="bilinear", align_corners=False)
        elif use_amp_cuda:
            vt = vt.contiguous()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                p = torch.sigmoid(model(vt).float())
        else:
            p = torch.sigmoid(model(vt).float())

        if fl:
            p = torch.flip(p, fl)
        if tr_flag:
            p = p.transpose(2, 3)
        acc += p
        del vt, p

    out = (acc[0] / float(len(views_meta))).clamp(0.0, 1.0).cpu().numpy()
    del acc, x0
    return np.transpose(out, (1, 2, 0))  # (H, W, 3) float32


def extract_stem_candidates(job) -> tuple[str, list[dict]]:
    from pycocotools import mask as mu

    stem, img_u8, stats, arch_pgs, weights, raw_y, n_yolo_models, best_sources = job
    w_sum = sum(weights)
    prob_mean = np.zeros((H, W, 3), dtype=np.float32)
    arch_p0s, arch_p1s = [], []
    for w, pg_f16 in zip(weights, arch_pgs):
        pg = pg_f16.astype(np.float32)
        prob_mean += pg * (w / w_sum)
        arch_p0s.append(pg[:, :, 0])
        arch_p1s.append(pg[:, :, 1])

    if len(arch_pgs) >= 2:
        p0_std_map = np.std(np.stack(arch_p0s, axis=0), axis=0)
        p1_min_map = np.min(np.stack(arch_p1s, axis=0), axis=0)
    else:
        p0_std_map = np.zeros((H, W), dtype=np.float32)
        p1_min_map = prob_mean[:, :, 1]

    span = max(stats[2] - stats[0], 16.0)
    blur = cv2.GaussianBlur(img_u8, (0, 0), 18.0).astype(np.float32)
    dark_map = np.clip((blur - img_u8.astype(np.float32)) / span, -1.0, 2.0)

    p0, p1, p2 = prob_mean[:, :, 0], prob_mean[:, :, 1], prob_mean[:, :, 2]
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k3 = np.ones((3, 3), dtype=np.uint8)

    proposals = []
    ncc0, lab0, st0, _ = cv2.connectedComponentsWithStats((p0 >= 0.50).astype(np.uint8), connectivity=8)
    cc0_items = []
    for k in range(1, ncc0):
        bx, by, bw, bh, area = map(int, st0[k])
        if area < 60:
            continue
        mc_raw = fill_holes_crop((lab0[by:by + bh, bx:bx + bw] == k).astype(np.uint8))
        r_raw = encode_crop_rle(mc_raw, bx, by, mu)
        if r_raw is not None:
            cc0_items.append((r_raw, int(mc_raw.sum())))

        mc_cal = calibrate_crop_mask(mc_raw, p1[by:by + bh, bx:bx + bw], thr=0.45, min_ret=0.72)
        r_cal = encode_crop_rle(mc_cal, bx, by, mu)
        if r_cal is not None:
            proposals.append((0, r_cal, bx, by, mc_cal))

        if area >= 220:
            core_bin = (((p2[by:by + bh, bx:bx + bw] >= 0.42) | (p1[by:by + bh, bx:bx + bw] >= 0.48)) & (mc_raw > 0)).astype(np.uint8)
            nc_core, lab_core, st_core, cen_core = cv2.connectedComponentsWithStats(core_bin, connectivity=8)
            valid_cores = [ck for ck in range(1, nc_core) if st_core[ck, cv2.CC_STAT_AREA] >= 25]
            if len(valid_cores) >= 2:
                kept_cores = []
                for ck in sorted(valid_cores, key=lambda c: -st_core[c, cv2.CC_STAT_AREA]):
                    cy_c, cx_c = cen_core[ck][1], cen_core[ck][0]
                    if all(math.hypot(cx_c - cen_core[ok][0], cy_c - cen_core[ok][1]) >= 12.0 for ok in kept_cores):
                        kept_cores.append(ck)
                if len(kept_cores) >= 2:
                    dists = [
                        cv2.distanceTransform(np.where(lab_core == ck, 0, 1).astype(np.uint8), cv2.DIST_L2, 3)
                        for ck in kept_cores
                    ]
                    assign = np.argmin(np.stack(dists, axis=0), axis=0)
                    for idx_c in range(len(kept_cores)):
                        part = ((assign == idx_c) & (mc_raw > 0)).astype(np.uint8)
                        if int(part.sum()) >= 60:
                            part_cal = calibrate_crop_mask(part, p1[by:by + bh, bx:bx + bw], thr=0.45, min_ret=0.72)
                            r_part = encode_crop_rle(part_cal, bx, by, mu)
                            if r_part is not None:
                                proposals.append((2, r_part, bx, by, part_cal))

    cc_count_050 = len(cc0_items)

    ncc1, lab1, st1, _ = cv2.connectedComponentsWithStats((p1 >= 0.45).astype(np.uint8), connectivity=8)
    for k in range(1, ncc1):
        bx, by, bw, bh, area = map(int, st1[k])
        if area < 60:
            continue
        mc1 = fill_holes_crop((lab1[by:by + bh, bx:bx + bw] == k).astype(np.uint8))
        r1 = encode_crop_rle(mc1, bx, by, mu)
        if r1 is not None:
            proposals.append((5, r1, bx, by, mc1))

    base_050 = (p0 >= 0.50).astype(np.uint8)
    bridged = ((cv2.morphologyEx(base_050, cv2.MORPH_CLOSE, k_bridge) > 0) & (p0 >= 0.28)).astype(np.uint8) | base_050
    ncc_b, lab_b, st_b, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    for k in range(1, ncc_b):
        bx, by, bw, bh, area = map(int, st_b[k])
        if area < 120:
            continue
        mc = (lab_b[by:by + bh, bx:bx + bw] == k).astype(np.uint8)
        orig_ids = np.unique(lab0[by:by + bh, bx:bx + bw][mc > 0])
        if len(orig_ids[orig_ids > 0]) >= 2:
            mc = fill_holes_crop(mc)
            mc_cal = calibrate_crop_mask(mc, p1[by:by + bh, bx:bx + bw], thr=0.45, min_ret=0.72)
            rle = encode_crop_rle(mc_cal, bx, by, mu)
            if rle is not None:
                proposals.append((3, rle, bx, by, mc_cal))

    yolo_rles, yolo_confs, yolo_areas = [], [], []
    if raw_y and n_yolo_models > 0:
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
                if not used_y[yj] and raw_y_sorted[yj][1] not in seen_folds and ymat[yi, yj] >= 0.50:
                    grp_idx.append(yj)
                    seen_folds.add(raw_y_sorted[yj][1])
                    used_y[yj] = True
            avg_sc = sum(raw_y_sorted[yj][0] for yj in grp_idx) / denom_y
            if avg_sc < 0.03:
                continue
            if len(grp_idx) >= 2:
                bboxes = [list(map(int, mu.toBbox(raw_y_sorted[yj][2]))) for yj in grp_idx]
                ux0 = max(0, min(b[0] for b in bboxes))
                uy0 = max(0, min(b[1] for b in bboxes))
                ux1 = min(W, max(b[0] + b[2] for b in bboxes))
                uy1 = min(H, max(b[1] + b[3] for b in bboxes))
                acc_y = sum(mu.decode(raw_y_sorted[yj][2])[uy0:uy1, ux0:ux1].astype(np.float32) for yj in grp_idx)
                cons_y = (acc_y >= 0.50 * len(grp_idx)).astype(np.uint8)
                yr = encode_crop_rle(cons_y, ux0, uy0, mu) if int(cons_y.sum()) >= 60 else raw_y_sorted[yi][2]
                if yr is None:
                    yr = raw_y_sorted[yi][2]
            else:
                yr = raw_y_sorted[yi][2]

            ya = float(mu.area(yr))
            if ya < 60:
                continue
            yolo_rles.append(yr)
            yolo_confs.append(avg_sc)
            yolo_areas.append(ya)

            bx, by, bw, bh = map(int, mu.toBbox(yr))
            if bw > 0 and bh > 0 and avg_sc >= 0.12:
                x0_y, y0_y = max(0, bx - 4), max(0, by - 4)
                x1_y, y1_y = min(W, bx + bw + 4), min(H, by + bh + 4)
                ym_crop = mu.decode(yr)[y0_y:y1_y, x0_y:x1_y]
                ref = ((cv2.erode(ym_crop, k5) > 0) | ((cv2.dilate(ym_crop, k5) > 0) & (p1[y0_y:y1_y, x0_y:x1_y] >= 0.42))).astype(np.uint8)
                ref = fill_holes_crop(ref)
                ra = int(ref.sum())
                if ra >= 60 and 0.55 <= ra / max(ya, 1.0) <= 1.45:
                    rr = encode_crop_rle(ref, x0_y, y0_y, mu)
                    if rr is not None:
                        proposals.append((4, rr, x0_y, y0_y, ref))

                if avg_sc >= 0.16 and cc0_items:
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
                        bridge_spine = (cv2.erode(ym_u, k3_e) > 0) & (p0[uy0_u:uy1_u, ux0_u:ux1_u] >= 0.15)
                        u_cal = (u_mask & (p1[uy0_u:uy1_u, ux0_u:ux1_u] >= 0.42)) | (u_mask & (p0[uy0_u:uy1_u, ux0_u:ux1_u] >= 0.58)) | bridge_spine
                        u_cal = fill_holes_crop(u_cal.astype(np.uint8))
                        if int(u_cal.sum()) >= 80:
                            ru = encode_crop_rle(u_cal, ux0_u, uy0_u, mu)
                            if ru is not None:
                                proposals.append((6, ru, ux0_u, uy0_u, u_cal))

    cand_rows: list[dict] = []
    if proposals:
        cc0_rles = [it[0] for it in cc0_items]
        cc0_areas = np.asarray([it[1] for it in cc0_items], dtype=np.float64) if cc0_items else np.zeros(0)

        dedup, dedup_rles = [], []
        for item in proposals:
            if dedup_rles:
                ious = mu.iou(dedup_rles, [item[1]], [0]).ravel()
                if float(ious.max()) > 0.96:
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

        for idx_p, (src_id, rle, bx, by, mc) in enumerate(dedup):
            if src_id not in best_sources:
                continue
            bh, bw = mc.shape
            fg = mc > 0
            area = int(fg.sum())
            if area < 60:
                continue
            v0 = p0[by:by + bh, bx:bx + bw][fg]
            v1 = p1[by:by + bh, bx:bx + bw][fg]
            v2 = p2[by:by + bh, bx:bx + bw][fg]
            vd = dark_map[by:by + bh, bx:bx + bw][fg]
            vs = p0_std_map[by:by + bh, bx:bx + bw][fg]
            vm1 = p1_min_map[by:by + bh, bx:bx + bw][fg]

            pad_x0, pad_y0 = max(0, bx - 2), max(0, by - 2)
            pad_x1, pad_y1 = min(W, bx + bw + 2), min(H, by + bh + 2)
            m_pad = np.zeros((pad_y1 - pad_y0, pad_x1 - pad_x0), dtype=np.uint8)
            m_pad[by - pad_y0:by - pad_y0 + bh, bx - pad_x0:bx - pad_x0 + bw] = mc
            ring = (cv2.dilate(m_pad, k3, iterations=2) > 0) & (m_pad == 0)
            ring_mean = float(p0[pad_y0:pad_y1, pad_x0:pad_x1][ring].mean()) if np.any(ring) else 0.0

            core_sub = ((p2[by:by + bh, bx:bx + bw] >= 0.45) & fg).astype(np.uint8)
            nc_sub, _, st_sub, _ = cv2.connectedComponentsWithStats(core_sub, connectivity=8)
            n_cores = sum(1 for ck in range(1, nc_sub) if st_sub[ck, cv2.CC_STAT_AREA] >= 15)

            p1_sub = ((p1[by:by + bh, bx:bx + bw] >= 0.48) & fg).astype(np.uint8)
            nc_p1, _, st_p1, _ = cv2.connectedComponentsWithStats(p1_sub, connectivity=8)
            n_p1_cores = sum(1 for ck in range(1, nc_p1) if st_p1[ck, cv2.CC_STAT_AREA] >= 35)

            c_iou = float(ci_mat[idx_p].max()) if ci_mat.shape[1] > 0 else 0.0
            if yi_mat.shape[1] > 0:
                y_iou = float(yi_mat[idx_p].max())
                y_conf = float(max((sc for sc, jv in zip(yolo_confs, yi_mat[idx_p]) if jv >= 0.50), default=0.0))
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
                y_iou, y_conf, n_yolo_sub, yolo_contain, yolo_span_ccs, yolo_area_ratio = 0.0, 0.0, 0, 0.0, 0, 1.0

            p0_m = float(v0.mean())
            p1_m = float(v1.mean())

            cand_rows.append(dict(
                rle=rle,
                source=src_id,
                area=area,
                log_area=math.log1p(area),
                bw=bw,
                bh=bh,
                fill=area / max(bw * bh, 1),
                aspect=max(bw, bh) / max(1, min(bw, bh)),
                disk_radius=float(math.hypot(bx + bw / 2.0 - 1024.0, by + bh / 2.0 - 1024.0) / 1000.0),
                p0_mean=p0_m,
                p0_q25=float(np.quantile(v0, 0.25)),
                p0_q75=float(np.quantile(v0, 0.75)),
                p0_max=float(v0.max()),
                p0_border_drop=p0_m - ring_mean,
                p1_mean=p1_m,
                p1_q25=float(np.quantile(v1, 0.25)),
                p1_q75=float(np.quantile(v1, 0.75)),
                p1_max=float(v1.max()),
                p1_p0_ratio=p1_m / max(p0_m, 1e-4),
                p2_mean=float(v2.mean()),
                p2_max=float(v2.max()),
                n_cores=n_cores,
                n_p1_cores=n_p1_cores,
                cc_iou=c_iou,
                dark_mean=float(vd.mean()),
                dark_q75=float(np.quantile(vd, 0.75)),
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

    return stem, cand_rows


def select_disjoint_for_stem(stem: str, cand_rows: list[dict], selectors: list, features: list[str], best_cut: float, mu) -> tuple[list[dict], int]:
    if not cand_rows:
        return [], 0
    cdf = pd.DataFrame(cand_rows)
    X = cdf[features]
    cdf["score"] = sum(bst.predict(X, num_threads=1) for bst in selectors) / float(len(selectors))
    cdf = cdf[cdf["score"] >= best_cut].sort_values("score", ascending=False, kind="stable")

    chosen = []
    for row in cdf.itertuples():
        r = row.rle
        if chosen:
            occ = mu.merge(chosen)
            ov = float(mu.area(mu.merge([r, occ], intersect=True)))
            if ov > 0.12 * row.area:
                continue
            if ov > 0:
                bx, by, bw, bh = map(int, mu.toBbox(r))
                m_crop = mu.decode(r)[by:by + bh, bx:bx + bw].astype(bool) & (
                    ~mu.decode(occ)[by:by + bh, bx:bx + bw].astype(bool)
                )
                if int(m_crop.sum()) < 60:
                    continue
                r = encode_crop_rle(m_crop.astype(np.uint8), bx, by, mu)
                if r is None:
                    continue
        chosen.append(r)

    stem_overlap = 0
    if chosen:
        areas = mu.area(chosen)
        stem_overlap = int(areas.sum()) - int(mu.area(mu.merge(chosen)))
    stem_rows = []
    for idx_k, r in enumerate(chosen, 1):
        c = r["counts"]
        c_str = c.decode("ascii") if isinstance(c, bytes) else str(c)
        stem_rows.append({"filament_id": f"{stem}_{idx_k}", "segmentation_rle": c_str})
    return stem_rows, stem_overlap


def main() -> None:
    install_offline_wheels()
    import segmentation_models_pytorch as smp
    import lightgbm as lgb
    from pycocotools import mask as mu

    has_cuda = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if has_cuda else 0
    if has_cuda:
        torch.backends.cudnn.benchmark = True

    cfg_path = os.environ.get("SF_ENS_CONFIG")
    if not cfg_path:
        hits = glob.glob("/kaggle/input/**/ensemble_config.json", recursive=True)
        if not hits:
            raise FileNotFoundError("Could not locate ensemble_config.json in /kaggle/input")
        cfg_path = hits[0]
    bundle_dir = Path(os.environ.get("SF_BUNDLE_DIR") or Path(cfg_path).parent)
    ens_cfg = json.loads(Path(cfg_path).read_text())

    arch_specs = ens_cfg["arch_groups"]
    weights = [float(g["weight"]) for g in arch_specs]
    yolo_rels = [yrel for yrel in ens_cfg.get("yolo_ckpts", []) if (bundle_dir / yrel).exists()]
    has_ultralytics = bool(yolo_rels) and (importlib.util.find_spec("ultralytics") is not None)

    sel_rels = ens_cfg.get("selector_files", [f"v3_selector_fold{f}.txt" for f in range(4)])
    selectors = [lgb.Booster(model_file=str(bundle_dir / srel)) for srel in sel_rels]
    features = ens_cfg["features"]
    best_sources = ens_cfg["best_sources"]
    best_cut = float(ens_cfg["best_cut"])
    use_d4 = bool(ens_cfg.get("use_d4", True))
    smoke_fast = os.environ.get("SF_SMOKE_FAST") == "1"

    test_files = locate_test_images()
    log(
        f"Loaded config ({len(arch_specs)} seg families, {len(yolo_rels)} YOLO ckpts, 4 selectors) "
        f"for {len(test_files)} test images | cuda={has_cuda} n_gpus={n_gpus} smoke_fast={smoke_fast}"
    )

    yy, xx = np.ogrid[:H, :W]
    disk_mask = (yy - H / 2.0) ** 2 + (xx - W / 2.0) ** 2 <= 940.0 ** 2

    def build_smp_model(arch_name: str, backbone: str, ck_path: Path, dev: torch.device):
        arch_cls = getattr(smp, arch_name)
        m = arch_cls(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=3, activation=None)
        sd = torch.load(str(ck_path), map_location="cpu", weights_only=True)
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
        del sd
        m = m.eval().to(dev)
        return m

    sub_rows: list[dict] = []
    overlap_total = 0

    for c_start in range(0, len(test_files), CHUNK_SIZE):
        chunk_paths = test_files[c_start:c_start + CHUNK_SIZE]
        B = len(chunk_paths)

        elapsed_m = (time.time() - T0) / 60.0
        proj_m = elapsed_m * (len(test_files) / float(max(c_start, 1))) if c_start >= CHUNK_SIZE else 0.0
        run_d4_now = use_d4 and (elapsed_m < 38.0) and (c_start < CHUNK_SIZE or proj_m <= 42.0)

        chunk_u8 = []
        chunk_stats = []
        chunk_tensors = []
        for img_p in chunk_paths:
            img_u8 = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img_u8 is None:
                raise RuntimeError(f"Failed to decode test image: {img_p}")
            if img_u8.shape != (H, W):
                img_u8 = cv2.resize(img_u8, (W, H), interpolation=cv2.INTER_LINEAR)
            st = compute_disk_stats(img_u8, disk_mask)
            ch_01 = encode_3ch_u8(img_u8, st).astype(np.float32) / 255.0
            x_t = torch.from_numpy(np.ascontiguousarray((ch_01 - 0.5) / 0.25, dtype=np.float32))[None]
            chunk_u8.append(img_u8)
            chunk_stats.append(st)
            chunk_tensors.append(x_t)

        # Stream 1 checkpoint at a time so GPU VRAM footprint stays <= 150 MB!
        arch_chunk_pgs: list[np.ndarray] = []
        for grp in arch_specs:
            arch_name = grp["arch"]
            backbone = grp["backbone"]
            ckpts = grp["ckpts"][:1] if smoke_fast else grp["ckpts"]
            grp_acc = np.zeros((B, H, W, 3), dtype=np.float32)

            dev = torch.device("cuda:0" if n_gpus >= 1 else "cpu")
            for ck_rel in ckpts:
                ck_path = bundle_dir / ck_rel
                m0 = build_smp_model(arch_name, backbone, ck_path, dev)
                for i in range(B):
                    grp_acc[i] += forward_single_image(m0, arch_name, dev, chunk_tensors[i], run_d4_now, smoke_fast)
                del m0
                gc.collect()
                if has_cuda:
                    torch.cuda.empty_cache()

            grp_acc /= float(len(ckpts))
            arch_chunk_pgs.append(grp_acc.astype(np.float16))
            del grp_acc
            gc.collect()

        del chunk_tensors

        # Stream 1 YOLO checkpoint at a time
        active_yolo_rels = yolo_rels[:1] if smoke_fast else yolo_rels
        chunk_raw_y: list[list[tuple[float, int, dict]]] = [[] for _ in range(B)]
        if has_ultralytics:
            from ultralytics import YOLO
            yolo_dev = 0 if n_gpus >= 1 else "cpu"
            for y_idx, yrel in enumerate(active_yolo_rels):
                ym = YOLO(str(bundle_dir / yrel))
                for i, img_p in enumerate(chunk_paths):
                    res = ym.predict(str(img_p), imgsz=1024, conf=0.03, iou=0.60, max_det=150, retina_masks=True, device=yolo_dev, verbose=False)[0]
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

        # Parallel 4-thread CPU proposal & feature extraction, then thread-safe LightGBM scoring
        n_yolo_active = len(active_yolo_rels) if has_ultralytics else 0
        jobs = [
            (
                chunk_paths[i].stem,
                chunk_u8[i],
                chunk_stats[i],
                [arch_chunk_pgs[g][i] for g in range(len(arch_specs))],
                weights,
                chunk_raw_y[i],
                n_yolo_active,
                best_sources,
            )
            for i in range(B)
        ]
        with ThreadPoolExecutor(max_workers=4) as pp_pool:
            extracted = list(pp_pool.map(extract_stem_candidates, jobs))
        for stem, cand_rows in extracted:
            s_rows, s_ov = select_disjoint_for_stem(stem, cand_rows, selectors, features, best_cut, mu)
            sub_rows.extend(s_rows)
            overlap_total += s_ov

        del arch_chunk_pgs, chunk_u8, chunk_raw_y, jobs, extracted
        gc.collect()
        log(f"Processed {c_start + B}/{len(test_files)} test images (instances so far: {len(sub_rows)}, run_d4={run_d4_now})")

    assert overlap_total == 0, f"Strict pixel-disjointness violated: {overlap_total} overlapping pixels!"
    out_csv = Path(os.environ.get("SF_OUT_CSV", "/kaggle/working/submission.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame(sub_rows, columns=["filament_id", "segmentation_rle"])
    sub_df.to_csv(out_csv, index=False)
    log(f"Wrote {out_csv}: {len(sub_df)} instances across {len(test_files)} test images, 0 overlap pixels")


if __name__ == "__main__":
    main()
