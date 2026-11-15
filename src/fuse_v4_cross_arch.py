"""
Cross-Generation Panoptic Fusion (`v3` 3-Channel + `v4` 4-Channel Seam + 3-Ch Dinkelbach YOLO + Exact MWIS).

Key Upgrades in v4.6:
  1. Channel-Aware Cross-Generation Blending:
     - `p0` (Dinkelbach compromise silhouette): weighted toward 4-ch `v4` models (`p0_weights`) to preserve
       unbloated Dinkelbach contours (`SQ ~ 0.688`) with zero double-shaving.
     - `p1, p2` (human consensus & interior distance spine): full 6-architecture blend (`weights`).
     - `p3` (5px repulsion seam): 4-ch `v4` blend.
     - `p0_v3` (Tversky soft-union recall channel): tracked explicitly from 3-ch `v3` families to expose
       `p0_v3_mean` and `p0_v3_gap = p0_v3_mean - p0_mean` (neural multi-annotator ambiguity signal).
  2. Unshaved Seam-Cut `src=0` + Unsplit `src=1` + Optional Tight `src=8`:
     - Eliminates `calibrate_crop_mask` double-shaving on `v4` Dinkelbach masks (`+0.0081` raw CC PQ).
  3. Exact Dinkelbach MWIS Split Cost (`cut` for singletons, `split_cost` for `1-vs-k` conflicts).
  4. Thread-local persistent ZipFile handles (~10x faster OOF & 180-test verification).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from pycocotools import mask as mu

cv2.setNumThreads(1)

H = W = 2048
T0 = time.time()
_TLS = threading.local()

BASE_FEATURES = [
    "source", "area", "log_area", "bw", "bh", "fill", "aspect", "stroke_width", "disk_radius",
    "p0_mean", "p0_q25", "p0_q75", "p0_max", "p0_border_drop",
    "p0_v3_mean", "p0_v3_gap",
    "p1_mean", "p1_q25", "p1_q75", "p1_max", "p1_p0_ratio",
    "p2_mean", "p2_max",
    "p3_mean", "p3_max", "p3_ring_max",
    "n_cores", "n_p1_cores",
    "cc_iou", "image_cc_count",
    "p0_arch_std", "p1_arch_min",
    "yolo_iou", "yolo_conf",
    "n_yolo_sub", "yolo_contain", "yolo_span_ccs", "yolo_area_ratio",
    # Within-stem relational hierarchy features (computed identically at train & test time)
    "n_overlap_gt15", "n_strict_children", "children_area_sum_ratio", "children_p1_wmean",
    "has_strict_parent", "max_parent_area_ratio", "max_parent_p1",
    "best_alt_iou", "p0_vs_best_alt", "p1_vs_best_alt",
    "nearest_disjoint_bbox_gap", "stem_prop_count", "stem_p1_rank_pct", "stem_area_share",
    "src_is_0", "src_is_1", "src_is_4", "src_is_6", "src_is_7", "src_is_8", "src_is_9", "src_is_10",
]

YOLO_FEATURE_SET = {
    "yolo_iou", "yolo_conf", "n_yolo_sub", "yolo_contain",
    "yolo_span_ccs", "yolo_area_ratio", "src_is_4", "src_is_6",
}
SEG_FEATURES = [c for c in BASE_FEATURES if c not in YOLO_FEATURE_SET]


def log(*args) -> None:
    print(f"[{(time.time() - T0) / 60:6.2f}m]", *args, flush=True)


def _read_png_from_zip(zp: str, member: str) -> np.ndarray:
    cache = getattr(_TLS, "zip_cache", None)
    if cache is None:
        cache = {}
        _TLS.zip_cache = cache
    zf = cache.get(zp)
    if zf is None:
        zf = zipfile.ZipFile(zp, "r")
        cache[zp] = zf
    buf = zf.read(member)
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)


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
    """Extract 2 local spine tip coordinates + outward tip tangent vectors for curved filaments."""
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
    """Mutate `rows` in place to attach within-stem proposal-hierarchy & neighborhood features."""
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


def extract_candidates_from_maps(
    stem: str,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p0_v3: np.ndarray,
    seam_w_sum: float,
    p0_std_map: np.ndarray,
    p1_min_map: np.ndarray,
    yolo_items: list[tuple[float, str]] | None,
    gt_sets: list[list[dict]] | None,
) -> list[dict]:
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
        # Family 0: Unshaved Dinkelbach seam-cut component (no double-shaving!)
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

        # Family 8: Optional Calibrated / Anti-Bloat Tight Consensus (`p1 >= 0.44` trim, `p1 >= 0.64`, `p2 >= 0.23`)
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

    # Family 9: Diffuse / Faint Recall Hysteresis (`p0_rec >= 0.25` seeded by `max(p0_rec) >= 0.36 & max(p1) >= 0.25`)
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

    # Family 10: Oriented Principal-Tip-Tangent Linker across 4..46px gaps (2- & 3-segment chains)
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

    yolo_rles, yolo_confs, yolo_areas = [], [], []
    if yolo_items:
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k3_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for sc, c_str in yolo_items:
            if sc < 0.03:
                continue
            yr = {"size": [H, W], "counts": c_str.encode("ascii") if isinstance(c_str, str) else c_str}
            ya = float(mu.area(yr))
            if ya < 60:
                continue
            yolo_rles.append(yr)
            yolo_confs.append(float(sc))
            yolo_areas.append(ya)

            bx, by, bw, bh = map(int, mu.toBbox(yr))
            if bw > 0 and bh > 0 and sc >= 0.12:
                x0, y0 = max(0, bx - 4), max(0, by - 4)
                x1, y1 = min(W, bx + bw + 4), min(H, by + bh + 4)
                ym = mu.decode(yr)[y0:y1, x0:x1]
                ref = ((cv2.erode(ym, k5) > 0) | ((cv2.dilate(ym, k5) > 0) & ((p0[y0:y1, x0:x1] >= 0.45) | (p1[y0:y1, x0:x1] >= 0.42)))).astype(np.uint8)
                ref = fill_holes_crop(ref)
                ra = int(ref.sum())
                if ra >= 60 and 0.55 <= ra / max(ya, 1.0) <= 1.45:
                    rr = encode_crop_rle(ref, x0, y0)
                    if rr is not None:
                        proposals.append((4, rr, x0, y0, ref, float(sc)))

                if sc >= 0.16 and cc0_items:
                    spanned_ids = []
                    lab0_crop = lab0[y0:y1, x0:x1]
                    for cid in np.unique(lab0_crop[ym > 0]):
                        if cid == 0:
                            continue
                        c_tot = int(st0[cid, cv2.CC_STAT_AREA])
                        if c_tot >= 60:
                            c_in = int(((lab0_crop == cid) & (ym > 0)).sum())
                            if c_in >= 0.55 * c_tot:
                                spanned_ids.append(int(cid))
                    if len(spanned_ids) >= 2:
                        ux0 = min([x0] + [int(st0[cid, cv2.CC_STAT_LEFT]) for cid in spanned_ids])
                        uy0 = min([y0] + [int(st0[cid, cv2.CC_STAT_TOP]) for cid in spanned_ids])
                        ux1 = max([x1] + [int(st0[cid, cv2.CC_STAT_LEFT] + st0[cid, cv2.CC_STAT_WIDTH]) for cid in spanned_ids])
                        uy1 = max([y1] + [int(st0[cid, cv2.CC_STAT_TOP] + st0[cid, cv2.CC_STAT_HEIGHT]) for cid in spanned_ids])
                        u_mask = np.isin(lab0[uy0:uy1, ux0:ux1], spanned_ids)
                        ym_u = mu.decode(yr)[uy0:uy1, ux0:ux1]
                        seam_g_u = (p3[uy0:uy1, ux0:ux1] < 0.42) if seam_w_sum > 0 else True
                        bridge_spine = (cv2.erode(ym_u, k3_e) > 0) & (p0_rec[uy0:uy1, ux0:ux1] >= 0.15) & seam_g_u
                        u_cal = fill_holes_crop((u_mask | bridge_spine).astype(np.uint8))
                        if int(u_cal.sum()) >= 80:
                            ru = encode_crop_rle(u_cal, ux0, uy0)
                            if ru is not None:
                                proposals.append((6, ru, ux0, uy0, u_cal, float(sc)))

    if not proposals:
        return []

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

    values = np.zeros(len(dedup), dtype=np.float64)
    if gt_sets is not None:
        for gs in gt_sets:
            if gs:
                iou_m = np.asarray(mu.iou(dedup_rles, gs, [0] * len(gs)))
                values += np.where(iou_m > 0.50, iou_m, 0.0).max(axis=1)

    k3 = np.ones((3, 3), dtype=np.uint8)
    rows = []
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
        ring_p3_max = float(p3[pad_y0:pad_y1, pad_x0:pad_x1][ring].max()) if np.any(ring) else 0.0

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

        rows.append(dict(
            stem=stem,
            source=src_id,
            rle=rle,
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
            p3_ring_max=ring_p3_max,
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
            v=float(values[idx_p]),
            A=len(gt_sets) if gt_sets is not None else 0,
        ))

    compute_stem_relational_features(rows)
    return rows


def extract_fused_for_stem(job) -> tuple[str, bytes, list[dict]]:
    stem, arch_zips_and_weights, yolo_items, gt_sets, export_png = job

    n_arch = len(arch_zips_and_weights)
    w_sum = sum(w for _, w, _ in arch_zips_and_weights)
    w0_sum = sum(w0 for _, _, w0 in arch_zips_and_weights)
    p0_acc = np.zeros((H, W), dtype=np.float32)
    p12_acc = np.zeros((H, W, 2), dtype=np.float32)
    seam_acc = np.zeros((H, W), dtype=np.float32)
    seam_w_sum = 0.0
    v3_p0_acc = np.zeros((H, W), dtype=np.float32)
    v3_w_sum = 0.0
    arch_p0s = []
    arch_p1s = []

    for zps, w, w0 in arch_zips_and_weights:
        pg_3 = np.zeros((H, W, 3), dtype=np.float32)
        pg_s = np.zeros((H, W), dtype=np.float32)
        has_4ch = False
        for zp in zps:
            arr = _read_png_from_zip(zp, f"{stem}.png")
            if arr.ndim == 3 and arr.shape[2] == 4:
                has_4ch = True
                pg_3 += arr[:, :, :3].astype(np.float32)
                pg_s += arr[:, :, 3].astype(np.float32)
            else:
                pg_3 += arr[:, :, :3].astype(np.float32)
        pg_3 /= 255.0 * len(zps)
        p0_acc += pg_3[:, :, 0] * (w0 / w0_sum)
        p12_acc += pg_3[:, :, 1:3] * (w / w_sum)
        arch_p0s.append(pg_3[:, :, 0])
        arch_p1s.append(pg_3[:, :, 1])
        if has_4ch:
            seam_acc += (pg_s / (255.0 * len(zps))) * w0
            seam_w_sum += w0
        else:
            v3_p0_acc += pg_3[:, :, 0] * w
            v3_w_sum += w

    p0 = p0_acc
    p1, p2 = p12_acc[:, :, 0], p12_acc[:, :, 1]
    p3 = (seam_acc / seam_w_sum) if seam_w_sum > 0 else np.zeros((H, W), dtype=np.float32)
    p0_v3 = (v3_p0_acc / v3_w_sum) if v3_w_sum > 0 else p0

    if n_arch >= 2:
        p0_std_map = np.std(np.stack(arch_p0s, axis=0), axis=0)
        p1_min_map = np.min(np.stack(arch_p1s, axis=0), axis=0)
    else:
        p0_std_map = np.zeros((H, W), dtype=np.float32)
        p1_min_map = p1

    png_bytes = b""
    if export_png:
        bgra_u8 = np.rint(np.clip(np.dstack([p0, p1, p2, p3]) * 255.0, 0.0, 255.0)).astype(np.uint8)
        ok_png, png_buf = cv2.imencode(".png", bgra_u8)
        png_bytes = png_buf.tobytes() if ok_png else b""

    rows = extract_candidates_from_maps(
        stem=stem,
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        p0_v3=p0_v3,
        seam_w_sum=seam_w_sum,
        p0_std_map=p0_std_map,
        p1_min_map=p1_min_map,
        yolo_items=yolo_items,
        gt_sets=gt_sets,
    )
    return stem, png_bytes, rows


def _mwis_component(weights: np.ndarray, adj_bits: list[int]) -> list[int]:
    """Exact branch-and-bound Maximum-Weight Independent Set on a conflict component."""
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


def mwis_select_disjoint(
    df: pd.DataFrame,
    cut: float,
    allowed_sources: list[int],
    split_cost: float = 0.24,
    max_overlap_frac: float = 0.12,
) -> dict:
    """Select pixel-disjoint instances via connected-component MWIS with Dinkelbach split cost."""
    result = {}
    sub = df[(df["score"] >= cut) & (df["source"].isin(allowed_sources))]
    for stem, grp in sub.groupby("stem", sort=False):
        grp_s = grp.sort_values("score", ascending=False, kind="stable")
        rles = grp_s["rle"].tolist()
        areas = grp_s["area"].to_numpy(dtype=np.float64)
        scores = grp_s["score"].to_numpy(dtype=np.float64)
        m = len(rles)
        if m == 0:
            continue

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
                    # Every candidate in `comp` already passed `score >= cut`: pick top singleton
                    picked_indices.append(max(comp, key=lambda idx: scores[idx]))
            else:
                comp_sorted = sorted(comp, key=lambda idx: -scores[idx])
                sel = [comp_sorted[0]]
                for u in comp_sorted[1:]:
                    if scores[u] > split_cost and all(not conflict[u, v] for v in sel):
                        sel.append(u)
                picked_indices.extend(sel)

        picked_indices.sort(key=lambda idx: -scores[idx])
        chosen = []
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
        result[stem] = chosen
    return result


def score_subset(gt_dict: dict, pred_dict: dict, stems: list[str]) -> dict:
    tp_sum, tp, fp, fn = 0.0, 0, 0, 0
    for s in stems:
        p = pred_dict.get(s, [])
        for g in gt_dict[s]:
            if not g:
                fp += len(p)
                continue
            if not p:
                fn += len(g)
                continue
            iou = np.asarray(mu.iou(g, p, [0] * len(p)))
            hit = iou > 0.50
            tp_sum += float(iou[hit].sum())
            tp += int(hit.sum())
            fp += int((hit.sum(axis=0) == 0).sum())
            fn += int((hit.sum(axis=1) == 0).sum())
    den = tp + 0.5 * fp + 0.5 * fn
    return {
        "PQ": float(tp_sum / den) if den > 0 else 0.0,
        "SQ": float(tp_sum / tp) if tp > 0 else 0.0,
        "RQ": float(tp / den) if den > 0 else 0.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-dirs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="Consensus/spine weights (p1, p2)")
    ap.add_argument("--p0-weights", nargs="+", type=float, default=None, help="Silhouette/seam weights (p0, p3)")
    ap.add_argument("--yolo-oof", type=str, default=None)
    ap.add_argument("--coco-json", type=str, default="filament_comp_research/clean_ann/comp_train.json")
    ap.add_argument("--fold-json", type=str, default="filament_comp_research/clean_4fold_split.json")
    ap.add_argument("--val141-json", type=str, default="filament_comp_research/clean_val_split.json")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--export-stage1-zip", action="store_true", help="Write oof_stage1_blended.zip for Stage-2 ROI")
    ap.add_argument("--verify-test-zips", action="store_true", help="Verify 180-test-stem area & instance statistics")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arch_dirs = [Path(d) for d in args.arch_dirs]
    weights = args.weights if args.weights else [1.0] * len(arch_dirs)
    p0_weights = args.p0_weights if args.p0_weights else list(weights)
    assert len(weights) == len(arch_dirs) and len(p0_weights) == len(arch_dirs)

    coco = json.loads(Path(args.coco_json).read_text())
    fold_map = json.loads(Path(args.fold_json).read_text())["fold_of_stem"]
    val141_stems = set(json.loads(Path(args.val141_json).read_text())["val"])

    anns_by_img: dict[str, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("segmentation"):
            anns_by_img[a["image_id"]].append(mu.merge(mu.frPyObjects(a["segmentation"], H, W)))
    gt_by_stem: dict[str, list[list[dict]]] = defaultdict(list)
    for im in coco["images"]:
        gt_by_stem[Path(im["file_name"]).stem].append(anns_by_img.get(im["id"], []))

    oof_maps: list[dict[str, list[str]]] = []
    test_zips_per_arch: list[list[str]] = []
    arch_specs = []
    for adir, w, w0 in zip(arch_dirs, weights, p0_weights):
        amap: dict[str, list[str]] = defaultdict(list)
        for zp in sorted(adir.glob("oof_fold*.zip")):
            with zipfile.ZipFile(zp) as zf:
                for nm in zf.namelist():
                    amap[nm[:-4]].append(str(zp))
        oof_maps.append(amap)
        test_zips_per_arch.append([str(p) for p in sorted(adir.glob("test_fold*.zip"))])
        m_path = adir / "meta.json"
        if m_path.exists():
            mj = json.loads(m_path.read_text())
            arch_specs.append({"dir": str(adir), "arch": mj.get("arch", "Unet"), "backbone": mj["backbone"], "weight": w, "p0_weight": w0})
        else:
            arch_specs.append({"dir": str(adir), "arch": "Unet", "backbone": adir.name, "weight": w, "p0_weight": w0})

    common_stems = sorted(set.intersection(*(set(m.keys()) for m in oof_maps)))
    log(f"Fusing {len(arch_dirs)} segmentation families across {len(common_stems)} common OOF stems")

    oof_yolo = pickle.loads(Path(args.yolo_oof).read_bytes()) if args.yolo_oof and Path(args.yolo_oof).exists() else {}
    log(f"YOLO OOF loaded: {len(oof_yolo)} stems")

    jobs = [
        (
            s,
            [(oof_maps[a][s], weights[a], p0_weights[a]) for a in range(len(arch_dirs))],
            oof_yolo.get(s, []),
            gt_by_stem[s],
            bool(args.export_stage1_zip),
        )
        for s in common_stems
    ]

    rows = []
    zf_out = zipfile.ZipFile(out_dir / "oof_stage1_blended.zip", "w", zipfile.ZIP_STORED) if args.export_stage1_zip else None
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for idx, (stem, png_bytes, r) in enumerate(pool.map(extract_fused_for_stem, jobs), 1):
                if zf_out is not None and png_bytes:
                    zf_out.writestr(f"{stem}.png", png_bytes)
                rows.extend(r)
                if idx % 200 == 0 or idx == len(jobs):
                    log(f"Extracted fused v3+v4 candidates: {idx}/{len(jobs)} stems ({len(rows)} proposals)")
    finally:
        if zf_out is not None:
            zf_out.close()

    oof_df = pd.DataFrame(rows)
    oof_df["fold"] = oof_df["stem"].map(lambda s: int(fold_map[s]))
    oof_df["score"] = 0.0

    seeds = [2040, 2041, 2042]
    bag_models: dict[int, list[str]] = defaultdict(list)
    seg_bag_models: dict[int, list[str]] = defaultdict(list)
    boosters_all = []
    seg_boosters_all = []
    oof_p_yolo = np.zeros(len(oof_df), dtype=np.float64)
    oof_p_seg = np.zeros(len(oof_df), dtype=np.float64)
    for f in range(4):
        tr = oof_df["fold"] != f
        va = oof_df["fold"] == f
        y_tr = np.clip((oof_df.loc[tr, "v"] / np.maximum(oof_df.loc[tr, "A"], 1)).to_numpy(dtype=np.float64), 0.0, 1.0)
        w_tr = np.maximum(oof_df.loc[tr, "A"].to_numpy(dtype=np.float64), 1.0)
        p_va = np.zeros(int(va.sum()), dtype=np.float64)
        p_va_seg = np.zeros(int(va.sum()), dtype=np.float64)
        sub_idx = 0
        for sd in seeds:
            for obj in ("regression", "cross_entropy"):
                lgb_params = dict(
                    objective=obj,
                    learning_rate=0.025,
                    num_leaves=23,
                    min_data_in_leaf=40,
                    lambda_l2=12.0,
                    feature_fraction=0.80,
                    bagging_fraction=0.85,
                    bagging_freq=1,
                    verbosity=-1,
                    num_threads=1,
                    seed=sd,
                )
                dtrain = lgb.Dataset(oof_df.loc[tr, BASE_FEATURES], label=y_tr, weight=w_tr)
                bst = lgb.train(lgb_params, dtrain, num_boost_round=360)
                p_va += bst.predict(oof_df.loc[va, BASE_FEATURES], num_threads=1) / float(len(seeds) * 2)
                fname = f"selector_fold{f}_m{sub_idx}.txt"
                bst.save_model(str(out_dir / fname))
                if sub_idx == 0:
                    bst.save_model(str(out_dir / f"selector_fold{f}.txt"))
                bag_models[f].append(fname)
                boosters_all.append(bst)

                seg_params = dict(lgb_params, feature_fraction=0.75)
                dtrain_seg = lgb.Dataset(oof_df.loc[tr, SEG_FEATURES], label=y_tr, weight=w_tr)
                bst_seg = lgb.train(seg_params, dtrain_seg, num_boost_round=360)
                p_va_seg += bst_seg.predict(oof_df.loc[va, SEG_FEATURES], num_threads=1) / float(len(seeds) * 2)
                seg_fname = f"seg_selector_fold{f}_m{sub_idx}.txt"
                bst_seg.save_model(str(out_dir / seg_fname))
                seg_bag_models[f].append(seg_fname)
                seg_boosters_all.append(bst_seg)
                sub_idx += 1
        oof_p_yolo[va.to_numpy()] = p_va
        oof_p_seg[va.to_numpy()] = p_va_seg

    yolo_gbdt_weight = 0.60 if oof_yolo else 0.0
    oof_df["score_yolo_gbdt"] = oof_p_yolo
    oof_df["score_seg_gbdt"] = oof_p_seg
    oof_df["score"] = yolo_gbdt_weight * oof_p_yolo + (1.0 - yolo_gbdt_weight) * oof_p_seg

    oof_df.to_pickle(out_dir / "fused_oof_candidates.pkl")

    multi_stems = [s for s in common_stems if len(gt_by_stem[s]) > 1]
    single_stems = [s for s in common_stems if len(gt_by_stem[s]) == 1]
    val141_present = [s for s in common_stems if s in val141_stems]
    fold_stems_map = {f: [s for s in common_stems if int(fold_map[s]) == f] for f in range(4)}

    configs = [
        ("clean_0_1_3_6", [0, 1, 3, 6]),
        ("clean_link_0_1_3_6_10", [0, 1, 3, 6, 10]),
        ("seam_link_0_1_3_6_7_10", [0, 1, 3, 6, 7, 10]),
        ("seam_0_1_2_3_6_7_10", [0, 1, 2, 3, 6, 7, 10]),
        ("seam_link_0_1_3_4_6_7_10", [0, 1, 3, 4, 6, 7, 10]),
    ]
    cuts = (0.19, 0.20, 0.21, 0.22, 0.23, 0.24)
    split_costs = (0.21, 0.24, 0.27)

    report_rows = []
    for name, srcs in configs:
        for split_cost in split_costs:
            for cut in cuts:
                pred = mwis_select_disjoint(oof_df, cut, srcs, split_cost=split_cost)
                s_all = score_subset(gt_by_stem, pred, common_stems)
                s_mul = score_subset(gt_by_stem, pred, multi_stems)
                s_sng = score_subset(gt_by_stem, pred, single_stems)
                s_141 = score_subset(gt_by_stem, pred, val141_present)
                per_fold = [score_subset(gt_by_stem, pred, fold_stems_map[f]) for f in range(4)]
                areas_oof = [float(mu.area(r)) for rs in pred.values() for r in rs]
                mean_area_oof = float(np.mean(areas_oof)) if areas_oof else 0.0
                row = dict(
                    name=name, sources=srcs, cut=cut, split_cost=split_cost,
                    n_pred_oof=len(areas_oof), mean_area_oof=mean_area_oof,
                    oof_707_all=s_all, oof_707_multi=s_mul, oof_707_single=s_sng,
                    oof_val141=s_141, per_fold=per_fold,
                )
                report_rows.append(row)
        best_cfg_here = max((r for r in report_rows if r["name"] == name), key=lambda r: r["oof_707_all"]["PQ"])
        sa = best_cfg_here["oof_707_all"]
        log(
            f"{name:26s} best cut={best_cfg_here['cut']:.2f} split={best_cfg_here['split_cost']:.2f} | "
            f"OOF707 PQ={sa['PQ']:.4f} (SQ={sa['SQ']:.4f} RQ={sa['RQ']:.4f} "
            f"mul={best_cfg_here['oof_707_multi']['PQ']:.4f} sng={best_cfg_here['oof_707_single']['PQ']:.4f}) "
            f"| n={best_cfg_here['n_pred_oof']} area={best_cfg_here['mean_area_oof']:.1f} "
            f"| Val141={best_cfg_here['oof_val141']['PQ']:.4f}"
        )

    nested_tp_sum, nested_tp, nested_fp, nested_fn = 0.0, 0, 0, 0
    nested_picks = []
    for f in range(4):
        def _other_pq(r):
            tp = sum(r["per_fold"][j]["TP"] for j in range(4) if j != f)
            fp = sum(r["per_fold"][j]["FP"] for j in range(4) if j != f)
            fn = sum(r["per_fold"][j]["FN"] for j in range(4) if j != f)
            ts = sum(r["per_fold"][j]["SQ"] * r["per_fold"][j]["TP"] for j in range(4) if j != f)
            den = tp + 0.5 * fp + 0.5 * fn
            return ts / den if den > 0 else 0.0
        pick = max(report_rows, key=_other_pq)
        nested_picks.append((f, pick["name"], pick["cut"], pick["split_cost"], pick["per_fold"][f]["PQ"]))
        pf = pick["per_fold"][f]
        nested_tp_sum += pf["SQ"] * pf["TP"]
        nested_tp += pf["TP"]
        nested_fp += pf["FP"]
        nested_fn += pf["FN"]
    nested_den = nested_tp + 0.5 * nested_fp + 0.5 * nested_fn
    nested_pq = float(nested_tp_sum / nested_den) if nested_den > 0 else 0.0
    log(f"Nested fold picks: {nested_picks}")

    best = max(report_rows, key=lambda r: r["oof_707_all"]["PQ"])
    log(
        f"★ BEST FUSED v3+v4.6 ENSEMBLE: {best['name']} cut={best['cut']:.2f} split_cost={best['split_cost']:.2f} "
        f"-> 707-OOF PQ={best['oof_707_all']['PQ']:.4f} (SQ={best['oof_707_all']['SQ']:.4f} RQ={best['oof_707_all']['RQ']:.4f}) "
        f"| Nested PQ={nested_pq:.4f} | Val141 PQ={best['oof_val141']['PQ']:.4f} "
        f"| OOF_n={best['n_pred_oof']} OOF_area={best['mean_area_oof']:.1f}"
    )

    # Export clean OOF prediction CSV for official host diagnostics & technical report plots
    best_oof_pred = mwis_select_disjoint(oof_df, best["cut"], best["sources"], split_cost=best["split_cost"])
    oof_sub_rows = []
    for s in common_stems:
        for idx_k, r in enumerate(best_oof_pred.get(s, []), 1):
            c = r["counts"]
            c_str = c.decode("ascii") if isinstance(c, bytes) else str(c)
            oof_sub_rows.append({"filament_id": f"{s}_{idx_k}", "segmentation_rle": c_str})
    pd.DataFrame(oof_sub_rows, columns=["filament_id", "segmentation_rle"]).to_csv(out_dir / "oof_submission.csv", index=False)

    if args.verify_test_zips and all(len(tzs) > 0 for tzs in test_zips_per_arch):
        with zipfile.ZipFile(test_zips_per_arch[0][0]) as zf0:
            test_stems = sorted(nm[:-4] for nm in zf0.namelist())
        t_jobs = [
            (
                s,
                [(test_zips_per_arch[a], weights[a], p0_weights[a]) for a in range(len(arch_dirs))],
                [],
                None,
                False,
            )
            for s in test_stems
        ]
        t_rows = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _, _, r in pool.map(extract_fused_for_stem, t_jobs):
                t_rows.extend(r)
        t_df = pd.DataFrame(t_rows)
        t_df["score"] = sum(bst.predict(t_df[SEG_FEATURES], num_threads=1) for bst in seg_boosters_all) / float(len(seg_boosters_all))
        seg_srcs = [s for s in best["sources"] if s not in (4, 6)]
        t_pred = mwis_select_disjoint(t_df, 0.20, seg_srcs, split_cost=best["split_cost"])
        t_areas = [float(mu.area(r)) for rs in t_pred.values() for r in rs]
        log(
            f"[180-TEST SEG-ONLY VERIFICATION] stems={len(t_pred)}/{len(test_stems)} | "
            f"rows={len(t_areas)} | mean_area={np.mean(t_areas):.1f} | med_area={np.median(t_areas):.1f} | "
            f"total_Mpx={np.sum(t_areas)/1e6:.3f}"
        )

    selector_files = [fn for f in range(4) for fn in bag_models[f]]
    seg_selector_files = [fn for f in range(4) for fn in seg_bag_models[f]]
    ensemble_cfg = dict(
        arch_specs=arch_specs,
        weights=weights,
        p0_weights=p0_weights,
        has_yolo=bool(oof_yolo),
        features=BASE_FEATURES,
        seg_features=SEG_FEATURES,
        selector_files=selector_files,
        seg_selector_files=seg_selector_files,
        yolo_gbdt_weight=yolo_gbdt_weight,
        best_sources=best["sources"],
        best_cut=best["cut"],
        split_cost=best["split_cost"],
        oof_PQ=best["oof_707_all"]["PQ"],
        oof_SQ=best["oof_707_all"]["SQ"],
        oof_RQ=best["oof_707_all"]["RQ"],
        oof_TP=best["oof_707_all"]["TP"],
        oof_FP=best["oof_707_all"]["FP"],
        oof_FN=best["oof_707_all"]["FN"],
        oof_multi_PQ=best["oof_707_multi"]["PQ"],
        oof_single_PQ=best["oof_707_single"]["PQ"],
        oof_707_PQ=best["oof_707_all"]["PQ"],
        nested_oof_PQ=nested_pq,
        nested_oof_707_PQ=nested_pq,
        val141_PQ=best["oof_val141"]["PQ"],
    )
    (out_dir / "ensemble_config.json").write_text(json.dumps(ensemble_cfg, indent=2))
    (out_dir / "ensemble_report.json").write_text(json.dumps(dict(best=best, nested_pq=nested_pq, nested_picks=nested_picks, grid=report_rows), indent=2))


if __name__ == "__main__":
    main()
