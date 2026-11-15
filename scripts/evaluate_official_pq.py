"""
Official Host Metric Verification & Diagnostic Plotter (`evaluate_official_pq.py`).

Implements the exact Panoptic Quality (`PQ = SQ * RQ` at `IoU > 0.50`), IoU histogram,
Dice histogram, and `1:n / n:1` matching degree bar chart from the official competition
`self-evaluation-notebook.ipynb`, and verifies strict pixel-disjointness (`0` overlap pixels).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pycocotools import mask as mu

H = W = 2048


def build_gt_dataframe_from_coco(coco_json_path: Path, keep_stems: set[str] | None = None) -> pd.DataFrame:
    coco = json.loads(coco_json_path.read_text())
    anns_by_img: dict[str, list[dict]] = {}
    for a in coco["annotations"]:
        seg = a.get("segmentation")
        if not seg:
            continue
        rle = mu.merge(mu.frPyObjects(seg, H, W))
        anns_by_img.setdefault(a["image_id"], []).append(rle)

    rows = []
    annot_counter_per_stem: dict[str, int] = {}
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        if keep_stems is not None and stem not in keep_stems:
            continue
        a_idx = annot_counter_per_stem.get(stem, 0) + 1
        annot_counter_per_stem[stem] = a_idx
        annot_img_id = f"A{a_idx}-{stem}"
        rles = anns_by_img.get(im["id"], [])
        if not rles:
            # Preserve empty-GT annotator records via sentinel row with empty RLE
            rows.append({"filament_id": f"{annot_img_id}_0", "segmentation_rle": ""})
        else:
            for k, r in enumerate(rles, 1):
                c = r["counts"]
                c_str = c.decode("ascii") if isinstance(c, bytes) else str(c)
                rows.append({"filament_id": f"{annot_img_id}_{k}", "segmentation_rle": c_str})
    return pd.DataFrame(rows)


def evaluate_and_plot(coco_json: Path, pred_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.read_csv(pred_csv)
    pred_df["stem"] = pred_df["filament_id"].map(lambda s: str(s).rsplit("_", 1)[0])
    pred_by_stem: dict[str, list[dict]] = {}
    overlap_pixels = 0
    for stem, grp in pred_df.groupby("stem", sort=False):
        rles = [
            {"size": [H, W], "counts": str(c).encode("ascii")}
            for c in grp["segmentation_rle"].tolist()
            if isinstance(c, str) and len(c) > 0
        ]
        if rles:
            areas = mu.area(rles)
            overlap_pixels += int(areas.sum()) - int(mu.area(mu.merge(rles)))
        pred_by_stem[stem] = rles

    coco = json.loads(coco_json.read_text())
    anns_by_img: dict[str, list[dict]] = {}
    for a in coco["annotations"]:
        if a.get("segmentation"):
            anns_by_img.setdefault(a["image_id"], []).append(mu.merge(mu.frPyObjects(a["segmentation"], H, W)))

    tp_ious: list[float] = []
    pos_ious: list[float] = []
    pos_dices: list[float] = []
    gt_degrees: list[int] = []
    pred_degrees: list[int] = []
    fp_count = 0
    fn_count = 0

    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        g_rles = anns_by_img.get(im["id"], [])
        p_rles = pred_by_stem.get(stem, [])
        n_gt = len(g_rles)
        n_pred = len(p_rles)

        if n_gt == 0:
            fp_count += n_pred
            pred_degrees.extend([0] * n_pred)
            continue
        if n_pred == 0:
            fn_count += n_gt
            gt_degrees.extend([0] * n_gt)
            continue

        iou_mat = np.asarray(mu.iou(g_rles, p_rles, [0] * n_pred), dtype=np.float64)
        dice_mat = np.where(iou_mat > 0.0, 2.0 * iou_mat / (1.0 + iou_mat), 0.0)

        pos_mask = iou_mat > 0.0
        if np.any(pos_mask):
            pos_ious.extend(iou_mat[pos_mask].tolist())
            pos_dices.extend(dice_mat[pos_mask].tolist())

        gt_degrees.extend(pos_mask.sum(axis=1).tolist())
        pred_degrees.extend(pos_mask.sum(axis=0).tolist())

        hit_mat = iou_mat > 0.50
        tp_ious.extend(iou_mat[hit_mat].tolist())
        fp_count += int((hit_mat.sum(axis=0) == 0).sum())
        fn_count += int((hit_mat.sum(axis=1) == 0).sum())

    tp_count = len(tp_ious)
    den = tp_count + 0.5 * fp_count + 0.5 * fn_count
    pq = float(sum(tp_ious) / den) if den > 0 else 0.0
    sq = float(sum(tp_ious) / tp_count) if tp_count > 0 else 0.0
    rq = float(tp_count / den) if den > 0 else 0.0

    # Generate host-style diagnostic vector PDF plot (`IoU`, `Dice`, `1:n / n:1`)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), dpi=300)

    for ax, vals, title, xlab in [
        (axes[0], np.asarray(pos_ious), "Pairwise IoU Distribution (> 0)", "IoU"),
        (axes[1], np.asarray(pos_dices), "Pairwise Dice Distribution (> 0)", "Dice"),
    ]:
        bins = np.linspace(0.0, 1.0, 41)
        cnts, edges = np.histogram(vals, bins=bins)
        probs = cnts / max(cnts.sum(), 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, probs, width=0.024, edgecolor="black", linewidth=0.4, alpha=0.78, color="#1f4e79")
        if vals.size > 0:
            m_v = float(vals.mean())
            ax.axvline(m_v, color="#c00000", linestyle="--", linewidth=1.4, label=f"Mean = {m_v:.3f}")
            ax.axvline(0.50 if xlab == "IoU" else (2.0 * 0.5 / 1.5), color="#2e75b6", linestyle=":", linewidth=1.2, label="TP Threshold")
            ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)

    ax3 = axes[2]
    offset = 0.5
    gt_deg_c = Counter(gt_degrees)
    pr_deg_c = Counter(pred_degrees)
    for deg, cnt in sorted(gt_deg_c.items()):
        if deg <= 5:
            ax3.bar(deg + offset, cnt, width=0.78, color="#c00000" if deg == 0 else "#2e8b57", edgecolor="black", linewidth=0.4, alpha=0.8)
    for deg, cnt in sorted(pr_deg_c.items()):
        if deg <= 5:
            ax3.bar(-(deg + offset), cnt, width=0.78, color="#ed7d31" if deg == 0 else "#4472c4", edgecolor="black", linewidth=0.4, alpha=0.8)
    ax3.axvline(0.0, color="black", linewidth=0.9)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ticks = [-(d + offset) for d in range(5, -1, -1)] + [d + offset for d in range(0, 6)]
    tlabels = [f"{d}:1" for d in range(5, -1, -1)] + [f"1:{d}" for d in range(0, 6)]
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(tlabels, rotation=40, fontsize=8)
    ax3.set_title("Many-to-One & One-to-Many Overlap Links", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Matching Multiplicity (IoU > 0)", fontsize=9)
    ax3.set_ylabel("Instance Count", fontsize=9)
    ax3.legend(
        handles=[
            Patch(facecolor="#2e8b57", edgecolor="black", label=f"GT Hits: {sum(v for k, v in gt_deg_c.items() if k > 0)}"),
            Patch(facecolor="#c00000", edgecolor="black", label=f"GT Misses: {gt_deg_c.get(0, 0)}"),
            Patch(facecolor="#4472c4", edgecolor="black", label=f"Pred Hits: {sum(v for k, v in pr_deg_c.items() if k > 0)}"),
            Patch(facecolor="#ed7d31", edgecolor="black", label=f"Pred Misses: {pr_deg_c.get(0, 0)}"),
        ],
        fontsize=7.5,
    )

    fig.tight_layout()
    fig.savefig(out_dir / "official_pq_diagnostics.pdf")
    plt.close(fig)

    metrics = {
        "PQ": pq,
        "SQ": sq,
        "RQ": rq,
        "TP": tp_count,
        "FP": fp_count,
        "FN": fn_count,
        "overlap_pixels": overlap_pixels,
        "n_predictions": int(len(pred_df)),
    }
    (out_dir / "official_pq_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    assert overlap_pixels == 0, f"Found {overlap_pixels} overlapping pixels!"
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-json", type=Path, required=True)
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_out"))
    args = ap.parse_args()
    evaluate_and_plot(args.coco_json, args.pred_csv, args.out_dir)


if __name__ == "__main__":
    main()
