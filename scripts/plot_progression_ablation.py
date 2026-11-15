"""
Generate publication-grade 2-panel vector PDF (`progression_ablation.pdf`) for the technical report:
  (a) Leak-free 707-stem OOF Panoptic Quality (`PQ`), Segmentation Quality (`SQ`), and Val-141 `PQ` progression.
  (b) Multi-Annotator (`296` stems) vs Single-Annotator (`411` stems) OOF `PQ` across supervision regimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("solar_filament_2026_prize_release/technical_report/figures/progression_ablation.pdf"),
    )
    args = ap.parse_args()
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.4), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel (a): Controlled 707-OOF PQ progression
    labels = [
        "Baseline\n(Dynamic)",
        "v3 YOLO11m\n(1-ch)",
        "v4 YOLO11m\n(3-ch Dink.)",
        "v3 EffNet-B4\n(1-of-K Union)",
        "v4 RegNetY-064\n(4-ch FPN)",
        "v4 HRNet-W32\n(4-ch U-Net)",
        "v3 4-Family\nEnsemble",
        "v4 ConvNeXt-S\n(4-ch Solo)",
        "v4.8 6-Family +\n52-GBDT + MWIS",
    ]
    oof_pqs = [0.3680, 0.3627, 0.4031, 0.4248, 0.4437, 0.4523, 0.4541, 0.4570, 0.4641]
    colors = [
        "#8c8c8c",
        "#b4c7e7",
        "#5b9bd5",
        "#f4b183",
        "#70ad47",
        "#548235",
        "#ed7d31",
        "#2e75b6",
        "#1f4e79",
    ]

    x = np.arange(len(labels))
    bars = ax1.bar(x, oof_pqs, width=0.64, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    for bar, v in zip(bars, oof_pqs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.0035,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold" if v >= 0.4570 else "normal",
        )
    ax1.set_ylim(0.33, 0.485)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.2, rotation=22, ha="right")
    ax1.set_ylabel("707-Stem Out-Of-Fold PQ", fontsize=9.5, fontweight="bold")
    ax1.set_title("(a) Leak-Free 4-Fold OOF Progression (707 Stems, 1,154 Records)", fontsize=9.8, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    # Panel (b): Multi-Annotator vs Single-Annotator & SQ comparison
    regimes = [
        "v3 EffNet-B4\n(Union)",
        "v3 4-Family\nEnsemble",
        "v4 ConvNeXt-S\n(Dink. Solo)",
        "v4.8 6-Family\n+ 52-GBDT MWIS",
    ]
    mul_pq = [0.4112, 0.4482, 0.4539, 0.4629]
    sng_pq = [0.4479, 0.4641, 0.4622, 0.4660]
    val141 = [0.4385, 0.4648, 0.4696, 0.4762]

    xr = np.arange(len(regimes))
    w = 0.25
    b1 = ax2.bar(xr - w, mul_pq, width=w, color="#1f4e79", edgecolor="black", linewidth=0.5, label="Multi-Annotator (296)")
    b2 = ax2.bar(xr, sng_pq, width=w, color="#2e8b57", edgecolor="black", linewidth=0.5, label="Single-Annotator (411)")
    b3 = ax2.bar(xr + w, val141, width=w, color="#c55a11", edgecolor="black", linewidth=0.5, label="Holdout Val-141")

    for grp in (b1, b2, b3):
        for bar in grp:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.0025, f"{h:.3f}", ha="center", va="bottom", fontsize=6.6)

    ax2.set_ylim(0.39, 0.498)
    ax2.set_xticks(xr)
    ax2.set_xticklabels(regimes, fontsize=7.8)
    ax2.set_ylabel("Panoptic Quality (PQ)", fontsize=9.5, fontweight="bold")
    ax2.set_title("(b) Multi-Annotator, Single-Annotator & Val-141 Breakdown", fontsize=9.8, fontweight="bold")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    fig.tight_layout()
    fig.savefig(args.out_pdf)
    plt.close(fig)
    print(f"[+] Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
