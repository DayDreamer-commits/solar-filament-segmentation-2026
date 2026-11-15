#!/usr/bin/env python3
"""Generate the vector architecture flowchart (Figure 1) for the technical report."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def build_pipeline_figure(out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.6, 2.55), dpi=300)
    ax.set_xlim(0.0, 13.6)
    ax.set_ylim(0.0, 2.55)
    ax.axis("off")

    def draw_box(x: float, y: float, w: float, h: float, title: str, body: str,
                 edge: str, face: str, title_color: str) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            linewidth=1.5, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2.0, y + h - 0.24, title,
            ha="center", va="top", fontsize=9.1, fontweight="bold",
            color=title_color, linespacing=1.18,
        )
        ax.text(
            x + w / 2.0, y + 0.16, body,
            ha="center", va="bottom", fontsize=7.9,
            color="#1f2937", linespacing=1.22,
        )

    def draw_arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.30",
                            lw=1.5, color="#374151"),
        )

    # Stage 1: Input & 3-Ch Normalization
    draw_box(
        0.15, 0.30, 2.15, 1.95,
        "1. H-Alpha Input &\n3-Ch Disk Norm",
        "2048x2048 GONG Disk\nDisk P10 / P50 / P90\n16x16 CLAHE Contrast\nLimb Flat (sigma=18 px)",
        edge="#2563eb", face="#eff6ff", title_color="#1d4ed8",
    )

    # Stage 2a: 6-Family Seg Stream (top)
    draw_box(
        2.62, 1.35, 2.60, 1.05,
        "2a. 6-Family Seg Stream",
        "v4 (4-Ch): ConvNeXt, HRNet, RegNetY\nv3 (3-Ch): EffB4, D169, ResNeXt50",
        edge="#16a34a", face="#f0fdf4", title_color="#15803d",
    )

    # Stage 2b: 3-Ch YOLO11m-Seg (bottom)
    draw_box(
        2.62, 0.15, 2.60, 1.05,
        "2b. 3-Ch YOLO11m-Seg",
        "4-Fold 1024 px Instance Detector\nDinkelbach Polygons, w / sqrt(kM)",
        edge="#d97706", face="#fffbeb", title_color="#b45309",
    )

    # Stage 3: Topological Proposal Graph
    draw_box(
        5.54, 0.30, 2.48, 1.95,
        "3. Topological\nProposal Graph",
        "Unshaved S = {0,1,2,3,6,7,10}\nSeam Cuts & Spine Voronoi\nYOLO Unions & Tip Links\n60 Relational Features",
        edge="#7c3aed", face="#f5f3ff", title_color="#6d28d9",
    )

    # Stage 4: Dual-Anchored LightGBM
    draw_box(
        8.34, 0.30, 2.48, 1.95,
        "4. Dual-Anchored\nLightGBM Rankers",
        "24x Full Boosters (60-feat)\n24x Seg-Only Boosters (52-feat)\nu = 0.60 u_full + 0.40 u_seg\nCalibrated Expected IoU",
        edge="#dc2626", face="#fef2f2", title_color="#b91c1c",
    )

    # Stage 5: Exact Dinkelbach MWIS
    draw_box(
        11.14, 0.30, 2.31, 1.95,
        "5. Exact Dinkelbach\nMWIS Packing",
        "Branch-and-Bound MWIS\ntau = 0.22, split = 0.24\nStrict 0-Pixel Overlap\nRLE submission.csv",
        edge="#0d9488", face="#f0fdfa", title_color="#0f766e",
    )

    # Arrows
    draw_arrow(2.35, 1.55, 2.58, 1.82)
    draw_arrow(2.35, 1.00, 2.58, 0.72)
    draw_arrow(5.27, 1.82, 5.50, 1.55)
    draw_arrow(5.27, 0.72, 5.50, 1.00)
    draw_arrow(8.07, 1.275, 8.30, 1.275)
    draw_arrow(10.87, 1.275, 11.10, 1.275)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[1]
        / "technical_report"
        / "figures"
        / "pipeline_architecture.pdf"
    )
    build_pipeline_figure(target)
    print(f"Saved {target}")
