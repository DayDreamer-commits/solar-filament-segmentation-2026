"""Export pixel-disjoint Out-Of-Fold predictions from `fused_oof_candidates.pkl` or `oof_dual_roi_scored.pkl` to COCO RLE CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fuse_v4_cross_arch import mwis_select_disjoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-pkl", type=Path, required=True)
    ap.add_argument("--config-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_pickle(args.cand_pkl)
    cfg = json.loads(args.config_json.read_text())
    if "meta_score" in df.columns and "best" in cfg:
        df["score"] = df["meta_score"]
        if cfg["best"].get("use_refined_mask", False) and "rle_refined" in df.columns:
            has_r = df["rle_refined"].notna()
            df.loc[has_r, "rle"] = df.loc[has_r, "rle_refined"]
        cut = float(cfg["best"]["cut"])
        srcs = list(cfg["best"]["sources"])
    else:
        cut = float(cfg["best_cut"])
        srcs = list(cfg["best_sources"])

    pred_dict = mwis_select_disjoint(df, cut=cut, allowed_sources=srcs)
    rows = []
    for stem in sorted(pred_dict.keys()):
        for k, r in enumerate(pred_dict[stem], 1):
            c = r["counts"]
            c_str = c.decode("ascii") if isinstance(c, bytes) else str(c)
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": c_str})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out_csv, index=False)
    print(f"[+] Exported {len(rows)} OOF instances across {len(pred_dict)} stems -> {args.out_csv}")


if __name__ == "__main__":
    main()
