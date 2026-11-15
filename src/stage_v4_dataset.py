"""
Stage v4.6 Cross-Generation Panoptic Models (`v3` 3-ch + `v4` 4-ch + 3-ch `YOLO` + MWIS)
for Kaggle Private Dataset Upload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def infer_arch_and_backbone(model_dir: Path) -> tuple[str, str, bool]:
    for fname in ("val_report.json", "meta.json"):
        p = model_dir / fname
        if p.exists():
            d = json.loads(p.read_text())
            if "backbone" in d:
                backbone = str(d["backbone"])
                arch = str(d.get("arch", ""))
                if not arch:
                    arch = "UnetPlusPlus" if (backbone == "densenet169" or "upp" in model_dir.name.lower()) else "Unet"
                is_v4 = ("v4" in model_dir.name.lower()) or ("tu-" in backbone) or ("regnety" in backbone)
                return arch, backbone, is_v4
    raise FileNotFoundError(f"Could not find backbone metadata in {model_dir}")


def sanitize_tag(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True, help="Directories with meta.json / model_fold{0..3}.pt")
    ap.add_argument("--fuse-dir", type=str, required=True, help="Directory with ensemble_config.json and selector_*.txt")
    ap.add_argument("--yolo-dir", type=str, default="", help="Optional directory with yolo_fold{0..3}.pt")
    ap.add_argument("--dual-roi-dir", type=str, default="", help="Optional directory with roi_dual_fold{0..3}.pt")
    ap.add_argument("--stage-dir", type=str, default="v4_models_upload")
    ap.add_argument("--dataset-id", type=str, default="daydreamerthinks/solar-filament-v4-panoptic-models")
    ap.add_argument("--tta-views", type=int, default=4, choices=[1, 2, 4, 8], help="Default TTA views for v4 4-ch models")
    ap.add_argument("--v3-tta-views", type=int, default=2, choices=[1, 2, 4], help="TTA views for v3 3-ch support models")
    ap.add_argument("--skip-seg-ckpts", action="store_true", help="Omit seg .pt files when mounted via external datasets")
    ap.add_argument("--skip-yolo-ckpts", action="store_true", help="Omit yolo .pt files when mounted externally")
    ap.add_argument("--seg-weights-stage-dir", type=str, default="", help="Optional separate stage dir for v4 seg .pt files")
    ap.add_argument("--seg-weights-dataset-id", type=str, default="daydreamerthinks/solar-filament-v4-seg-weights")
    args = ap.parse_args()

    fuse_dir = Path(args.fuse_dir)
    stage_dir = Path(args.stage_dir)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    fuse_cfg = json.loads((fuse_dir / "ensemble_config.json").read_text())
    weights = fuse_cfg.get("weights", [1.0] * len(args.model_dirs))
    p0_weights = fuse_cfg.get("p0_weights", list(weights))
    assert len(weights) == len(args.model_dirs) and len(p0_weights) == len(args.model_dirs)

    seg_stage_dir = Path(args.seg_weights_stage_dir) if args.seg_weights_stage_dir else None
    if seg_stage_dir is not None:
        if seg_stage_dir.exists():
            shutil.rmtree(seg_stage_dir)
        seg_stage_dir.mkdir(parents=True, exist_ok=True)
        (seg_stage_dir / "dataset-metadata.json").write_text(json.dumps({
            "title": "Solar Filament v4 Seg Weights",
            "id": args.seg_weights_dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
        }, indent=2))

    v3_mounted_prefixes = ("unet_efficientnet_b4_", "unet_resnext50_32x4d_", "unetplusplus_densenet169_")

    arch_groups = []
    for mdir_str, w, w0 in zip(args.model_dirs, weights, p0_weights):
        mdir = Path(mdir_str)
        arch, backbone, is_v4 = infer_arch_and_backbone(mdir)
        tag = sanitize_tag(f"{arch}_{backbone}")
        ckpts = []
        for f in range(4):
            src = mdir / f"model_fold{f}.pt"
            dst_name = f"{tag}_fold{f}.pt"
            if not args.skip_seg_ckpts:
                if not src.exists():
                    raise FileNotFoundError(f"Missing fold checkpoint: {src}")
                link_or_copy(src, stage_dir / dst_name)
            if seg_stage_dir is not None and not dst_name.startswith(v3_mounted_prefixes):
                if not src.exists():
                    raise FileNotFoundError(f"Missing v4 fold checkpoint for seg stage: {src}")
                link_or_copy(src, seg_stage_dir / dst_name)
            ckpts.append(dst_name)
        fam_views = int(args.tta_views) if is_v4 else int(args.v3_tta_views)
        arch_groups.append({
            "arch": arch,
            "backbone": backbone,
            "weight": float(w),
            "p0_weight": float(w0),
            "ckpts": ckpts,
            "tta_views": fam_views,
        })

    yolo_ckpts = []
    yolo_imgsz = 1024
    yolo_use_3ch = False
    if args.yolo_dir:
        ydir = Path(args.yolo_dir)
        yrep_p = ydir / "yolo_val_report.json"
        if yrep_p.exists():
            yrep = json.loads(yrep_p.read_text())
            yolo_imgsz = int(yrep.get("imgsz", 1024))
            yolo_use_3ch = bool(yrep.get("use_3ch", False))
        for f in range(4):
            src = ydir / f"yolo_fold{f}.pt"
            dst_name = f"yolo_fold{f}.pt"
            if not args.skip_yolo_ckpts:
                if not src.exists():
                    raise FileNotFoundError(f"Missing YOLO fold checkpoint: {src}")
                link_or_copy(src, stage_dir / dst_name)
            yolo_ckpts.append(dst_name)

    selector_files = list(fuse_cfg.get("selector_files", []))
    if selector_files:
        for fn in selector_files:
            shutil.copy2(fuse_dir / fn, stage_dir / fn)
    seg_selector_files = list(fuse_cfg.get("seg_selector_files", []))
    if seg_selector_files:
        for fn in seg_selector_files:
            if (fuse_dir / fn).exists():
                shutil.copy2(fuse_dir / fn, stage_dir / fn)
    for f in range(4):
        s_main = fuse_dir / f"selector_fold{f}.txt"
        if s_main.exists():
            shutil.copy2(s_main, stage_dir / f"selector_fold{f}.txt")
            if not selector_files:
                selector_files.append(f"selector_fold{f}.txt")

    dual_roi_cfg = {"enabled": False}
    best_sources = list(fuse_cfg["best_sources"])
    best_cut = float(fuse_cfg["best_cut"])
    split_cost = float(fuse_cfg.get("split_cost", 0.24))
    yolo_gbdt_weight = float(fuse_cfg.get("yolo_gbdt_weight", 1.0))
    final_nested_pq = float(fuse_cfg.get("nested_oof_PQ", 0.0))

    bundle_cfg = {
        "arch_groups": arch_groups,
        "yolo_ckpts": yolo_ckpts,
        "yolo_imgsz": yolo_imgsz,
        "yolo_use_3ch": yolo_use_3ch,
        "dual_roi": dual_roi_cfg,
        "features": fuse_cfg["features"],
        "seg_features": fuse_cfg.get("seg_features", []),
        "selector_files": selector_files,
        "seg_selector_files": seg_selector_files,
        "yolo_gbdt_weight": yolo_gbdt_weight,
        "best_sources": best_sources,
        "best_cut": best_cut,
        "split_cost": split_cost,
        "tta_views": int(args.tta_views),
        "oof_PQ": fuse_cfg.get("oof_PQ"),
        "oof_SQ": fuse_cfg.get("oof_SQ"),
        "oof_RQ": fuse_cfg.get("oof_RQ"),
        "nested_oof_PQ": final_nested_pq,
        "val141_PQ": fuse_cfg.get("val141_PQ"),
    }
    cfg_json = json.dumps(bundle_cfg, indent=2)
    (stage_dir / "ensemble_config.json").write_text(cfg_json)
    (stage_dir / "ensemble_v4_config.json").write_text(cfg_json)

    ds_meta = {
        "title": "Solar Filament v4 Panoptic Models",
        "id": args.dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(ds_meta, indent=2))
    print(
        f"[+] Staged {len(arch_groups)} seg families ({sum(len(g['ckpts']) for g in arch_groups)} ckpts), "
        f"{len(yolo_ckpts)} YOLO ckpts (3ch={yolo_use_3ch}), cut={best_cut:.2f}, split_cost={split_cost:.2f}, "
        f"Nested PQ={final_nested_pq:.4f} -> {stage_dir}"
    )


if __name__ == "__main__":
    main()
