"""Generate all 4-fold Stage-1 segmentation (v4 4-ch + v3 3-ch) and 3-ch Dinkelbach YOLO training scripts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG_DIR = ROOT.parent / "configs" if (ROOT.parent / "configs" / "clean_4fold_split.json").exists() else Path("filament_comp_research")

FOLD_MAP = json.dumps(json.loads((CFG_DIR / "clean_4fold_split.json").read_text())["fold_of_stem"])
VAL141 = json.dumps(json.loads((CFG_DIR / "clean_val_split.json").read_text())["val"])

V4_TEMPLATE = (ROOT / "train_v4_panoptic_template.py").read_text()
V4_EXPERIMENTS = [
    ("solar-filament-v4-convnext-small", "Unet", "tu-convnext_small.fb_in22k_ft_in1k_384", 4, 2, 24),
    ("solar-filament-v4-hrnet-w32", "Unet", "tu-hrnet_w32.ms_in1k", 2, 4, 24),
    ("solar-filament-v4-regnety-064-fpn", "FPN", "timm-regnety_064", 3, 3, 24),
]

for slug, arch, backbone, bs, accum, epochs in V4_EXPERIMENTS:
    d = ROOT / slug
    d.mkdir(parents=True, exist_ok=True)
    code = (
        V4_TEMPLATE.replace("__ARCH__", arch)
        .replace("__BACKBONE__", backbone)
        .replace("__BATCH_SIZE__", str(bs))
        .replace("__GRAD_ACCUM__", str(accum))
        .replace("__EPOCHS__", str(epochs))
        .replace("__FOLD_MAP__", FOLD_MAP)
        .replace("__VAL141_STEMS__", VAL141)
    )
    (d / "train.py").write_text(code)
    print("wrote", d / "train.py")

v3_tpl_path = ROOT / "train_v3_panoptic_template.py"
if v3_tpl_path.exists():
    V3_TEMPLATE = v3_tpl_path.read_text()
    V3_EXPERIMENTS = [
        ("solar-filament-v3-effnet-b4", "Unet", "efficientnet-b4", 8, 28),
        ("solar-filament-v3-densenet169-upp", "UnetPlusPlus", "densenet169", 4, 22),
        ("solar-filament-v3-resnext50", "Unet", "resnext50_32x4d", 8, 28),
    ]
    for slug, arch, backbone, bs, epochs in V3_EXPERIMENTS:
        d = ROOT / slug
        d.mkdir(parents=True, exist_ok=True)
        code = (
            V3_TEMPLATE.replace("__ARCH__", arch)
            .replace("__BACKBONE__", backbone)
            .replace("__BATCH_SIZE__", str(bs))
            .replace("__EPOCHS__", str(epochs))
            .replace("__FOLD_MAP__", FOLD_MAP)
            .replace("__VAL141_STEMS__", VAL141)
        )
        (d / "train.py").write_text(code)
        print("wrote", d / "train.py")

yolo_slug = "solar-filament-v4-yolo11m-1024"
yolo_dir = ROOT / yolo_slug
yolo_dir.mkdir(parents=True, exist_ok=True)
yolo_code = (ROOT / "train_v4_dinkelbach_yolo.py").read_text().replace("__FOLD_MAP__", FOLD_MAP)
(yolo_dir / "train.py").write_text(yolo_code)
print("wrote", yolo_dir / "train.py")
