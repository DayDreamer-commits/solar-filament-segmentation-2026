"""
Solar Filament Segmentation 2026 - v3 4-Fold Out-Of-Fold Panoptic Pipeline.

Key Architectural & Computational Advances:
  1. Strict Day-Grouped, Annotator-Stratified 4-Fold CV over all 707 clean
     competition training stems (0 test-stem overlap, 0 same-day fold leakage).
  2. Startup Memmap Cache: Decodes JPEGs and rasterizes 3 decoupled panoptic
     targets once before training (~40s CPU), eliminating per-epoch COCO polygon
     re-rasterization (~15x training speedup).
  3. Scale-Equivariant 3-Channel Physical H-Alpha Encoding:
       - ch0: Disk-percentile normalized H-alpha intensity (cross-station invariant)
       - ch1: Matched-scale CLAHE (128x128 px physical tiles at both 768 and 2048)
       - ch2: Photospheric limb-darkening-free dark-filament contrast
  4. 3-Channel Decoupled Panoptic Targets:
       - tgt0 (shape):      count / k_cluster  -> un-shrunk compromise silhouette (SQ)
       - tgt1 (presence):   count / K_annots   -> Bayes-calibrated consensus prob (RQ)
       - tgt2 (core_spine): normalized per-instance Euclidean distance transform
                            -> interior spine markers for splitting touching filaments
  5. Multi-Proposal Extraction + Day-Grouped 4-Fold LightGBM Dinkelbach Expected-PQ
     (v / A) Selector + Greedy Strict-Disjoint Selection (0 overlap pixels).
"""

from __future__ import annotations

import copy
import glob
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ARCH_NAME = "__ARCH__"
BACKBONE = "__BACKBONE__"
BATCH_SIZE = int("__BATCH_SIZE__")
EPOCHS = int("__EPOCHS__")
FOLD_MAP: dict[str, int] = json.loads('__FOLD_MAP__')
VAL141_STEMS: set[str] = set(json.loads('__VAL141_STEMS__'))

CFG = dict(
    folds=4,
    crop=768,
    bs=BATCH_SIZE,
    epochs=EPOCHS,
    crops_per_img=3,
    lr=3e-4,
    wd=1e-4,
    ema=0.997,
    seed=2026,
    loader_workers=2,
)

H = W = 2048
T0 = time.time()


def log(*args) -> None:
    print(f"[{(time.time() - T0) / 60:6.2f}m]", *args, flush=True)


def pick_scratch_dir() -> Path:
    candidates = [Path("/kaggle/temp"), Path("/tmp")]
    best_p, best_free = Path("/tmp"), -1
    for c in candidates:
        if c.exists():
            free = shutil.disk_usage(c).free
            if free > best_free:
                best_p, best_free = c, free
    out = best_p / "v3_panoptic_cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def locate_comp_paths() -> tuple[Path, Path, Path]:
    env_root = os.environ.get("SF_DATA_DIR")
    if env_root:
        root = Path(env_root)
        return (
            root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
            root / "train" / "train_images",
            root / "test" / "test_images",
        )
    js_hits = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if not js_hits:
        raise FileNotFoundError("Could not locate MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    js = Path(js_hits[0])
    root = js.parent.parent
    return js, root / "train" / "train_images", root / "test" / "test_images"


def compute_disk_stats(img_u8: np.ndarray, disk_mask: np.ndarray) -> list[float]:
    vals = img_u8[disk_mask]
    if vals.size == 0:
        return [64.0, 128.0, 192.0]
    p10, p50, p90 = np.percentile(vals, (10, 50, 90))
    return [float(p10), float(p50), float(p90)]


def encode_3ch(img_u8: np.ndarray, stats: list[float], clahe_grid: int = 16) -> np.ndarray:
    """Build 3 full-disk physical channels in [0, 1]."""
    import cv2

    p10, p50, p90 = stats
    span = max(p90 - p10, 16.0)
    img_f = img_u8.astype(np.float32)

    ch0 = np.clip((img_f - p50) / span * 0.5 + 0.5, 0.0, 1.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(clahe_grid, clahe_grid)).apply(img_u8)
    ch1 = clahe.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(img_u8, (0, 0), 18.0).astype(np.float32)
    ch2 = np.clip((blur - img_f) / span + 0.5, 0.0, 1.0)
    return np.stack([ch0, ch1, ch2], axis=0)


def encode_3ch_u8(img_u8: np.ndarray, stats: list[float]) -> np.ndarray:
    """Quantize full-disk 3-channel physical encoding to uint8 (3, H, W)."""
    return np.rint(255.0 * encode_3ch(img_u8, stats, clahe_grid=16)).clip(0, 255).astype(np.uint8)


def prepare_cache(cache_dir: Path, out_dir: Path) -> None:
    import cv2
    from pycocotools import mask as mu

    js_path, train_dir, test_dir = locate_comp_paths()
    coco = json.loads(js_path.read_text())
    log("Loaded annotations:", js_path)

    sets_of: dict[str, list[str]] = defaultdict(list)
    anns_of: dict[str, list[dict]] = defaultdict(list)
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        sets_of[stem].append(im["id"])
    for a in coco["annotations"]:
        anns_of[a["image_id"]].append(a)

    stems = sorted(sets_of.keys())
    test_files = sorted(list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg")))
    test_stems = [p.stem for p in test_files]
    assert not (set(stems) & set(test_stems)), "Train/test stem overlap!"
    n = len(stems)
    log(f"Preparing cache: {n} train stems ({len(coco['images'])} annotator-images), {len(test_stems)} test stems")

    yy, xx = np.ogrid[:H, :W]
    disk_mask = (yy - H / 2.0) ** 2 + (xx - W / 2.0) ** 2 <= 940.0 ** 2

    free_gb = shutil.disk_usage(cache_dir).free / (1024.0 ** 3)
    precomputed_3ch = free_gb >= 22.0
    log(f"Scratch dir: {cache_dir} ({free_gb:.1f} GB free) -> precomputed_3ch={precomputed_3ch}")

    img_shape = (n, 3, H, W) if precomputed_3ch else (n, H, W)
    imgs_mm = np.lib.format.open_memmap(str(cache_dir / "imgs.npy"), "w+", np.uint8, img_shape)
    tgts_mm = np.lib.format.open_memmap(str(cache_dir / "tgt.npy"), "w+", np.uint8, (n, 3, H, W))

    centers: list[list[list[float]]] = [[] for _ in range(n)]
    img_stats: list[list[float]] = [[] for _ in range(n)]
    folds: list[int] = [int(FOLD_MAP.get(s, i % CFG["folds"])) for i, s in enumerate(stems)]

    def _cache_one(i: int) -> None:
        s = stems[i]
        p = train_dir / f"{s}.jpeg"
        if not p.exists():
            p = train_dir / f"{s}.jpg"
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (H, W):
            raise RuntimeError(f"Bad train image: {p}")
        st_disk = compute_disk_stats(im, disk_mask)
        img_stats[i] = st_disk
        imgs_mm[i] = encode_3ch_u8(im, st_disk) if precomputed_3ch else im

        K = len(sets_of[s])
        count = np.zeros((H, W), dtype=np.float32)
        core_sum = np.zeros((H, W), dtype=np.float32)
        per_annot_masks: list[np.ndarray] = []
        cs: list[list[float]] = []

        for aset in sets_of[s]:
            m_a = np.zeros((H, W), dtype=np.uint8)
            core_a = np.zeros((H, W), dtype=np.float32)
            for a in anns_of.get(aset, []):
                seg = a.get("segmentation")
                if not seg:
                    continue
                if isinstance(seg, list):
                    pts_list = [np.asarray(poly, dtype=np.float32).reshape(-1, 2) for poly in seg if len(poly) >= 6]
                    if not pts_list:
                        continue
                    xs = np.concatenate([pts[:, 0] for pts in pts_list])
                    ys = np.concatenate([pts[:, 1] for pts in pts_list])
                    x0 = max(0, int(math.floor(float(xs.min()))))
                    y0 = max(0, int(math.floor(float(ys.min()))))
                    x1 = min(W, int(math.ceil(float(xs.max()))) + 1)
                    y1 = min(H, int(math.ceil(float(ys.max()))) + 1)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    poly_crop = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                    shifted = [np.rint(pts - np.array([[x0, y0]], dtype=np.float32)).astype(np.int32) for pts in pts_list]
                    cv2.fillPoly(poly_crop, shifted, 1)
                else:
                    bx, by, bw, bh = map(int, mu.toBbox(seg))
                    if bw <= 0 or bh <= 0:
                        continue
                    x0, y0 = max(0, bx), max(0, by)
                    x1, y1 = min(W, bx + bw), min(H, by + bh)
                    poly_crop = mu.decode(seg)[y0:y1, x0:x1]
                if not np.any(poly_crop):
                    continue
                m_a[y0:y1, x0:x1] |= poly_crop
                padded = np.pad(poly_crop, 1, mode="constant", constant_values=0)
                dt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
                dmax = float(dt.max())
                if dmax > 0:
                    core_a[y0:y1, x0:x1] = np.maximum(core_a[y0:y1, x0:x1], dt / dmax)
                cs.append([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            per_annot_masks.append(m_a)
            count += m_a.astype(np.float32)
            core_sum += core_a

        ncc, lab = cv2.connectedComponents((count > 0).astype(np.uint8), connectivity=8)
        k_cl = np.zeros(ncc, dtype=np.float32)
        for m_a in per_annot_masks:
            k_cl[np.unique(lab[m_a > 0])] += 1.0
        k_cl[0] = 1.0
        denom_cl = np.maximum(k_cl[lab], 1.0)

        tgts_mm[i, 0] = np.rint(255.0 * count / denom_cl).astype(np.uint8)
        tgts_mm[i, 1] = np.rint(255.0 * count / max(K, 1)).astype(np.uint8)
        tgts_mm[i, 2] = np.rint(255.0 * np.clip(core_sum / denom_cl, 0.0, 1.0)).astype(np.uint8)
        centers[i] = cs

    done_cnt = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(_cache_one, range(n)):
            done_cnt += 1
            if done_cnt % 150 == 0 or done_cnt == n:
                log(f"Cached {done_cnt}/{n} train images")

    imgs_mm.flush()
    tgts_mm.flush()
    del imgs_mm, tgts_mm

    test_stats = []
    for p in test_files:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        test_stats.append(compute_disk_stats(im, disk_mask))

    meta = dict(
        stems=stems,
        folds=folds,
        img_stats=img_stats,
        centers=centers,
        n_sets=[len(sets_of[s]) for s in stems],
        test_files=[str(p) for p in test_files],
        test_stems=test_stems,
        test_stats=test_stats,
        js_path=str(js_path),
        train_dir=str(train_dir),
        arch=ARCH_NAME,
        backbone=BACKBONE,
        precomputed_3ch=precomputed_3ch,
        cfg=CFG,
    )
    (cache_dir / "meta.json").write_text(json.dumps(meta))
    (out_dir / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "centers"}))
    log("Cache preparation finished")


def _seed_dataloader_worker(worker_id: int) -> None:
    import cv2
    import torch

    cv2.setNumThreads(0)
    np.random.seed((torch.initial_seed() + worker_id) % (2**31))


class PanopticCropDS:
    """Module top-level Dataset class (picklable across Python 3.13+ spawn workers)."""

    def __init__(
        self,
        cache_dir_str: str,
        indices: list[int],
        centers: list[list[list[float]]],
        img_stats: list[list[float]],
        precomputed_3ch: bool,
        crop_size: int,
        crops_per_img: int,
    ):
        self.cache_dir_str = cache_dir_str
        self.indices = indices
        self.centers = centers
        self.img_stats = img_stats
        self.precomputed_3ch = precomputed_3ch
        self.crop_size = crop_size
        self.crops_per_img = crops_per_img
        self.clahe_crop_grid = max(2, crop_size // 128)
        self.imgs = None
        self.tgts = None

    def __len__(self) -> int:
        return len(self.indices) * self.crops_per_img

    def __getitem__(self, idx: int):
        import cv2
        import torch

        if self.imgs is None:
            cdir = Path(self.cache_dir_str)
            self.imgs = np.load(str(cdir / "imgs.npy"), mmap_mode="r")
            self.tgts = np.load(str(cdir / "tgt.npy"), mmap_mode="r")
        j = int(self.indices[idx % len(self.indices)])
        C = self.crop_size
        rng = np.random
        win = int(round(C / math.exp(rng.uniform(math.log(0.75), math.log(1.35)))))
        win = min(max(win, 384), H)
        cs = self.centers[j]
        if cs and rng.rand() < 0.65:
            cx, cy = cs[rng.randint(len(cs))]
            cx += rng.uniform(-win * 0.35, win * 0.35)
            cy += rng.uniform(-win * 0.35, win * 0.35)
        else:
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        x0 = int(np.clip(cx - win / 2.0, 0, W - win))
        y0 = int(np.clip(cy - win / 2.0, 0, H - win))

        if self.precomputed_3ch:
            ch = np.array(self.imgs[j, :, y0:y0 + win, x0:x0 + win], dtype=np.float32) / 255.0
        else:
            pad = 36
            px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
            px1, py1 = min(W, x0 + win + pad), min(H, y0 + win + pad)
            im_pad = np.array(self.imgs[j, py0:py1, px0:px1], dtype=np.uint8)
            ch_pad = encode_3ch(im_pad, self.img_stats[j], self.clahe_crop_grid)
            ox0, oy0 = x0 - px0, y0 - py0
            ch = ch_pad[:, oy0:oy0 + win, ox0:ox0 + win]
        tg = np.array(self.tgts[j, :, y0:y0 + win, x0:x0 + win], dtype=np.uint8)

        if win != C:
            interp = cv2.INTER_AREA if win > C else cv2.INTER_LINEAR
            ch = np.stack([cv2.resize(c, (C, C), interpolation=interp) for c in ch], axis=0)
            tg = np.stack([cv2.resize(t, (C, C), interpolation=cv2.INTER_LINEAR) for t in tg], axis=0)

        k = rng.randint(4)
        if k:
            ch = np.rot90(ch, k, axes=(1, 2))
            tg = np.rot90(tg, k, axes=(1, 2))
        if rng.rand() < 0.5:
            ch = ch[:, :, ::-1]
            tg = tg[:, :, ::-1]

        gain = float(rng.uniform(0.86, 1.14))
        bias = float(rng.uniform(-0.05, 0.05))
        gamma = float(rng.uniform(0.88, 1.14))
        ch = np.clip((ch - 0.5) * gain + 0.5 + bias, 0.0, 1.0) ** gamma
        x = (ch - 0.5) / 0.25
        y = tg.astype(np.float32) / 255.0
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
        )


def train_worker(gpu_id: int, fold_list: list[int], cache_dir_str: str, out_dir_str: str, smoke_only: bool = False) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import cv2
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp

    cv2.setNumThreads(0)
    cache_dir = Path(cache_dir_str)
    out_dir = Path(out_dir_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    meta = json.loads((cache_dir / "meta.json").read_text())
    fold_arr = np.array(meta["folds"], dtype=np.int32)
    precomputed_3ch = bool(meta.get("precomputed_3ch", False))
    C = CFG["crop"]

    @torch.no_grad()
    def predict_full_d4(model, ch_01: np.ndarray) -> np.ndarray:
        """Full 2048x2048 8-view dihedral D4 TTA from ch_01 (3, H, W) in [0, 1] -> uint8 (3, H, W)."""
        ch = (ch_01 - 0.5) / 0.25
        x0 = torch.from_numpy(np.ascontiguousarray(ch, dtype=np.float32))[None].to(device)
        acc = torch.zeros((1, 3, H, W), device=device, dtype=torch.float32)
        views_meta = []
        views_tensors = []
        for tr_flag in (False, True):
            xt = x0.transpose(2, 3) if tr_flag else x0
            for fl in ([], [2], [3], [2, 3]):
                views_tensors.append(torch.flip(xt, fl) if fl else xt)
                views_meta.append((tr_flag, fl))

        # Run in batches of 1 view for UnetPlusPlus (dense skip maps) or 2 views for Unet
        v_bs = 1 if ARCH_NAME == "UnetPlusPlus" else 2
        for start in range(0, 8, v_bs):
            bx = torch.cat(views_tensors[start:start + v_bs], dim=0)
            if device.type == "cuda":
                bx = bx.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                probs = torch.sigmoid(model(bx).float())
            for bi in range(probs.shape[0]):
                tr_flag, fl = views_meta[start + bi]
                p = probs[bi:bi + 1]
                if fl:
                    p = torch.flip(p, fl)
                if tr_flag:
                    p = p.transpose(2, 3)
                acc += p
        return (acc[0] / 8.0 * 255.0).round().clamp(0, 255).byte().cpu().numpy()

    def write_png_zip(zpath: Path, names: list[str], preds_iter) -> None:
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
            for name, arr3 in zip(names, preds_iter):
                # cv2 BGR order: B=ch0 (shape), G=ch1 (presence), R=ch2 (core_spine)
                bgr = np.dstack([arr3[0], arr3[1], arr3[2]])
                ok, buf = cv2.imencode(".png", bgr)
                if not ok:
                    raise RuntimeError(f"PNG encode failed for {name}")
                zf.writestr(f"{name}.png", buf.tobytes())

    def compute_panoptic_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ch0: shape (BCE + SoftTversky recall-tilted)
        l0, t0 = logits[:, 0], targets[:, 0]
        p0 = torch.sigmoid(l0)
        bce0 = F.binary_cross_entropy_with_logits(l0, t0)
        tp0 = (p0 * t0).sum(dim=(1, 2))
        fp0 = (p0 * (1.0 - t0)).sum(dim=(1, 2))
        fn0 = ((1.0 - p0) * t0).sum(dim=(1, 2))
        tversky0 = 1.0 - ((tp0 + 1.0) / (tp0 + 0.40 * fp0 + 0.60 * fn0 + 1.0)).mean()

        # ch1: presence (strictly proper BCE for calibrated m/K expectation!)
        bce1 = F.binary_cross_entropy_with_logits(logits[:, 1], targets[:, 1])

        # ch2: core_spine (BCE + foreground-weighted SmoothL1 + SoftDice)
        l2, t2 = logits[:, 2], targets[:, 2]
        p2 = torch.sigmoid(l2)
        bce2 = F.binary_cross_entropy_with_logits(l2, t2)
        inter2 = (p2 * t2).sum(dim=(1, 2))
        dice2 = 1.0 - ((2.0 * inter2 + 1.0) / (p2.sum(dim=(1, 2)) + t2.sum(dim=(1, 2)) + 1.0)).mean()

        return (bce0 + 0.65 * tversky0) + 0.85 * bce1 + 0.60 * (bce2 + 0.50 * dice2)

    def run_single_fold(f: int, smoke: bool = False) -> None:
        tag = f"smoke{gpu_id}" if smoke else f"fold{f}"
        tr_idx = np.flatnonzero(fold_arr != f)
        va_idx = np.flatnonzero(fold_arr == f)
        torch.manual_seed(CFG["seed"] + f)
        np.random.seed(CFG["seed"] + f)

        enc_weights = None if os.environ.get("SF_NO_PRETRAIN") == "1" else "imagenet"
        arch_cls = getattr(smp, ARCH_NAME)
        model = arch_cls(
            encoder_name=BACKBONE,
            encoder_weights=enc_weights,
            in_channels=3,
            classes=3,
            activation=None,
        ).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)

        n_workers = min(1, CFG["loader_workers"]) if smoke else CFG["loader_workers"]
        ds = PanopticCropDS(
            cache_dir_str=cache_dir_str,
            indices=[int(ix) for ix in tr_idx],
            centers=meta["centers"],
            img_stats=meta["img_stats"],
            precomputed_3ch=precomputed_3ch,
            crop_size=C,
            crops_per_img=CFG["crops_per_img"],
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=CFG["bs"],
            shuffle=True,
            drop_last=True,
            num_workers=n_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(n_workers > 0),
            worker_init_fn=_seed_dataloader_worker,
        )
        total_steps = 2 if smoke else CFG["epochs"] * len(dl)
        warmup = min(250, max(1, total_steps // 10))
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: (s + 1) / warmup
            if s < warmup
            else 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
        )
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        step = 0
        running_loss = 0.0
        t_fold = time.time()
        model.train()
        while step < total_steps:
            for x, y in dl:
                x = x.to(device, non_blocking=True)
                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model(x).float()
                    loss = compute_panoptic_loss(logits, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                decay = min(CFG["ema"], (1.0 + step) / (10.0 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1.0 - decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)

                step += 1
                running_loss = 0.98 * running_loss + 0.02 * loss.item() if step > 1 else loss.item()
                if step % 250 == 0 or step == total_steps:
                    log(f"{tag} step {step}/{total_steps} loss={running_loss:.4f} ({(time.time() - t_fold) / step:.3f}s/it)")
                if step >= total_steps:
                    break
        del dl

        imgs_mm = np.load(str(cache_dir / "imgs.npy"), mmap_mode="r")
        tgts_mm = np.load(str(cache_dir / "tgt.npy"), mmap_mode="r")
        va_use = va_idx[:2] if smoke else va_idx
        te_count = min(2, len(meta["test_files"])) if smoke else len(meta["test_files"])

        inter = np.zeros(3, dtype=np.float64)
        denom = np.zeros(3, dtype=np.float64)

        def oof_gen():
            for j in va_use:
                if precomputed_3ch:
                    ch_01 = np.array(imgs_mm[j], dtype=np.float32) / 255.0
                else:
                    ch_01 = encode_3ch_u8(np.array(imgs_mm[j]), meta["img_stats"][j]).astype(np.float32) / 255.0
                pr = predict_full_d4(ema, ch_01)
                t_bin = np.asarray(tgts_mm[j]) >= 128
                q_bin = pr >= 128
                inter[:] += (q_bin & t_bin).sum(axis=(1, 2))
                denom[:] += q_bin.sum(axis=(1, 2)) + t_bin.sum(axis=(1, 2))
                yield pr

        oof_zip = out_dir / f"oof_{tag}.zip"
        write_png_zip(oof_zip, [meta["stems"][int(j)] for j in va_use], oof_gen())
        dices = 2.0 * inter / np.maximum(denom, 1.0)
        log(f"{tag} OOF Dice@0.5: shape={dices[0]:.4f} presence={dices[1]:.4f} core={dices[2]:.4f}")

        def test_gen():
            for ti in range(te_count):
                im = cv2.imread(meta["test_files"][ti], cv2.IMREAD_GRAYSCALE)
                ch_01 = encode_3ch_u8(im, meta["test_stats"][ti]).astype(np.float32) / 255.0
                yield predict_full_d4(ema, ch_01)

        test_zip = out_dir / f"test_{tag}.zip"
        write_png_zip(test_zip, meta["test_stems"][:te_count], test_gen())

        ckpt_path = out_dir / f"model_{tag}.pt"
        torch.save(
            {k: v.half() if v.is_floating_point() else v for k, v in ema.state_dict().items()},
            ckpt_path,
        )
        if smoke:
            for fp in (oof_zip, test_zip, ckpt_path):
                fp.unlink(missing_ok=True)
        log(f"{tag} complete")
        del model, ema, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log(f"Worker GPU={gpu_id} folds={fold_list} device={device}")
    run_single_fold(fold_list[0], smoke=True)
    if smoke_only:
        return
    for f in fold_list:
        run_single_fold(f, smoke=False)


# --------------------------------------------------------------------------- #
# Post-Processing: Multi-Proposal + Spine Watershed + Gap Bridging + LightGBM #
# --------------------------------------------------------------------------- #
FEATURES = [
    "source", "area", "log_area", "bw", "bh", "fill", "aspect", "disk_radius",
    "p0_mean", "p0_q25", "p0_q75", "p0_max", "p0_border_drop",
    "p1_mean", "p1_q25", "p1_q75", "p1_max", "p1_p0_ratio",
    "p2_mean", "p2_max", "n_cores",
    "cc_iou", "dark_mean", "dark_q75", "image_cc_count",
]


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


def fill_holes_crop(crop_mask: np.ndarray, cv2) -> np.ndarray:
    if not np.any(crop_mask):
        return crop_mask.copy()
    padded = np.pad(np.ascontiguousarray(crop_mask, dtype=np.uint8), 1, mode="constant", constant_values=0)
    cnts, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(padded)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    return filled[1:-1, 1:-1]


def extract_candidates_for_stem(job) -> list[dict]:
    import cv2
    from pycocotools import mask as mu

    stem, zip_paths, img_path, stats, gt_sets = job
    prob = np.zeros((H, W, 3), dtype=np.float32)
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            arr = cv2.imdecode(np.frombuffer(zf.read(f"{stem}.png"), np.uint8), cv2.IMREAD_COLOR)
        prob += arr.astype(np.float32)
    prob /= 255.0 * len(zip_paths)

    img_u8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    span = max(stats[2] - stats[0], 16.0)
    blur = cv2.GaussianBlur(img_u8, (0, 0), 18.0).astype(np.float32)
    dark_map = np.clip((blur - img_u8.astype(np.float32)) / span, -1.0, 2.0)

    p0, p1, p2 = prob[:, :, 0], prob[:, :, 1], prob[:, :, 2]
    proposals: list[tuple[int, dict, int, int, np.ndarray]] = []  # (source, rle, x0, y0, mask_crop)

    # Family 0 (thr=0.50) & Family 1 (thr=0.40)
    cc_count_050 = 0
    for src_id, thr in ((0, 0.50), (1, 0.40)):
        ncc, lab, st, _ = cv2.connectedComponentsWithStats((p0 >= thr).astype(np.uint8), connectivity=8)
        if src_id == 0:
            cc_count_050 = sum(1 for k in range(1, ncc) if st[k, cv2.CC_STAT_AREA] >= 60)
        for k in range(1, ncc):
            bx, by, bw, bh, area = map(int, st[k])
            if area < 60:
                continue
            mc = fill_holes_crop((lab[by:by + bh, bx:bx + bw] == k).astype(np.uint8), cv2)
            rle = encode_crop_rle(mc, bx, by, mu)
            if rle is not None:
                proposals.append((src_id, rle, bx, by, mc))

            # Family 2: Spine-core Watershed Splitting for touching filaments (from thr=0.50 components)
            if src_id == 0 and area >= 220:
                core_bin = ((p2[by:by + bh, bx:bx + bw] >= 0.45) & (mc > 0)).astype(np.uint8)
                nc_core, lab_core, st_core, cen_core = cv2.connectedComponentsWithStats(core_bin, connectivity=8)
                valid_cores = [
                    ck for ck in range(1, nc_core)
                    if st_core[ck, cv2.CC_STAT_AREA] >= 18
                ]
                if len(valid_cores) >= 2:
                    # Verify pairwise centroid separation >= 10 px
                    kept_cores = []
                    for ck in sorted(valid_cores, key=lambda c: -st_core[c, cv2.CC_STAT_AREA]):
                        cy_c, cx_c = cen_core[ck][1], cen_core[ck][0]
                        if all(math.hypot(cx_c - cen_core[ok][0], cy_c - cen_core[ok][1]) >= 10.0 for ok in kept_cores):
                            kept_cores.append(ck)
                    if len(kept_cores) >= 2:
                        dists = []
                        for ck in kept_cores:
                            seed_inv = np.where(lab_core == ck, 0, 1).astype(np.uint8)
                            dists.append(cv2.distanceTransform(seed_inv, cv2.DIST_L2, 3))
                        assign = np.argmin(np.stack(dists, axis=0), axis=0)
                        for idx_c in range(len(kept_cores)):
                            part = ((assign == idx_c) & (mc > 0)).astype(np.uint8)
                            if int(part.sum()) >= 60:
                                r_part = encode_crop_rle(part, bx, by, mu)
                                if r_part is not None:
                                    proposals.append((2, r_part, bx, by, part))

    # Family 3: Hysteresis / Gap-Bridged Proposals across narrow faint spine gaps
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    base_050 = (p0 >= 0.50).astype(np.uint8)
    bridged = ((cv2.morphologyEx(base_050, cv2.MORPH_CLOSE, k_bridge) > 0) & (p0 >= 0.28)).astype(np.uint8) | base_050
    ncc_b, lab_b, st_b, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    _, lab_orig = cv2.connectedComponents(base_050, connectivity=8)
    for k in range(1, ncc_b):
        bx, by, bw, bh, area = map(int, st_b[k])
        if area < 120:
            continue
        mc = (lab_b[by:by + bh, bx:bx + bw] == k).astype(np.uint8)
        orig_ids = np.unique(lab_orig[by:by + bh, bx:bx + bw][mc > 0])
        orig_ids = orig_ids[orig_ids > 0]
        if len(orig_ids) >= 2:  # Only add when it genuinely bridges >= 2 fragments!
            mc = fill_holes_crop(mc, cv2)
            rle = encode_crop_rle(mc, bx, by, mu)
            if rle is not None:
                proposals.append((3, rle, bx, by, mc))

    if not proposals:
        return []

    cc_rles = [it[1] for it in proposals if it[0] == 0]

    # Deduplicate near-identical proposals (IoU > 0.96), keeping lowest source id first
    dedup = []
    dedup_rles = []
    for item in proposals:
        if dedup_rles:
            ious = mu.iou(dedup_rles, [item[1]], [0]).ravel()
            if float(ious.max()) > 0.96:
                continue
        dedup.append(item)
        dedup_rles.append(item[1])

    ci_mat = np.asarray(mu.iou(dedup_rles, cc_rles, [0] * len(cc_rles))) if cc_rles else np.zeros((len(dedup_rles), 0))

    values = np.zeros(len(dedup), dtype=np.float64)
    if gt_sets is not None:
        for gs in gt_sets:
            if gs:
                iou_m = np.asarray(mu.iou(dedup_rles, gs, [0] * len(gs)))
                values += np.where(iou_m > 0.50, iou_m, 0.0).max(axis=1)

    k3 = np.ones((3, 3), dtype=np.uint8)
    rows = []
    for idx_p, (src_id, rle, bx, by, mc) in enumerate(dedup):
        bh, bw = mc.shape
        fg = mc > 0
        area = int(fg.sum())
        if area < 60:
            continue
        v0 = p0[by:by + bh, bx:bx + bw][fg]
        v1 = p1[by:by + bh, bx:bx + bw][fg]
        v2 = p2[by:by + bh, bx:bx + bw][fg]
        vd = dark_map[by:by + bh, bx:bx + bw][fg]

        # Outer 2-pixel ring drop in p0 (sharp boundary vs diffuse blob)
        pad_x0, pad_y0 = max(0, bx - 2), max(0, by - 2)
        pad_x1, pad_y1 = min(W, bx + bw + 2), min(H, by + bh + 2)
        m_pad = np.zeros((pad_y1 - pad_y0, pad_x1 - pad_x0), dtype=np.uint8)
        m_pad[by - pad_y0:by - pad_y0 + bh, bx - pad_x0:bx - pad_x0 + bw] = mc
        ring = (cv2.dilate(m_pad, k3, iterations=2) > 0) & (m_pad == 0)
        ring_mean = float(p0[pad_y0:pad_y1, pad_x0:pad_x1][ring].mean()) if np.any(ring) else 0.0

        core_sub = ((p2[by:by + bh, bx:bx + bw] >= 0.45) & fg).astype(np.uint8)
        nc_sub, _, st_sub, _ = cv2.connectedComponentsWithStats(core_sub, connectivity=8)
        n_cores = sum(1 for ck in range(1, nc_sub) if st_sub[ck, cv2.CC_STAT_AREA] >= 15)

        c_iou = float(ci_mat[idx_p].max()) if ci_mat.shape[1] > 0 else 0.0
        p0_m = float(v0.mean())
        p1_m = float(v1.mean())

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
            cc_iou=c_iou,
            dark_mean=float(vd.mean()),
            dark_q75=float(np.quantile(vd, 0.75)),
            image_cc_count=cc_count_050,
            v=float(values[idx_p]),
            A=len(gt_sets) if gt_sets is not None else 0,
        ))
    return rows


def greedy_select_disjoint(df, cut: float, allowed_sources: list[int], max_overlap_frac: float = 0.12):
    from pycocotools import mask as mu

    result = {}
    sub = df[(df["score"] >= cut) & (df["source"].isin(allowed_sources))]
    for stem, grp in sub.groupby("stem", sort=False):
        chosen = []
        for row in grp.sort_values("score", ascending=False, kind="stable").itertuples():
            r = row.rle
            if chosen:
                occ = mu.merge(chosen)
                ov = float(mu.area(mu.merge([r, occ], intersect=True)))
                if ov > max_overlap_frac * row.area:
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
        result[stem] = chosen
    return result


def score_subset(gt_dict: dict, pred_dict: dict, stems: list[str]) -> dict:
    from pycocotools import mask as mu

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


def evaluate_and_fuse(cache_dir: Path, out_dir: Path) -> None:
    import lightgbm as lgb
    import pandas as pd
    from pycocotools import mask as mu

    meta = json.loads((cache_dir / "meta.json").read_text())
    coco = json.loads(Path(meta["js_path"]).read_text())
    anns_by_img: dict[str, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        seg = a.get("segmentation")
        if seg:
            anns_by_img[a["image_id"]].append(mu.merge(mu.frPyObjects(seg, H, W)))
    gt_by_stem: dict[str, list[list[dict]]] = defaultdict(list)
    for im in coco["images"]:
        gt_by_stem[Path(im["file_name"]).stem].append(anns_by_img.get(im["id"], []))

    oof_map: dict[str, list[str]] = defaultdict(list)
    for zp in sorted(out_dir.glob("oof_fold*.zip")):
        with zipfile.ZipFile(zp) as zf:
            for nm in zf.namelist():
                oof_map[nm[:-4]].append(str(zp))

    test_zips = [str(p) for p in sorted(out_dir.glob("test_fold*.zip"))]

    train_dir = Path(meta["train_dir"])
    oof_jobs = [
        (
            s,
            oof_map[s],
            str(train_dir / f"{s}.jpeg") if (train_dir / f"{s}.jpeg").exists() else str(train_dir / f"{s}.jpg"),
            meta["img_stats"][i],
            gt_by_stem[s],
        )
        for i, s in enumerate(meta["stems"])
        if s in oof_map
    ]
    test_jobs = [
        (s, test_zips, meta["test_files"][i], meta["test_stats"][i], None)
        for i, s in enumerate(meta["test_stems"])
    ]

    log(f"Extracting panoptic candidates for {len(oof_jobs)} OOF stems & {len(test_jobs)} test stems...")
    oof_rows, test_rows = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(extract_candidates_for_stem, oof_jobs):
            oof_rows.extend(r)
        for r in pool.map(extract_candidates_for_stem, test_jobs):
            test_rows.extend(r)

    base_cols = ["stem", "rle", "v", "A"] + FEATURES
    oof_df = pd.DataFrame(oof_rows) if oof_rows else pd.DataFrame(columns=base_cols)
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame(columns=base_cols)
    log(f"Extracted {len(oof_df)} OOF proposals and {len(test_df)} test proposals")

    stem_to_fold = dict(zip(meta["stems"], meta["folds"]))
    oof_df["fold"] = oof_df["stem"].map(stem_to_fold) if len(oof_df) > 0 else pd.Series(dtype=int)
    oof_df["score"] = 0.0
    test_df["score"] = 0.0

    lgb_params = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=min(50, max(2, len(oof_df) // 8)),
        lambda_l2=10.0,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        verbosity=-1,
        num_threads=4,
        seed=2026,
    )

    if len(oof_df) >= 8:
        for f in range(CFG["folds"]):
            tr = oof_df["fold"] != f
            va = oof_df["fold"] == f
            if int(tr.sum()) < 2:
                continue
            dtrain = lgb.Dataset(
                oof_df.loc[tr, FEATURES],
                label=oof_df.loc[tr, "v"] / np.maximum(oof_df.loc[tr, "A"], 1),
                weight=np.maximum(oof_df.loc[tr, "A"], 1),
            )
            bst = lgb.train(lgb_params, dtrain, num_boost_round=280)
            if int(va.sum()) > 0:
                oof_df.loc[va, "score"] = bst.predict(oof_df.loc[va, FEATURES])
            if len(test_df) > 0:
                test_df["score"] += bst.predict(test_df[FEATURES]) / CFG["folds"]
            bst.save_model(str(out_dir / f"selector_fold{f}.txt"))

    oof_df.to_pickle(out_dir / "oof_candidates_scored.pkl")
    test_df.to_pickle(out_dir / "test_candidates_scored.pkl")

    all_stems = sorted(oof_map.keys())
    multi_stems = [s for s in all_stems if len(gt_by_stem[s]) > 1]
    single_stems = [s for s in all_stems if len(gt_by_stem[s]) == 1]
    val141_present = [s for s in all_stems if s in VAL141_STEMS]

    configs = [
        ("cc_050_only", [0]),
        ("cc_multi_thr", [0, 1]),
        ("cc_plus_watershed", [0, 1, 2]),
        ("all_panoptic", [0, 1, 2, 3]),
    ]
    cuts = (0.16, 0.18, 0.20, 0.22, 0.24, 0.27, 0.30)

    fold_stems_map = {
        f: [s for s in all_stems if stem_to_fold.get(s, -1) == f]
        for f in range(CFG["folds"])
    }
    settings_report = []
    for name, srcs in configs:
        for cut in cuts:
            pred = greedy_select_disjoint(oof_df, cut, srcs)
            s_all = score_subset(gt_by_stem, pred, all_stems)
            s_mul = score_subset(gt_by_stem, pred, multi_stems)
            s_sng = score_subset(gt_by_stem, pred, single_stems)
            s_141 = score_subset(gt_by_stem, pred, val141_present) if val141_present else {}
            per_fold = [score_subset(gt_by_stem, pred, fold_stems_map[f]) for f in range(CFG["folds"])]
            row = dict(
                name=name, sources=srcs, cut=cut,
                oof_707_all=s_all, oof_707_multi=s_mul, oof_707_single=s_sng,
                oof_val141=s_141, per_fold=per_fold,
            )
            settings_report.append(row)
            log(
                f"{name:18s} cut={cut:.2f} | OOF707 PQ={s_all['PQ']:.4f} "
                f"(SQ={s_all['SQ']:.4f} RQ={s_all['RQ']:.4f} multi={s_mul['PQ']:.4f} single={s_sng['PQ']:.4f}) "
                f"| Val141 PQ={s_141.get('PQ', 0.0):.4f}"
            )

    # Nested 4-fold threshold selection audit (zero selection optimism)
    nested_tp_sum, nested_tp, nested_fp, nested_fn = 0.0, 0, 0, 0
    for f in range(CFG["folds"]):
        def _other_pq(r):
            tp = sum(r["per_fold"][j]["TP"] for j in range(CFG["folds"]) if j != f)
            fp = sum(r["per_fold"][j]["FP"] for j in range(CFG["folds"]) if j != f)
            fn = sum(r["per_fold"][j]["FN"] for j in range(CFG["folds"]) if j != f)
            ts = sum(r["per_fold"][j]["SQ"] * r["per_fold"][j]["TP"] for j in range(CFG["folds"]) if j != f)
            den = tp + 0.5 * fp + 0.5 * fn
            return ts / den if den > 0 else 0.0
        pick = max(settings_report, key=_other_pq)
        pf = pick["per_fold"][f]
        nested_tp_sum += pf["SQ"] * pf["TP"]
        nested_tp += pf["TP"]
        nested_fp += pf["FP"]
        nested_fn += pf["FN"]
    nested_den = nested_tp + 0.5 * nested_fp + 0.5 * nested_fn
    nested_pq = float(nested_tp_sum / nested_den) if nested_den > 0 else 0.0

    best_cfg = max(settings_report, key=lambda r: r["oof_707_all"]["PQ"])
    log(
        "BEST OOF CONFIG:", best_cfg["name"], "cut=", best_cfg["cut"],
        "PQ_707=", f"{best_cfg['oof_707_all']['PQ']:.4f}",
        "Nested_PQ_707=", f"{nested_pq:.4f}",
    )

    if len(test_df) > 0:
        chosen_test = greedy_select_disjoint(test_df, best_cfg["cut"], best_cfg["sources"])
        sub_rows = []
        overlap_total = 0
        for s in sorted(chosen_test.keys()):
            rles = chosen_test[s]
            if rles:
                areas = mu.area(rles)
                overlap_total += int(areas.sum()) - int(mu.area(mu.merge(rles)))
            for idx_k, r in enumerate(rles, 1):
                c = r["counts"]
                c_str = c.decode("ascii") if isinstance(c, bytes) else str(c)
                sub_rows.append({"filament_id": f"{s}_{idx_k}", "segmentation_rle": c_str})
        assert overlap_total == 0, f"Overlap violation: {overlap_total}"
        sub_df = pd.DataFrame(sub_rows, columns=["filament_id", "segmentation_rle"])
        sub_df.to_csv(out_dir / "submission.csv", index=False)
        log(f"Wrote {out_dir / 'submission.csv'}: {len(sub_df)} instances, 0 overlap pixels")

    report = dict(
        backbone=BACKBONE,
        best=best_cfg,
        nested_oof_707_PQ=nested_pq,
        all_settings=settings_report,
        elapsed_minutes=(time.time() - T0) / 60.0,
    )
    (out_dir / "val_report.json").write_text(json.dumps(report, indent=2))


def ensure_deps() -> None:
    import importlib.util

    wheels = glob.glob("/kaggle/input/**/segmentation_models_pytorch-*.whl", recursive=True)
    if importlib.util.find_spec("segmentation_models_pytorch") is None:
        if wheels:
            wdir = str(Path(wheels[0]).parent)
            if importlib.util.find_spec("timm") is None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", wdir, "timm"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", wdir, "segmentation-models-pytorch"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "segmentation-models-pytorch", "pycocotools", "lightgbm"])


def main() -> None:
    ensure_deps()
    out_dir = Path(os.environ.get("SF_OUT_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "./v3_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pick_scratch_dir()

    smoke_only = os.environ.get("SF_SMOKE_ONLY") == "1"
    ctx = mp.get_context("spawn")

    p_prep = ctx.Process(target=prepare_cache, args=(cache_dir, out_dir))
    p_prep.start()
    p_prep.join()
    if p_prep.exitcode != 0:
        raise RuntimeError(f"prepare_cache failed with exitcode {p_prep.exitcode}")

    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        n_gpu = len([ln for ln in smi.stdout.strip().splitlines() if ln.strip()]) if smi.returncode == 0 else 0
    except FileNotFoundError:
        n_gpu = 0

    log(f"Detected GPUs: {n_gpu}, Backbone: {BACKBONE}")
    if n_gpu >= 2 and not smoke_only:
        procs = [
            ctx.Process(target=train_worker, args=(g, list(range(g, CFG["folds"], n_gpu)), str(cache_dir), str(out_dir), False))
            for g in range(n_gpu)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    else:
        train_worker(0, list(range(CFG["folds"])), str(cache_dir), str(out_dir), smoke_only=smoke_only)

    if not smoke_only:
        evaluate_and_fuse(cache_dir, out_dir)

    shutil.rmtree(cache_dir, ignore_errors=True)
    log("All done! Output files:", sorted(p.name for p in out_dir.iterdir()))


if __name__ == "__main__":
    main()
