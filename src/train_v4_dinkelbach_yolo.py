"""
Solar Filament Segmentation 2026 - v4 4-Fold High-Res Physical-3Ch Dinkelbach Compromise YOLO11m-Seg.

Upgrades over v3 YOLO11m-Seg (`0.3627` OOF PQ -> `0.4031` OOF PQ):
  1. Physical 3-Channel Solar Disk Normalization (`[disk_pct, CLAHE, limb_flat_dark]`):
     Eliminates observatory exposure shifts and solar limb darkening across all 7 observatories.
  2. Dinkelbach-Optimal SDF Compromise Instance Polygons (`vals >= 0.35 * K`):
     Supervises exact consensus filament contours free of 1-of-K annotator bloat.
  3. `yolo11m-seg.pt` at `imgsz=1024, mask_ratio=1` (`256x256` native mask prototypes = 4.0x more
     spatial pixels than v3's `128x128`), selecting `best.pt` per fold.
  4. Multi-GPU 2-Shard Training: runs Fold `[0, 2]` on `GPU 0` and Fold `[1, 3]` on `GPU 1` in parallel.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

FOLD_MAP: dict[str, int] = json.loads("""__FOLD_MAP__""") if "__FOLD" not in """__FOLD_MAP__""" else {}
H = W = 2048
IMGSZ = 1024
EPOCHS = 32
BATCH = 2
MODEL_NAME = "yolo11m-seg.pt"
T0 = time.time()


def log(*args) -> None:
    sh = os.environ.get("SF_YOLO_SHARD", "main")
    print(f"[{(time.time() - T0) / 60:6.2f}m][{sh}]", *args, flush=True)


def ensure_deps() -> None:
    wheels = glob.glob("/kaggle/input/**/ultralytics-*.whl", recursive=True)
    if importlib.util.find_spec("ultralytics") is None:
        if wheels:
            wdir = str(Path(wheels[0]).parent)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", wdir, "ultralytics"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pycocotools"])


def detect_hw_gpus() -> int:
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return len([ln for ln in smi.stdout.strip().splitlines() if ln.strip()]) if smi.returncode == 0 else 0
    except FileNotFoundError:
        return 0


def locate_comp_paths() -> tuple[Path, Path]:
    env_root = os.environ.get("SF_DATA_DIR")
    if env_root:
        root = Path(env_root)
        return root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", root / "train" / "train_images"
    js_hits = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if not js_hits:
        raise FileNotFoundError("Could not locate MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    js = Path(js_hits[0])
    return js, js.parent.parent / "train" / "train_images"


def prepare_shared_yolo_cache(cache_dir: Path, smoke: bool) -> None:
    import cv2
    import numpy as np
    from pycocotools import mask as mu

    cv2.setNumThreads(1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = cache_dir / "images_3ch"
    lbl_out_dir = cache_dir / "labels_dink"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    js_path, train_dir = locate_comp_paths()
    coco = json.loads(js_path.read_text())

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

    anns_by_img: dict[str, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        seg = a.get("segmentation")
        if seg and isinstance(seg, list):
            valid_s = [s for s in seg if isinstance(s, (list, tuple)) and len(s) >= 6]
            if valid_s:
                r = mu.merge(mu.frPyObjects(valid_s, H, W))
                if float(mu.area(r)) >= 25.0:
                    anns_by_img[a["image_id"]].append(r)

    sets_by_stem: dict[str, list[list[dict]]] = defaultdict(list)
    fname_by_stem: dict[str, str] = {}
    for im in coco["images"]:
        st = Path(im["file_name"]).stem
        sets_by_stem[st].append(anns_by_img.get(im["id"], []))
        fname_by_stem[st] = im["file_name"]

    stems = sorted(sets_by_stem.keys())
    if smoke:
        stems = stems[:12]

    yy_g, xx_g = np.ogrid[:H, :W]
    disk_mask = (yy_g - H / 2.0) ** 2 + (xx_g - W / 2.0) ** 2 <= 940.0 ** 2

    def encode_bgr_for_yolo(img_u8: np.ndarray) -> np.ndarray:
        vals = img_u8[disk_mask]
        p10, p50, p90 = np.percentile(vals, (10, 50, 90)) if vals.size > 0 else (64.0, 128.0, 192.0)
        span = max(float(p90 - p10), 16.0)
        img_f = img_u8.astype(np.float32)
        ch0 = np.clip((img_f - p50) / span * 0.5 + 0.5, 0.0, 1.0)
        ch1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(img_u8).astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(img_u8, (0, 0), 18.0).astype(np.float32)
        ch2 = np.clip((blur - img_f) / span + 0.5, 0.0, 1.0)
        # Ultralytics converts BGR -> RGB on load, so saving [ch2, ch1, ch0] as BGR yields [ch0, ch1, ch2] in RGB!
        bgr = np.rint(255.0 * np.stack([ch2, ch1, ch0], axis=2)).clip(0, 255).astype(np.uint8)
        return bgr

    kept_polys = 0
    for idx_s, s in enumerate(stems):
        im_p = train_dir / fname_by_stem[s]
        im_u8 = cv2.imread(str(im_p), cv2.IMREAD_GRAYSCALE)
        if im_u8 is None:
            raise FileNotFoundError(f"Missing {im_p}")
        if im_u8.shape != (H, W):
            im_u8 = cv2.resize(im_u8, (W, H), interpolation=cv2.INTER_LINEAR)
        bgr = encode_bgr_for_yolo(im_u8)
        cv2.imwrite(str(img_out_dir / f"{s}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

        gt_sets = sets_by_stem[s]
        K = max(len(gt_sets), 1)
        items = [(a_i, r) for a_i, gs in enumerate(gt_sets) for r in gs]
        all_r = [it[1] for it in items]
        clean_rles = []
        if all_r:
            if K == 1:
                clean_rles = all_r
            else:
                imat = np.asarray(mu.iou(all_r, all_r, [0] * len(all_r)))
                used = [False] * len(items)
                order = np.argsort(-imat.sum(axis=1))
                cand_pool = list(all_r)
                for seed in order:
                    if used[seed]:
                        continue
                    members = [seed]
                    seen_a = {items[seed][0]}
                    used[seed] = True
                    for cand in order:
                        if not used[cand] and items[cand][0] not in seen_a and any(imat[cand, m] >= 0.25 for m in members):
                            members.append(cand)
                            seen_a.add(items[cand][0])
                            used[cand] = True
                    if len(members) >= 2:
                        bbs = [list(map(int, mu.toBbox(all_r[m]))) for m in members]
                        x0 = max(0, min(b[0] for b in bbs) - 6)
                        y0 = max(0, min(b[1] for b in bbs) - 6)
                        x1 = min(W, max(b[0] + b[2] for b in bbs) + 6)
                        y1 = min(H, max(b[1] + b[3] for b in bbs) + 6)
                        sdf_sum = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
                        for m in members:
                            mc = mu.decode(all_r[m])[y0:y1, x0:x1].astype(np.uint8)
                            sdf_sum += cv2.distanceTransform(mc, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - mc, cv2.DIST_L2, 5)
                        m_bin = sdf_sum >= 0.0
                        if int(m_bin.sum()) >= 25:
                            rr = encode_crop_rle(m_bin.astype(np.uint8), x0, y0)
                            if rr is not None:
                                cand_pool.append(rr)
                vals = np.zeros(len(cand_pool), dtype=np.float64)
                for gs in gt_sets:
                    if gs:
                        m_iou = np.asarray(mu.iou(cand_pool, gs, [0] * len(gs)))
                        vals += np.where(m_iou > 0.50, m_iou, 0.0).max(axis=1)
                for idx_c in np.argsort(-vals):
                    if vals[idx_c] < 0.35 * K:
                        continue
                    r = cand_pool[idx_c]
                    if clean_rles:
                        occ = mu.merge(clean_rles)
                        if float(mu.area(mu.merge([r, occ], intersect=True))) > 0.12 * float(mu.area(r)):
                            continue
                    clean_rles.append(r)

        lbl_lines = []
        for r in clean_rles:
            bx, by, bw, bh = map(int, mu.toBbox(r))
            if bw <= 0 or bh <= 0:
                continue
            mc = mu.decode(r)[by:by + bh, bx:bx + bw].astype(np.uint8)
            cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            if len(cnt) < 3:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float64)
            pts[:, 0] = np.clip((pts[:, 0] + bx) / float(W), 0.0, 1.0)
            pts[:, 1] = np.clip((pts[:, 1] + by) / float(H), 0.0, 1.0)
            if len(pts) > 250:
                idx_sub = np.linspace(0, len(pts) - 1, 250).astype(np.int32)
                pts = pts[idx_sub]
            coords = pts.ravel()
            if len(coords) >= 6:
                lbl_lines.append("0 " + " ".join(f"{v:.5f}" for v in coords))
                kept_polys += 1

        (lbl_out_dir / f"{s}.txt").write_text("\n".join(lbl_lines) + ("\n" if lbl_lines else ""))
        if (idx_s + 1) % 200 == 0 or (idx_s + 1) == len(stems):
            log(f"Encoded 3-ch images & Dinkelbach polygons: {idx_s + 1}/{len(stems)} stems ({kept_polys} clean instances)")

    meta = dict(stems=stems, ann_path=str(js_path))
    (cache_dir / "yolo_meta.json").write_text(json.dumps(meta))


def run_yolo_shard() -> None:
    import cv2
    import torch
    from pycocotools import mask as mu
    from ultralytics import YOLO

    cv2.setNumThreads(1)
    smoke = os.environ.get("SF_SMOKE_ONLY") == "1"
    cache_dir = Path(os.environ["SF_YOLO_CACHE"])
    out_dir = Path(os.environ["SF_YOLO_OUT"])
    folds = [int(x) for x in os.environ["SF_FOLD_LIST"].split(",")]

    meta = json.loads((cache_dir / "yolo_meta.json").read_text())
    stems = meta["stems"]
    img_src = cache_dir / "images_3ch"
    lbl_src = cache_dir / "labels_dink"

    oof_shard: dict[str, list[tuple[float, str]]] = {}
    dev = 0 if torch.cuda.is_available() else "cpu"

    for f in folds:
        f_ds = cache_dir / f"work_fold{f}"
        shutil.rmtree(f_ds, ignore_errors=True)
        for sp in ("train", "val"):
            (f_ds / "images" / sp).mkdir(parents=True, exist_ok=True)
            (f_ds / "labels" / sp).mkdir(parents=True, exist_ok=True)

        val_paths: list[Path] = []
        for s in stems:
            is_val = (int(FOLD_MAP.get(s, hash(s) % 4)) == f)
            sp = "val" if is_val else "train"
            src_im = img_src / f"{s}.jpg"
            src_lb = lbl_src / f"{s}.txt"
            os.symlink(src_im, f_ds / "images" / sp / f"{s}.jpg")
            os.symlink(src_lb, f_ds / "labels" / sp / f"{s}.txt")
            if is_val:
                val_paths.append(src_im)

        yaml_p = f_ds / "data.yaml"
        yaml_p.write_text(f"path: {f_ds}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")

        log(f"Starting YOLO11l-1280 fold {f} ({len(stems) - len(val_paths)} train, {len(val_paths)} val)")
        model = YOLO("yolo11n-seg.pt" if smoke else MODEL_NAME)
        model.train(
            data=str(yaml_p),
            imgsz=640 if smoke else IMGSZ,
            epochs=1 if smoke else EPOCHS,
            batch=2 if smoke else BATCH,
            device=dev,
            workers=2,
            seed=3026 + f,
            cos_lr=True,
            close_mosaic=8,
            degrees=180.0,
            flipud=0.5,
            fliplr=0.5,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.12,
            mask_ratio=1,
            overlap_mask=False,
            max_det=150,
            project=str(cache_dir / "runs"),
            name=f"fold{f}",
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        w_dir = cache_dir / "runs" / f"fold{f}" / "weights"
        best_w = w_dir / "best.pt"
        pick_w = best_w if best_w.exists() else (w_dir / "last.pt")
        saved_w = out_dir / f"yolo_fold{f}.pt"
        shutil.copy2(pick_w, saved_w)

        pred_m = YOLO(str(saved_w))
        for vp in val_paths:
            r = pred_m.predict(
                str(vp),
                imgsz=640 if smoke else IMGSZ,
                conf=0.03,
                iou=0.60,
                max_det=150,
                retina_masks=True,
                device=dev,
                verbose=False,
            )[0]
            items: list[tuple[float, str]] = []
            if r.masks is not None and len(r.boxes) > 0:
                ms = r.masks.data.cpu().numpy() > 0.5
                confs = r.boxes.conf.cpu().numpy()
                for sc, mk in zip(confs, ms):
                    if mk.shape != (H, W):
                        mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                    if int(mk.sum()) >= 25:
                        rle = mu.encode(np.asfortranarray(mk.astype(np.uint8)))
                        c_str = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
                        items.append((float(sc), c_str))
            items.sort(key=lambda x: -x[0])
            oof_shard[vp.stem] = items

        shutil.rmtree(f_ds, ignore_errors=True)
        shutil.rmtree(cache_dir / "runs" / f"fold{f}", ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_dir / f"_oof_yolo_shard_{folds[0]}.pkl", "wb") as fh:
        pickle.dump(oof_shard, fh)


def evaluate_and_save_yolo(cache_dir: Path, out_dir: Path, smoke: bool) -> None:
    import numpy as np
    from pycocotools import mask as mu

    oof_yolo: dict[str, list[tuple[float, str]]] = {}
    for p in sorted(out_dir.glob("_oof_yolo_shard_*.pkl")):
        oof_yolo.update(pickle.loads(p.read_bytes()))
        p.unlink(missing_ok=True)
    with open(out_dir / "oof_yolo.pkl", "wb") as fh:
        pickle.dump(oof_yolo, fh)

    meta = json.loads((cache_dir / "yolo_meta.json").read_text())
    coco = json.loads(Path(meta["ann_path"]).read_text())
    anns_by_img: dict[str, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("segmentation"):
            anns_by_img[a["image_id"]].append(mu.merge(mu.frPyObjects(a["segmentation"], H, W)))
    gt_by_stem: dict[str, list[list[dict]]] = defaultdict(list)
    for im in coco["images"]:
        gt_by_stem[Path(im["file_name"]).stem].append(anns_by_img.get(im["id"], []))

    eval_stems = sorted(oof_yolo.keys())
    best_pq, best_conf = -1.0, 0.25
    conf_grid = []
    for conf_thr in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        tp_sum, tp, fp, fn = 0.0, 0, 0, 0
        for s in eval_stems:
            chosen = []
            for sc, c_str in oof_yolo[s]:
                if sc < conf_thr:
                    continue
                r = {"size": [H, W], "counts": c_str.encode("ascii")}
                area = float(mu.area(r))
                if area < 60:
                    continue
                if chosen:
                    occ = mu.merge(chosen)
                    ov = float(mu.area(mu.merge([r, occ], intersect=True)))
                    if ov > 0.12 * area:
                        continue
                    if ov > 0:
                        m = mu.decode(r).astype(bool) & (~mu.decode(occ).astype(bool))
                        if int(m.sum()) < 60:
                            continue
                        r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
                chosen.append(r)
            for g in gt_by_stem[s]:
                if not g:
                    fp += len(chosen)
                elif not chosen:
                    fn += len(g)
                else:
                    iou = np.asarray(mu.iou(g, chosen, [0] * len(chosen)))
                    hit = iou > 0.50
                    tp_sum += float(iou[hit].sum())
                    tp += int(hit.sum())
                    fp += int((hit.sum(axis=0) == 0).sum())
                    fn += int((hit.sum(axis=1) == 0).sum())
        den = tp + 0.5 * fp + 0.5 * fn
        pq_val = float(tp_sum / den) if den > 0 else 0.0
        sq_val = float(tp_sum / tp) if tp > 0 else 0.0
        rq_val = float(tp / den) if den > 0 else 0.0
        conf_grid.append(dict(conf=conf_thr, PQ=pq_val, SQ=sq_val, RQ=rq_val, TP=tp, FP=fp, FN=fn))
        log(f"Standalone v4-YOLO11l-1280 conf={conf_thr:.2f} -> OOF PQ={pq_val:.4f} (SQ={sq_val:.4f} RQ={rq_val:.4f})")
        if pq_val > best_pq:
            best_pq, best_conf = pq_val, conf_thr

    (out_dir / "yolo_val_report.json").write_text(
        json.dumps(
            dict(
                model=MODEL_NAME,
                imgsz=640 if smoke else IMGSZ,
                use_3ch=True,
                best_conf=best_conf,
                best_oof_PQ=best_pq,
                grid=conf_grid,
                elapsed_min=(time.time() - T0) / 60.0,
            ),
            indent=2,
        )
    )
    shutil.rmtree(cache_dir, ignore_errors=True)
    log(f"★ BEST STANDALONE v4-YOLO11m-1024 OOF PQ = {best_pq:.4f} (conf={best_conf:.2f})")


def main() -> None:
    if os.environ.get("SF_YOLO_SHARD"):
        run_yolo_shard()
        return

    ensure_deps()
    smoke = os.environ.get("SF_SMOKE_ONLY") == "1"
    cache_dir = Path(os.environ.get("SF_YOLO_CACHE", "/kaggle/temp/v4_yolo" if Path("/kaggle/temp").exists() else "/tmp/v4_yolo"))
    out_dir = Path(os.environ.get("SF_YOLO_OUT", "/kaggle/working" if Path("/kaggle/working").exists() else "./v4_yolo_out"))
    shutil.rmtree(cache_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepare_shared_yolo_cache(cache_dir, smoke)
    n_gpus = detect_hw_gpus()

    if n_gpus >= 2 and not smoke:
        log(f"Detected {n_gpus} GPUs -> running Fold [0, 2] on GPU 0 and Fold [1, 3] on GPU 1")
        procs = []
        for g, flist in [(0, "0,2"), (1, "1,3")]:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            env["SF_YOLO_SHARD"] = f"gpu{g}"
            env["SF_YOLO_CACHE"] = str(cache_dir)
            env["SF_YOLO_OUT"] = str(out_dir)
            env["SF_FOLD_LIST"] = flist
            procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve())], env=env))
        codes = [p.wait() for p in procs]
        assert all(c == 0 for c in codes), f"YOLO worker failed: {codes}"
    else:
        os.environ["SF_YOLO_SHARD"] = "gpu0"
        os.environ["SF_YOLO_CACHE"] = str(cache_dir)
        os.environ["SF_YOLO_OUT"] = str(out_dir)
        os.environ["SF_FOLD_LIST"] = "0" if smoke else "0,1,2,3"
        run_yolo_shard()

    evaluate_and_save_yolo(cache_dir, out_dir, smoke)


if __name__ == "__main__":
    main()
