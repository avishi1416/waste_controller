"""`python -m wastectl data prepare`: raw sources -> data/processed/{yolo,crops} + manifest + class counts.

Split policy: with WaRP present, test = WaRP test only and TACO is train/val. Without WaRP (laptop plan,
CLAUDE.md Round 5) TACO is split 80/10/10 train/val/test. Seed from config.
"""
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from wastectl.data.classmap import ClassMap
from wastectl.data.convert import classifier_folders, polygon_box, taco_records, warp_detection, yolo_seg_line
from wastectl.data.dedupe import dhash, keep_mask

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1  # TACO test share, used only when WaRP is absent
SPLIT_ORDER = {"test": 0, "val": 1, "train": 2}  # dedupe keeps test images first, so train never copies test


def _split_train_val(items, rng):
    idx = rng.permutation(len(items))
    n_val = max(1, int(len(items) * VAL_FRACTION)) if items else 0
    val = set(idx[:n_val].tolist())
    return ["val" if i in val else "train" for i in range(len(items))]


def _assign_splits(records, rng):
    warp_train = [r for r in records if r["source"] == "warp" and r["split_hint"] == "train"]
    # shortcut: groups by file (no sequence ids known); if WaRP names reveal sequences, group by them
    for r, s in zip(warp_train, _split_train_val(warp_train, rng)):
        r["split"] = s
    for r in records:
        if r["source"] == "warp" and r["split_hint"] == "test":
            r["split"] = "test"
    taco = [r for r in records if r["source"] == "taco"]
    has_warp = any(r["source"] == "warp" for r in records)
    idx = rng.permutation(len(taco))
    n_test = 0 if has_warp else int(len(taco) * TEST_FRACTION)
    n_val = int(len(taco) * VAL_FRACTION)
    for rank, i in enumerate(idx):
        taco[i]["split"] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"


def _out_name(r, raw):
    rel = Path(r["image"]).relative_to(raw).with_suffix("")
    return f"{r['source']}_" + "_".join(rel.parts[-2:] if r["source"] == "taco" else [rel.name])


def _write_image(r, dest):
    if r["source"] == "warp":
        shutil.copy2(r["image"], dest)
        return True
    img = ImageOps.exif_transpose(Image.open(r["image"])).convert("RGB")
    W, H = r["wh"]
    if (img.width > img.height) != (W > H) and abs(W - H) > 0.05 * max(W, H):
        return False  # orientation disagrees with the annotation: labels would be rotated
    img.save(dest, quality=95)
    return True


def _crop(img_path, poly, dest, min_side=32, pad=0.05):
    img = Image.open(img_path)
    x1, y1, x2, y2 = polygon_box(poly)
    w, h = x2 - x1, y2 - y1
    box = [(x1 - pad * w) * img.width, (y1 - pad * h) * img.height, (x2 + pad * w) * img.width, (y2 + pad * h) * img.height]
    box = [max(0, box[0]), max(0, box[1]), min(img.width, box[2]), min(img.height, box[3])]
    if box[2] - box[0] < min_side or box[3] - box[1] < min_side:
        return False
    img.crop(box).convert("RGB").save(dest, quality=95)
    return True


def prepare(cfg, crops=True):
    raw, out = Path(cfg["paths"]["data_raw"]), Path(cfg["paths"]["data_processed"])
    classes = cfg["classes"]
    cmap = ClassMap(classes)
    rng = np.random.default_rng(cfg["seed"])
    have = {s: (raw / s).exists() for s in ("warp", "taco", "trashnet")}
    if not any(have.values()):
        raise SystemExit(f"no datasets in {raw}; run `python -m wastectl data download` first")
    missing = [s for s, ok in have.items() if not ok]
    if missing:
        print(f"WARNING: missing sources {missing}; continuing with the rest")
    if out.exists():
        shutil.rmtree(out)  # processed data is fully derived from raw; rebuild from scratch

    # detection / segmentation
    records = []
    if have["warp"]:
        records += list(warp_detection(raw / "warp", cmap))
    if have["taco"]:
        records += list(taco_records(raw / "taco", cmap))
    _assign_splits(records, rng)
    records.sort(key=lambda r: SPLIT_ORDER[r["split"]])
    keep = keep_mask([dhash(r["image"]) for r in records])
    removed = Counter(r["split"] for r, k in zip(records, keep) if not k)
    records = [r for r, k in zip(records, keep) if k]

    counts, skipped = Counter(), Counter()
    for r in records:
        img_dir, lbl_dir = out / "yolo" / "images" / r["split"], out / "yolo" / "labels" / r["split"]
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        name = _out_name(r, raw)
        dest = img_dir / f"{name}.jpg"
        if not _write_image(r, dest):
            skipped["orientation_mismatch"] += 1
            continue
        (lbl_dir / f"{name}.txt").write_text("\n".join(yolo_seg_line(c, p) for c, p in r["instances"]), encoding="utf-8")
        for c, p in r["instances"]:
            counts[("det", r["split"], r["source"], classes[c])] += 1
            # TACO object crops feed the classifier (train/val only); WaRP crops come from WaRP-C
            if crops and r["source"] == "taco":
                cdir = out / "crops" / r["split"] / classes[c]
                cdir.mkdir(parents=True, exist_ok=True)
                n = counts[("crop", r["split"], "taco", classes[c])]
                if _crop(dest, p, cdir / f"{name}_{n}.jpg"):
                    counts[("crop", r["split"], "taco", classes[c])] += 1

    # classifier folders
    def add_crops(items, split_fn, source):
        items = list(items)
        for (img, c), split in zip(items, split_fn(items)):
            cdir = out / "crops" / split / classes[c]
            cdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, cdir / f"{source}_{img.parent.name}_{img.name}")
            counts[("crop", split, source, classes[c])] += 1

    if crops and have["warp"]:
        wc = next((raw / "warp").glob("**/Warp-C"))
        add_crops(classifier_folders(wc / "train_crops", "warp", cmap), lambda it: _split_train_val(it, rng), "warp")
        add_crops(classifier_folders(wc / "test_crops", "warp", cmap), lambda it: ["test"] * len(it), "warp")
    if crops and have["trashnet"]:
        add_crops(classifier_folders(raw / "trashnet", "trashnet", cmap), lambda it: _split_train_val(it, rng), "trashnet")

    # ultralytics dataset file; val falls back to train split names if a split is empty
    yolo = out / "yolo"
    splits = {s: f"images/{s}" for s in ("train", "val", "test") if (yolo / "images" / s).exists()}
    (yolo / "data.yaml").write_text(
        "\n".join([f"path: {yolo.resolve().as_posix()}", *(f"{k}: {v}" for k, v in splits.items()),
                   "names:", *(f"  {i}: {c}" for i, c in enumerate(classes))]) + "\n", encoding="utf-8")

    exp = Path(cfg["paths"]["experiments"]) / "data"
    exp.mkdir(parents=True, exist_ok=True)
    with open(exp / "class_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "split", "source", "class", "count"])
        w.writerows([*k, v] for k, v in sorted(counts.items()))
    manifest = {
        "sources": have,
        "images": dict(Counter(r["split"] for r in records)),
        "dedupe_removed": dict(removed),
        "skipped": dict(skipped),
        "warp_masks": "box-rectangle (run `data sam` for SAM pseudo-masks)" if have["warp"] else None,
        "seed": cfg["seed"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (exp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
