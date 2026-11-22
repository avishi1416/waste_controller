"""Recall by object size: which ground-truth objects does the detector find (IoU >= 0.5), split into size terciles."""
import json
from pathlib import Path

import numpy as np


def box_iou(box, boxes):
    """IoU of one xyxy box against an (N, 4) array of xyxy boxes."""
    ix = np.clip(np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]), 0, None)
    inter = ix * iy
    area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + area - inter)


def summarise(rows):
    """rows: (relative_area, found_any_class, found_right_class) per ground-truth object."""
    a = np.array(rows, float)
    qs = np.quantile(a[:, 0], [0, 1 / 3, 2 / 3, 1])
    out = {"n": len(a), "found": float(a[:, 1].mean()), "found_right_class": float(a[:, 2].mean()), "terciles": []}
    for lo, hi, name in zip(qs[:-1], qs[1:], ["small", "medium", "large"]):
        s = a[(a[:, 0] >= lo) & (a[:, 0] <= hi)]
        out["terciles"].append({"name": name, "area_pct": [float(lo * 100), float(hi * 100)], "n": len(s),
                                "found": float(s[:, 1].mean()), "found_right_class": float(s[:, 2].mean())})
    return out


def run(cfg, weights, split="test", conf=0.25):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    root = Path(cfg["paths"]["data_processed"]) / "yolo"
    rows = []
    for img in sorted((root / "images" / split).glob("*.jpg")):
        gt = []
        for line in (root / "labels" / split / f"{img.stem}.txt").read_text(encoding="utf-8").split("\n"):
            v = line.split()
            if len(v) >= 7:
                p = np.array(v[1:], float).reshape(-1, 2)
                gt.append((int(v[0]), *p.min(0), *p.max(0)))
        r = model.predict(str(img), imgsz=cfg["models"]["yolo"]["imgsz"], conf=conf, verbose=False)[0]
        pb = r.boxes.xyxyn.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        pc = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
        for c, *box in gt:
            hit = box_iou(box, pb) >= 0.5 if len(pb) else np.zeros(0, bool)
            rows.append(((box[2] - box[0]) * (box[3] - box[1]), bool(hit.any()), bool((hit & (pc == c)).any())))
    res = summarise(rows)
    out = Path(cfg["paths"]["experiments"]) / "eval" / f"size_recall_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
