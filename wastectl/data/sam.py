"""`python -m wastectl data sam`: replace WaRP box-rectangle polygons with SAM masks prompted by the GT boxes (Q28a).

Run on the GPU laptop after `data prepare`. Masks are pseudo-labels; their quality is checked on WaRP-S real masks.
"""
import json
from pathlib import Path

import numpy as np

from wastectl.data.convert import polygon_box, yolo_seg_line


def _read_labels(path):
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        v = line.split()
        if len(v) >= 7:
            out.append((int(v[0]), np.array(v[1:], dtype=float).reshape(-1, 2)))
    return out


def sam_refine(cfg, model_name="sam2.1_b.pt", device=None):
    from PIL import Image
    from ultralytics import SAM

    yolo = Path(cfg["paths"]["data_processed"]) / "yolo"
    model = SAM(model_name)
    done = fallback = 0
    for lbl in sorted((yolo / "labels").glob("*/warp_*.txt")):
        img = yolo / "images" / lbl.parent.name / f"{lbl.stem}.jpg"
        inst = _read_labels(lbl)
        if not inst:
            continue
        W, H = Image.open(img).size
        boxes = [[x1 * W, y1 * H, x2 * W, y2 * H] for x1, y1, x2, y2 in (polygon_box(p) for _, p in inst)]
        res = model(str(img), bboxes=boxes, device=device, verbose=False)[0]
        polys = res.masks.xyn if res.masks is not None else []
        lines = []
        for i, (c, rect) in enumerate(inst):
            poly = polys[i] if i < len(polys) and len(polys[i]) >= 3 else None
            if poly is None:
                fallback += 1  # SAM gave nothing usable: keep the box rectangle rather than lose the object
            lines.append(yolo_seg_line(c, rect if poly is None else np.asarray(poly)))
        lbl.write_text("\n".join(lines), encoding="utf-8")
        done += 1
    mpath = Path(cfg["paths"]["data_processed"]) / "manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["warp_masks"] = f"SAM pseudo-masks ({model_name}); {fallback} instances kept as box rectangles"
    mpath.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"SAM: refined {done} WaRP label files, {fallback} rectangle fallbacks")
