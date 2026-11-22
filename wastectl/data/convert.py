"""Source formats -> one record type: {source, image, split_hint, instances=[(class_id, polygon Nx2 normalised)]}."""
import json
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def box_to_polygon(cx, cy, w, h):
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return np.clip(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), 0, 1)


def polygon_box(poly):
    """Normalised polygon -> normalised xyxy box."""
    return (*poly.min(axis=0), *poly.max(axis=0))


def warp_detection(warp_root, classmap):
    """WaRP-D YOLO box labels. Polygons are box rectangles until `data sam` replaces them with SAM masks."""
    d = next(Path(warp_root).glob("**/Warp-D"))
    names = [n.strip() for n in (d / "classes.txt").read_text(encoding="utf-8").splitlines() if n.strip()]
    remap = [classmap.id("warp", n) for n in names]
    for split in ("train", "test"):
        for img in sorted((d / split / "images").iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = d / split / "labels" / f"{img.stem}.txt"
            inst = []
            if lbl.exists():
                for line in lbl.read_text(encoding="utf-8").split("\n"):
                    parts = line.split()
                    if len(parts) == 5 and remap[int(parts[0])] is not None:
                        c, *xywh = parts
                        inst.append((remap[int(c)], box_to_polygon(*map(float, xywh))))
            yield {"source": "warp", "image": img, "split_hint": split, "instances": inst}


def taco_records(taco_root, classmap):
    root = Path(taco_root)
    ann = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    cats = {c["id"]: classmap.id("taco", c["name"]) for c in ann["categories"]}
    by_img = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in ann["images"]:
        path = root / "images" / im["file_name"]
        if not path.exists():
            continue  # failed download; reported by the downloader
        W, H = im["width"], im["height"]
        inst = []
        for a in by_img.get(im["id"], []):
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg:
                poly = max(seg, key=len)  # YOLO-seg takes one polygon per instance: keep the largest part
                pts = np.array(poly, dtype=float).reshape(-1, 2) / [W, H]
            else:
                x, y, w, h = a["bbox"]
                pts = box_to_polygon((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
            if len(pts) >= 3 and cats[a["category_id"]] is not None:
                inst.append((cats[a["category_id"]], np.clip(pts, 0, 1)))
        yield {"source": "taco", "image": path, "split_hint": None, "instances": inst, "wh": (W, H)}


def classifier_folders(root, source, classmap):
    """Folder-per-class crops (WaRP-C, TrashNet): label = the image's parent folder name."""
    for img in sorted(Path(root).rglob("*")):
        # skip macOS zip junk (__MACOSX/, ._resource-fork files) that carries image extensions
        if img.suffix.lower() in IMG_EXT and "__MACOSX" not in img.parts and not img.name.startswith("."):
            c = classmap.id(source, img.parent.name)
            if c is not None:
                yield img, c


def yolo_seg_line(cls, poly):
    return f"{cls} " + " ".join(f"{v:.6f}" for v in poly.reshape(-1))
