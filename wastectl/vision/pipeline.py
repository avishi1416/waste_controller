"""Main product pipeline: YOLO-seg -> crops -> ResNet/GoogLeNet re-classification -> semantic map -> purity.

The classifier is optional: without it the detector's class is used. Missing weights are reported, never faked.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from wastectl.purity import StreamMonitor, frame_purity

# BGR colours per unified class, for overlays (colour-blind-safe Okabe-Ito palette)
COLORS = {"cardboard": (0, 159, 230), "soft_plastic": (233, 180, 86), "rigid_plastic": (115, 158, 0),
          "metal": (0, 94, 213), "other": (167, 121, 204)}
BACKGROUND = 255
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def default_weights(cfg):
    w = Path(cfg["paths"]["weights"])
    return {"yolo": w / f"{Path(cfg['models']['yolo']['weights']).stem}_waste.pt",
            "cls": {b: w / f"cls_{b}.pt" for b in cfg["models"]["classifier"]["backbones"]}}


class Pipeline:
    def __init__(self, cfg, yolo_weights=None, cls_backbone="resnet18", device="cpu", use_classifier=True):
        from ultralytics import YOLO

        self.cfg, self.device, self.classes = cfg, device, cfg["classes"]
        dw = default_weights(cfg)
        self.yolo_path = Path(yolo_weights or dw["yolo"])
        if not self.yolo_path.exists() and yolo_weights is None:
            raise FileNotFoundError(f"detector not trained yet: {self.yolo_path} (TODO.md step 2)")
        self.yolo = YOLO(str(self.yolo_path))
        self.status = {"detector": str(self.yolo_path), "classifier": None}
        self.cls, self.tf = None, None
        cls_path = dw["cls"].get(cls_backbone)
        if use_classifier and cls_path and cls_path.exists():
            from wastectl.vision import classifier

            self.cls, ck = classifier.load(cls_path, device)
            self.tf = classifier.transforms(ck["imgsz"], train=False)
            self.status["classifier"] = str(cls_path)
        p = cfg["purity"]
        self.conf_draw, self.conf_count = p["conf_draw"], p["conf_count"]

    def _det_class(self, idx):
        name = self.yolo.names[int(idx)]
        return name if name in self.classes else "other"  # non-waste models (smoke tests) map to `other`

    def _classify(self, rgb, boxes):
        import torch
        from PIL import Image

        crops = [self.tf(Image.fromarray(rgb[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)])) for x1, y1, x2, y2 in boxes]
        with torch.no_grad():
            p = torch.softmax(self.cls(torch.stack(crops).to(self.device)), 1).cpu().numpy()
        return p.argmax(1), p.max(1)

    def analyze(self, image, target=None):
        """image: path or BGR ndarray. Returns detections, semantic map, purity and timing."""
        bgr = cv2.imread(str(image)) if not isinstance(image, np.ndarray) else image
        if bgr is None:
            raise ValueError(f"cannot read image {image}")
        t0 = time.perf_counter()
        res = self.yolo.predict(bgr, conf=self.conf_draw, device=self.device, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            det_cls = [self._det_class(c) for c in res.boxes.cls.cpu().numpy()]
            polys = res.masks.xy if res.masks is not None else [None] * len(boxes)
            if self.cls is not None:
                ci, cc = self._classify(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), boxes)
            for i, (b, c, dc, poly) in enumerate(zip(boxes, confs, det_cls, polys)):
                final = self.classes[ci[i]] if self.cls is not None else dc
                dets.append({"box": b.round(1).tolist(), "conf": float(c), "det_cls": dc, "cls": final,
                             "cls_conf": float(cc[i]) if self.cls is not None else None,
                             "polygon": poly.round(1).tolist() if poly is not None and len(poly) else None})
        semantic = np.full(bgr.shape[:2], BACKGROUND, np.uint8)
        for d in sorted(dets, key=lambda d: d["conf"]):  # higher confidence painted last, so it wins overlaps
            if d["polygon"]:
                cv2.fillPoly(semantic, [np.array(d["polygon"], np.int32)], self.classes.index(d["cls"]))
        latency = time.perf_counter() - t0
        target = target or self.cfg["purity"]["target_stream"]
        return {"detections": dets, "semantic": semantic, "latency_s": latency,
                "purity": frame_purity(dets, target, self.conf_count), "target": target}

    def annotate(self, bgr, result):
        out = bgr.copy()
        overlay = bgr.copy()
        for i, c in enumerate(self.classes):
            overlay[result["semantic"] == i] = COLORS[c]
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
        for d in result["detections"]:
            x1, y1, x2, y2 = map(int, d["box"])
            col = COLORS[d["cls"]]
            contaminant = d["cls"] != result["target"] and d["conf"] >= self.conf_count
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255) if contaminant else col, 3 if contaminant else 2)
            label = f"{d['cls']} {d['conf']:.2f}" + (" CONTAMINANT" if contaminant else "")
            cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        p = result["purity"]
        txt = f"stream: {result['target']} | purity: " + ("n/a" if p["purity"] is None else f"{p['purity']:.0%}")
        cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out


def frames(source):
    """Yield BGR frames from a video file or a folder of images (sorted = stream order)."""
    src = Path(source)
    if src.is_dir():
        for p in sorted(x for x in src.iterdir() if x.suffix.lower() in IMG_EXT):
            yield cv2.imread(str(p))
        return
    cap = cv2.VideoCapture(str(src))
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f
    finally:
        cap.release()


def _jsonable(r):
    return {k: v for k, v in r.items() if k != "semantic"}


def infer(cfg, source, out_dir, target=None, device="cpu", yolo_weights=None, fps=10):
    pipe = Pipeline(cfg, yolo_weights=yolo_weights, device=device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    if src.is_file() and src.suffix.lower() in IMG_EXT:
        bgr = cv2.imread(str(src))
        r = pipe.analyze(bgr, target)
        cv2.imwrite(str(out / f"{src.stem}_annotated.jpg"), pipe.annotate(bgr, r))
        (out / f"{src.stem}.json").write_text(json.dumps({**_jsonable(r), "models": pipe.status}, indent=2), encoding="utf-8")
        return r
    monitor = StreamMonitor.from_config(cfg, target)
    writer, timeline = None, []
    for i, bgr in enumerate(frames(src)):
        r = pipe.analyze(bgr, monitor.target)
        s = monitor.update(r["detections"])
        timeline.append({"frame": i, **{k: s[k] for k in ("purity", "window_purity", "alert", "n_total", "contaminants")},
                         "latency_s": r["latency_s"]})
        img = pipe.annotate(bgr, r)
        if s["alert"]:
            cv2.putText(img, "CONTAMINATION ALERT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{src.stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
        writer.write(img)
    if writer:
        writer.release()
    (out / f"{src.stem}_timeline.json").write_text(json.dumps({"models": pipe.status, "frames": timeline}, indent=2), encoding="utf-8")
    return timeline


def bench(cfg, images, device="cpu", yolo_weights=None, warmup=3):
    """End-to-end latency/FPS on a list of images (+ peak GPU memory when on CUDA)."""
    import torch

    pipe = Pipeline(cfg, yolo_weights=yolo_weights, device=device)
    imgs = [cv2.imread(str(p)) for p in images]
    for im in imgs[:warmup]:
        pipe.analyze(im)
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.reset_peak_memory_stats()
    lat = np.array([pipe.analyze(im)["latency_s"] for im in imgs])
    m = {"device": str(device), "n_images": len(imgs), "mean_latency_ms": float(lat.mean() * 1000),
         "p95_latency_ms": float(np.percentile(lat, 95) * 1000), "fps": float(1 / lat.mean()), "models": pipe.status}
    if torch.cuda.is_available() and str(device) != "cpu":
        m["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return m
