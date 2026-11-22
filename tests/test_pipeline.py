"""Pipeline logic with a fake detector (no weights/network needed)."""
import json

import cv2
import numpy as np
import pytest

from conftest import needs_torch
from wastectl.config import load_config


class _T:  # minimal tensor stand-in: .cpu().numpy()
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _Boxes:
    xyxy = _T([[10, 10, 50, 50], [60, 10, 90, 40], [5, 60, 30, 90]])
    conf = _T([0.9, 0.8, 0.3])  # third: drawn (>= 0.25) but not counted toward purity (< 0.40)
    cls = _T([2, 3, 4])

    def __len__(self):
        return 3


class _Masks:
    xy = [np.array([[10, 10], [50, 10], [50, 50], [10, 50]], float),
          np.array([[60, 10], [90, 10], [90, 40]], float),
          np.array([[5, 60], [30, 60], [30, 90]], float)]


class _Res:
    boxes, masks = _Boxes(), _Masks()


class FakeYOLO:
    names = {0: "cardboard", 1: "soft_plastic", 2: "rigid_plastic", 3: "metal", 4: "other"}

    def __init__(self, *a, **k):
        pass

    def predict(self, img, **k):
        return [_Res()]


@pytest.fixture
def pipe(monkeypatch, tmp_path):
    ultralytics = pytest.importorskip("ultralytics")
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    from wastectl.vision.pipeline import Pipeline

    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "weights": str(tmp_path)}  # no classifier weights -> detector classes used
    return Pipeline(cfg, yolo_weights="fake.pt"), cfg


@needs_torch
def test_analyze_semantic_map_and_purity(pipe):
    p, cfg = pipe
    img = np.zeros((100, 100, 3), np.uint8)
    r = p.analyze(img)
    assert [d["cls"] for d in r["detections"]] == ["rigid_plastic", "metal", "other"]
    assert r["purity"]["purity"] == 0.5 and r["purity"]["contaminants"] == {"metal": 1}  # `other` conf 0.3 not counted
    sem = r["semantic"]
    assert sem[30, 30] == cfg["classes"].index("rigid_plastic") and sem[95, 95] == 255
    out = p.annotate(img, r)
    assert out.shape == img.shape and out.any()


@needs_torch
def test_infer_image_and_stream(pipe, tmp_path, monkeypatch):
    import wastectl.vision.pipeline as pl

    p, cfg = pipe
    monkeypatch.setattr(pl, "Pipeline", lambda *a, **k: p)
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(4):
        cv2.imwrite(str(frames / f"{i:03d}.jpg"), np.full((100, 100, 3), 40 * i, np.uint8))
    r = pl.infer(cfg, frames / "000.jpg", tmp_path / "out")
    assert (tmp_path / "out" / "000_annotated.jpg").exists() and r["purity"]["n_total"] == 2
    timeline = pl.infer(cfg, frames, tmp_path / "out")
    assert len(timeline) == 4 and all(t["alert"] for t in timeline)  # 50% purity < 0.90
    data = json.loads((tmp_path / "out" / "frames_timeline.json").read_text())
    assert data["frames"][-1]["window_purity"] == 0.5
