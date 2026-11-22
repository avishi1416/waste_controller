import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wastectl.config import load_config
from wastectl.data.classmap import ClassMap
from wastectl.data.convert import box_to_polygon, polygon_box
from wastectl.data.dedupe import dhash, hamming, keep_mask
from wastectl.data.prepare import prepare

CLASSES = load_config()["classes"]
# WaRP's 28 class names as listed in the WaRP paper/figure
WARP_NAMES = [
    *[f"bottle-{c}{s}" for c in ("blue", "blue5l", "dark", "green", "milk", "multicolor", "oil", "transp") for s in ("", "-full")],
    "bottle-yogurt", "canister", "cans", "detergent-box", "detergent-color", "detergent-transparent",
    "detergent-white", "glass-dark", "glass-green", "glass-transp", "juice-cardboard", "milk-cardboard",
]
TACO_ANN = Path(__file__).resolve().parents[1] / "data" / "raw" / "taco" / "annotations.json"


@pytest.fixture(scope="module")
def cmap():
    return ClassMap(CLASSES)


def test_warp_names_all_map(cmap):
    assert len(WARP_NAMES) == 28
    got = {n: cmap.name("warp", n) for n in WARP_NAMES}
    assert got["bottle-oil-full"] == "rigid_plastic" and got["cans"] == "metal" and got["glass-dark"] == "other"
    assert got["milk-cardboard"] == "cardboard"


@pytest.mark.skipif(not TACO_ANN.exists(), reason="TACO annotations not downloaded")
def test_every_taco_category_maps(cmap):
    for c in json.loads(TACO_ANN.read_text(encoding="utf-8"))["categories"]:
        cmap.name("taco", c["name"])


def test_trashnet_and_unknown(cmap):
    assert cmap.name("trashnet", "plastic") == "rigid_plastic"
    assert cmap.name("trashnet", "paper") == "other"
    with pytest.raises(KeyError):
        cmap.name("taco", "Mystery object")
    assert cmap.id("taco", "Cigarette") is None and cmap.id("taco", "Unlabeled litter") is None


def test_box_polygon_roundtrip():
    poly = box_to_polygon(0.5, 0.5, 0.2, 0.4)
    assert np.allclose(polygon_box(poly), (0.4, 0.3, 0.6, 0.7))
    assert box_to_polygon(0.05, 0.5, 0.2, 0.2).min() >= 0  # clipped to the image


def test_dedupe_drops_near_duplicates(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    paths = []
    for i, arr in enumerate([a, np.clip(a.astype(int) + 2, 0, 255).astype(np.uint8), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)]):
        p = tmp_path / f"{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    h = [dhash(p) for p in paths]
    assert hamming(h[0], [h[1]])[0] <= 3
    assert keep_mask(h) == [True, False, True]


def _img(path, rng, size=(96, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def _fake_raw(raw, rng):
    d = raw / "warp" / "Warp-Dataset" / "Warp-D"
    d.mkdir(parents=True)
    (d / "classes.txt").write_text("bottle-blue\ncans\nglass-dark\n")
    for split, n in (("train", 10), ("test", 3)):
        for i in range(n):
            _img(d / split / "images" / f"{split}{i}.jpg", rng)
            (d / split / "labels").mkdir(parents=True, exist_ok=True)
            (d / split / "labels" / f"{split}{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n")
    for split in ("train_crops", "test_crops"):
        for i in range(3):
            _img(raw / "warp" / "Warp-Dataset" / "Warp-C" / split / "bottle" / "bottle-blue" / f"{i}.jpg", rng, (40, 40))
    t = raw / "taco"
    ims, anns = [], []
    for i in range(10):
        _img(t / "images" / "batch_1" / f"{i:06d}.jpg", rng, (120, 80))
        ims.append({"id": i, "width": 120, "height": 80, "file_name": f"batch_1/{i:06d}.jpg"})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [10, 10, 60, 50],
                     "segmentation": [[10, 10, 70, 10, 70, 60, 10, 60]]})
    anns.append({"id": 99, "image_id": 0, "category_id": 1, "bbox": [0, 0, 20, 20], "segmentation": [[0, 0, 20, 0, 20, 20]]})
    (t / "annotations.json").write_text(json.dumps({"images": ims, "annotations": anns, "categories": [
        {"id": 0, "name": "Plastic film"}, {"id": 1, "name": "Cigarette"}]}))
    for i in range(10):
        _img(raw / "trashnet" / "dataset-resized" / "metal" / f"metal{i}.jpg", rng, (50, 50))
    junk = raw / "trashnet" / "__MACOSX" / "dataset-resized" / "glass" / "._x.jpg"
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"not an image")


def test_prepare_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "data_raw": str(tmp_path / "raw"), "data_processed": str(tmp_path / "proc"),
                    "experiments": str(tmp_path / "exp")}
    _fake_raw(tmp_path / "raw", rng)
    m = prepare(cfg)
    yolo = tmp_path / "proc" / "yolo"
    test_imgs = list((yolo / "images" / "test").glob("*.jpg"))
    assert test_imgs and all(p.name.startswith("warp_") for p in test_imgs)  # test = WaRP only
    assert all(not p.name.startswith("taco_") for p in test_imgs)
    assert any((yolo / "images" / "train").glob("taco_batch_1_*.jpg"))
    line = (yolo / "labels" / "test" / "warp_test0.txt").read_text().splitlines()[0].split()
    assert line[0] == str(CLASSES.index("rigid_plastic")) and len(line) == 9  # class + 4 corner points
    assert "names:" in (yolo / "data.yaml").read_text()
    crops = tmp_path / "proc" / "crops"
    assert any((crops / "test" / "rigid_plastic").iterdir())  # WaRP-C test crops
    assert not (crops / "test" / "metal").exists()  # TrashNet never in test
    assert not any(crops.glob("*/other/*"))  # __MACOSX junk ignored
    assert any((crops / "train" / "soft_plastic").iterdir())  # TACO object crops
    assert all(len(f.read_text().splitlines()) == 1 for f in (yolo / "labels").glob("*/taco_*.txt"))  # cigarette dropped
    assert m["images"]["test"] == 3 and (tmp_path / "exp" / "data" / "class_counts.csv").exists()


def test_prepare_taco_only_gets_test_split(tmp_path):
    import shutil

    rng = np.random.default_rng(2)
    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "data_raw": str(tmp_path / "raw"), "data_processed": str(tmp_path / "proc"),
                    "experiments": str(tmp_path / "exp")}
    _fake_raw(tmp_path / "raw", rng)
    shutil.rmtree(tmp_path / "raw" / "warp")
    m = prepare(cfg, crops=False)
    assert m["images"] == {"test": 1, "val": 1, "train": 8}  # 10 TACO images -> 80/10/10
    assert not (tmp_path / "proc" / "crops").exists()
    assert "test: images/test" in (tmp_path / "proc" / "yolo" / "data.yaml").read_text()
