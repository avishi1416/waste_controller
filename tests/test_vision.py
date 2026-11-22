import numpy as np
from PIL import Image

from conftest import needs_torch
from wastectl.vision.classifier import samples

CLASSES = ["cardboard", "soft_plastic", "rigid_plastic", "metal", "other"]


def test_samples_use_fixed_class_order(tmp_path):
    for c in ("metal", "cardboard"):  # soft/rigid/other folders missing on purpose
        (tmp_path / "test" / c).mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "test" / c / "a.jpg")
    (tmp_path / "test" / "metal" / "notes.txt").write_text("x")
    got = sorted((p.parent.name, y) for p, y in samples(tmp_path, "test", CLASSES))
    assert got == [("cardboard", 0), ("metal", 3)]  # ids from config order, not folder order


@needs_torch
def test_classifier_heads_output_n_classes():
    import torch

    from wastectl.vision.classifier import build

    for b in ("resnet18", "googlenet"):
        m = build(b, 5, pretrained=False).train()
        assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 5)
