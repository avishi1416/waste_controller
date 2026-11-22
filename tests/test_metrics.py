import numpy as np

from wastectl.config import load_config
from wastectl.eval.metrics import confusion_matrix, macro_f1, mask_iou, per_class_f1
from wastectl.explog import log_run, read_runs


def test_confusion_and_f1():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 4)
    assert cm.tolist()[:3] == [[1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0]]
    f1 = per_class_f1(cm)
    assert np.isclose(f1[0], 0.5) and np.isclose(f1[1], 0.8) and f1[2] == 0
    assert np.isclose(macro_f1(cm), (0.5 + 0.8 + 0) / 3)  # class 3 absent from y_true: excluded


def test_perfect_and_iou():
    assert macro_f1(confusion_matrix([0, 1, 2], [0, 1, 2], 3)) == 1.0
    a = np.zeros((4, 4), bool)
    a[:2] = True
    b = np.zeros((4, 4), bool)
    b[1:3] = True
    assert np.isclose(mask_iou(a, b), 4 / 12) and mask_iou(np.zeros(3), np.zeros(3)) == 1.0


def test_explog_appends(tmp_path):
    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "experiments": str(tmp_path)}
    log_run(cfg, "unit", {"f1": np.float64(0.5)}, {"lr": 0.1})
    log_run(cfg, "unit", {"f1": 0.6})
    runs = read_runs(cfg)
    assert [r["metrics"]["f1"] for r in runs] == [0.5, 0.6] and runs[0]["seed"] == 42
