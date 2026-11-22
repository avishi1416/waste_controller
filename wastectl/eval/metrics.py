"""Classification metrics in NumPy (no sklearn dependency). Detection mAP/IoU come from Ultralytics/COCO eval."""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm  # rows = true, cols = predicted


def per_class_f1(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    rec = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)


def macro_f1(cm):
    """Mean F1 over classes that appear in y_true (absent classes would otherwise count as 0)."""
    present = cm.sum(axis=1) > 0
    return float(per_class_f1(cm)[present].mean()) if present.any() else 0.0


def mask_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0
