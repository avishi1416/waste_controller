"""Contamination & purity: the single source of truth (CLAUDE.md section 5).

Count-based, wrong-material-in-stream: purity = target items / all counted items.
An item counts only at conf >= conf_count; `other` is always a contaminant. Empty input -> purity None (unknown, not pure).
"""
from collections import Counter, deque


def frame_purity(detections, target, conf_count):
    """detections: iterable of {"cls": str, "conf": float}."""
    counted = [d["cls"] for d in detections if d["conf"] >= conf_count]
    n_target = sum(c == target for c in counted)
    return {
        "purity": n_target / len(counted) if counted else None,
        "n_target": n_target,
        "n_total": len(counted),
        "contaminants": dict(Counter(c for c in counted if c != target)),
    }


class StreamMonitor:
    """Sliding-window purity over the last `window` frames; alerts when window purity < alert_below."""

    def __init__(self, target, window=30, alert_below=0.90, conf_count=0.40):
        self.target, self.alert_below, self.conf_count = target, alert_below, conf_count
        self.frames = deque(maxlen=window)

    @classmethod
    def from_config(cls, cfg, target=None):
        p = cfg["purity"]
        return cls(target or p["target_stream"], p["window_frames"], p["alert_below"], p["conf_count"])

    def update(self, detections):
        f = frame_purity(detections, self.target, self.conf_count)
        self.frames.append(f)
        tot = sum(x["n_total"] for x in self.frames)
        window = sum(x["n_target"] for x in self.frames) / tot if tot else None
        return {**f, "window_purity": window, "alert": window is not None and window < self.alert_below}
