from wastectl.config import load_config
from wastectl.purity import StreamMonitor, frame_purity


def d(cls, conf=0.9):
    return {"cls": cls, "conf": conf}


def test_frame_purity_counts_and_contaminants():
    r = frame_purity([d("rigid_plastic"), d("rigid_plastic"), d("metal"), d("other")], "rigid_plastic", 0.4)
    assert r["purity"] == 0.5 and r["n_total"] == 4 and r["contaminants"] == {"metal": 1, "other": 1}


def test_low_confidence_ignored_and_empty_is_unknown():
    assert frame_purity([d("metal", 0.3), d("rigid_plastic")], "rigid_plastic", 0.4)["purity"] == 1.0
    assert frame_purity([], "rigid_plastic", 0.4)["purity"] is None


def test_stream_window_and_alert():
    m = StreamMonitor("rigid_plastic", window=3, alert_below=0.9, conf_count=0.4)
    assert m.update([d("rigid_plastic")] * 9)["alert"] is False
    r = m.update([d("metal")] * 3)  # window: 9 target / 12 total = 0.75
    assert abs(r["window_purity"] - 0.75) < 1e-9 and r["alert"] is True
    m.update([])  # empty frame: no items, doesn't change the ratio
    r = m.update([d("rigid_plastic")] * 10)  # first frame (9 target) slides out: 10 / 13
    assert abs(r["window_purity"] - 10 / 13) < 1e-9


def test_empty_window_never_alerts():
    m = StreamMonitor.from_config(load_config())
    r = m.update([])
    assert r["window_purity"] is None and r["alert"] is False
    assert m.target == "rigid_plastic" and m.frames.maxlen == 30
