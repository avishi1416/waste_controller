"""Evaluation: test-set metrics and plots from the latest logged runs."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import LABELS, config, footer, page  # noqa: E402

from wastectl.explog import read_runs  # noqa: E402

page("Evaluation", "How well the model does on photos it never saw during training (the held-out test split).")
cfg = config()
runs = read_runs(cfg)
ev = next((r for r in reversed(runs) if r["kind"] == "eval-yolo" and r["metrics"].get("split") == "test"), None)
bench = next((r for r in reversed(runs) if r["kind"] == "bench"), None)

if ev is None:
    st.markdown('<div class="empty">No test-set evaluation yet. Run <code>python -m wastectl eval yolo</code> '
                "after training.</div>", unsafe_allow_html=True)
else:
    m = ev["metrics"]
    c = st.columns(4)
    c[0].metric("Mask mAP@0.5", f"{m.get('mask_map50', 0):.3f}", help="Segmentation quality: outline overlap ≥ 50%")
    c[1].metric("Mask mAP@0.5:0.95", f"{m.get('mask_map50_95', 0):.3f}")
    c[2].metric("Box mAP@0.5", f"{m['box_map50']:.3f}", help="Detection quality: box overlap ≥ 50%")
    c[3].metric("Speed (CPU)", f"{bench['metrics']['fps']:.1f} img/s" if bench else "—")
    st.subheader("Per material (box AP@0.5:0.95)")
    st.bar_chart({LABELS.get(k, k): v for k, v in m["per_class_box_ap50_95"].items()}, horizontal=True)
    st.caption(f"Run {ev['run_id']} · commit {ev.get('git')} · seed {ev['seed']}")

plots = Path(cfg["paths"]["experiments"]) / "runs" / "val_test"
shown = [(f, t) for f, t in [("confusion_matrix_normalized.png", "Confusion matrix (normalised)"),
                             ("MaskPR_curve.png", "Precision-recall, masks"), ("BoxPR_curve.png", "Precision-recall, boxes")]
         if (plots / f).exists()]
if shown:
    st.subheader("Plots")
    cols = st.columns(len(shown))
    for col, (f, t) in zip(cols, shown):
        col.image(str(plots / f), caption=t, width="stretch")
footer()
