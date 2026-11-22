"""Waste Analyzer: upload a photo, see every piece of waste outlined and named."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from ui import LABELS, config, footer, legend, page, pipeline

page("Waste Analyzer", "Upload a photo of waste. The model outlines each item and names its material, "
     "so you can see what a sorting line would pick out.")

cfg = config()
pipe, problem = pipeline()
if pipe is None:
    st.markdown(
        f'<div class="empty"><b>The detection model isn\'t trained yet.</b><br>{problem}<br><br>'
        "Train it with <code>python -m wastectl train yolo</code>, then reload this page.</div>",
        unsafe_allow_html=True,
    )
    footer()
    st.stop()

test_dir = Path(cfg["paths"]["data_processed"]) / "yolo" / "images" / "test"
samples = sorted(test_dir.glob("*.jpg"))[:40] if test_dir.exists() else []

left, right = st.columns([1, 2.2], gap="large")
with left:
    upload = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
    sample = st.selectbox("Or try a held-out test photo", ["—"] + [p.name for p in samples]) if samples else "—"
    conf = st.slider("Minimum confidence to show", 0.05, 0.9, float(cfg["purity"]["conf_draw"]), 0.05)
    target = st.selectbox("Stream to check for contamination (optional)", list(LABELS),
                          index=list(LABELS).index(cfg["purity"]["target_stream"]), format_func=LABELS.get)

bgr = None
if upload is not None:
    bgr = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
elif sample != "—":
    bgr = cv2.imread(str(test_dir / sample))

with right:
    if bgr is None:
        st.markdown('<div class="empty">Upload a photo or pick a test photo to start.</div>', unsafe_allow_html=True)
        legend()
        footer()
        st.stop()
    pipe.conf_draw = conf
    with st.spinner("Finding waste…"):
        r = pipe.analyze(bgr, target)
    st.image(cv2.cvtColor(pipe.annotate(bgr, r), cv2.COLOR_BGR2RGB), width="stretch")
    counts = Counter(d["cls"] for d in r["detections"])
    legend(counts)

dets = r["detections"]
with left:
    st.metric("Items found", len(dets))
    p = r["purity"]["purity"]
    st.metric(f"{LABELS[target]} share", "—" if p is None else f"{p:.0%}",
              help="Share of confidently detected items that belong to the chosen stream.")
    st.caption(f"Analysed in {r['latency_s'] * 1000:.0f} ms on CPU")

if dets:
    st.subheader("What it found")
    st.dataframe(
        [{"Material": LABELS[d["cls"]], "Confidence": round(d["conf"], 2),
          "Counts toward share": d["conf"] >= cfg["purity"]["conf_count"]} for d in sorted(dets, key=lambda d: -d["conf"])],
        width="stretch", hide_index=True,
    )
else:
    st.info("Nothing found above this confidence. Lower the slider, or try a closer photo with less clutter.")
footer()
