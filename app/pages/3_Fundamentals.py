"""Neural-network fundamentals built from scratch in NumPy (no deep-learning library)."""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import config, footer, page  # noqa: E402

page("Fundamentals", "The building blocks behind the detector, written from scratch in NumPy and tested: "
     "neurons, backpropagation, optimizers, classical learning rules and regularisation.")
out = Path(config()["paths"]["experiments"]) / "fundamentals"
res = json.loads((out / "results.json").read_text(encoding="utf-8")) if (out / "results.json").exists() else {}

SECTIONS = [
    ("Activations and neurons", "activations.png",
     "Step, sigmoid and tanh responses. McCulloch-Pitts threshold units implement AND, OR and NOT."),
    ("Tensors and broadcasting", "tensors_broadcasting.png", "Row and column broadcasting, and reshaping a 3×4 tensor."),
    ("MLP with hand-written backpropagation", "mlp_decision_boundary.png",
     lambda r: f"Gradient check error {r['mlp']['gradcheck_max_rel_error']:.1e}; training accuracy {r['mlp']['train_accuracy']:.0%} on three spirals."),
    ("Optimizers", "optimizers.png", "Same model and data: momentum methods escape the plateau where plain GD and SGD stall."),
    ("SVD", "svd_explained_variance.png", "Five singular components carry almost all the variance of 64-dimensional features."),
    ("Learning rules", "learning_rules.png",
     "Hebb diverges and Oja converges to the first principal component; winner-take-all finds clusters; "
     "the perceptron corrects errors; a Hopfield network recalls a stored pattern by lowering its energy."),
    ("Regularisation ablations", "ablations.png",
     "On a small training set the baseline overfits; augmentation and He initialisation generalise best, "
     "zero initialisation cannot learn, and heavy L2 underfits."),
]
for title, img, text in SECTIONS:
    if not (out / img).exists():
        continue
    st.subheader(title)
    st.image(str(out / img), width="stretch")
    try:
        st.caption(text(res) if callable(text) else text)
    except KeyError:
        pass
if not res:
    st.markdown('<div class="empty">Run <code>python -m wastectl fundamentals</code> to generate these plots.</div>',
                unsafe_allow_html=True)
footer()
