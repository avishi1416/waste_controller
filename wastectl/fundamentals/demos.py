"""`python -m wastectl fundamentals [name|all]`: runs each demo and saves plots + JSON to experiments/fundamentals/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wastectl.fundamentals.data import low_rank_features, spirals
from wastectl.fundamentals.linalg import gradcheck, svd_compress
from wastectl.fundamentals.mlp import MLP
from wastectl.fundamentals.neurons import GATES, mcculloch_pitts, sigmoid, step, tanh
from wastectl.fundamentals.optimizers import SGD, Momentum, Nesterov, train


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=130)
    plt.close(fig)


def tensors(out, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 4))
    bias = np.arange(4.0)
    col = np.arange(3.0)[:, None]
    res = {
        "creation_shape": a.shape,
        "broadcast_row": (a + bias).shape,      # (3,4) + (4,)
        "broadcast_col": (a * col).shape,       # (3,4) * (3,1)
        "inner_product": float(a[0] @ a[1]),
        "matmul_shape": (a @ a.T).shape,
        "reshape": a.reshape(2, 6).shape,
        "transpose_view": a.T.shape,
    }
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    for axis, (m, t) in zip(ax, [(a, "A (3x4)"), (a + bias, "A + row (4,)"), (a * col, "A * col (3,1)"), (a.reshape(2, 6), "reshape 2x6")]):
        axis.imshow(m, cmap="viridis")
        axis.set_title(t)
        axis.axis("off")
    _save(fig, out, "tensors_broadcasting")
    return {k: list(v) if isinstance(v, tuple) else v for k, v in res.items()}


def neurons(out, seed):
    X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    gates = {g: mcculloch_pitts(X2 if g != "NOT" else np.array([[0], [1]]), *wt).tolist() for g, wt in GATES.items()}
    z = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for f, name in [(step, "step"), (sigmoid, "sigmoid"), (tanh, "tanh")]:
        ax.plot(z, f(z), label=name)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Step vs sigmoid vs tanh")
    ax.legend()
    _save(fig, out, "activations")
    return {"mcculloch_pitts_truth_tables": gates}


def mlp(out, seed):
    X, y = spirals(seed=seed)
    model = MLP([2, 32, 32, 3], act="tanh", seed=seed)
    small = MLP([2, 5, 4, 3], act="tanh", l2=1e-3, seed=seed)
    err = gradcheck(small, *spirals(n_per_class=5, seed=seed))
    hist = train(model, Momentum(0.1), X, y, epochs=300, batch_size=32, seed=seed)
    acc = float((model.predict(X) == y).mean())
    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 200), np.linspace(-1.1, 1.1, 200))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.contourf(xx, yy, zz, alpha=0.3, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="viridis")
    ax.set_title(f"NumPy MLP decision boundary (acc {acc:.2f})")
    _save(fig, out, "mlp_decision_boundary")
    return {
        "gradcheck_max_rel_error": err,
        "train_accuracy": acc,
        "final_loss": float(hist["train"][-1]),
        "parameters": {"count": model.n_params(), "learned": ["W1..W3 weights", "b1..b3 biases"]},
        "hyperparameters": {**model.hyperparams(), "optimizer": "Momentum", "lr": 0.1, "beta": 0.9, "batch_size": 32, "epochs": 300},
    }


def optimizers(out, seed):
    X, y = spirals(seed=seed)
    runs = {
        "Batch GD": (SGD(0.5), None),
        "SGD (bs 32)": (SGD(0.1), 32),
        "Momentum": (Momentum(0.1, 0.9), 32),
        "Nesterov": (Nesterov(0.1, 0.9), 32),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    final = {}
    for name, (opt, bs) in runs.items():
        h = train(MLP([2, 32, 3], seed=seed), opt, X, y, epochs=150, batch_size=bs, seed=seed)
        ax.plot(h["train"], label=name)
        final[name] = float(h["train"][-1])
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Optimizer comparison (same model, same data)")
    ax.legend()
    _save(fig, out, "optimizers")
    return {"final_loss": final}


def svd(out, seed):
    X = low_rank_features(seed=seed)
    _, ratio = svd_compress(X, k=5)
    errs = {k: float(np.linalg.norm(X - svd_compress(X, k)[0]) / np.linalg.norm(X)) for k in (1, 2, 5, 10, 20)}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(1, 21), ratio[:20])
    ax.plot(range(1, 21), np.cumsum(ratio[:20]), "r.-", label="cumulative")
    ax.set_xlabel("singular component")
    ax.set_title("SVD explained variance (64-d features)")
    ax.legend()
    _save(fig, out, "svd_explained_variance")
    return {"explained_variance_top5": float(ratio[:5].sum()), "relative_reconstruction_error": errs}


DEMOS = {"tensors": tensors, "neurons": neurons, "mlp": mlp, "optimizers": optimizers, "svd": svd}


def run(name="all", out_dir="experiments/fundamentals", seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(DEMOS) if name == "all" else [name]
    results = {n: DEMOS[n](out, seed) for n in names}
    path = out / "results.json"
    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}  # single-demo runs keep the others
    path.write_text(json.dumps({**merged, **results}, indent=2), encoding="utf-8")
    return results
