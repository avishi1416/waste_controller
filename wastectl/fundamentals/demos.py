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


def learning_rules(out, seed):
    from wastectl.fundamentals.learning_rules import Hopfield, competitive, hebbian, oja, perceptron

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 2)) @ np.array([[3.0, 1.0], [1.0, 1.0]])
    X -= X.mean(axis=0)
    _, hn = hebbian(X, lr=0.001, epochs=10, seed=seed)
    w_oja, on = oja(X, lr=0.001, epochs=20, seed=seed)
    pc1 = np.linalg.svd(X, full_matrices=False)[2][0]
    blobs = np.vstack([c + 0.4 * rng.normal(size=(60, 2)) for c in ([0, 0], [4, 4], [0, 4])])
    protos = competitive(blobs, k=3, seed=seed)
    Xs = rng.normal(size=(200, 2))
    ys = (Xs @ [1.0, -0.5] > 0).astype(int)
    _, _, errs = perceptron(Xs, ys, epochs=30)
    P = rng.choice([-1.0, 1.0], size=(3, 100))
    noisy = P[0].copy()
    noisy[rng.choice(100, 15, replace=False)] *= -1
    rec, energy = Hopfield(P).recall(noisy, seed=seed)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    ax[0].semilogy(hn, label="Hebb |w|")
    ax[0].semilogy(on, label="Oja |w|")
    ax[0].set_title("Hebb diverges, Oja normalises")
    ax[0].legend()
    ax[1].scatter(blobs[:, 0], blobs[:, 1], s=6, c="grey")
    ax[1].scatter(protos[:, 0], protos[:, 1], s=120, c="red", marker="*")
    ax[1].set_title("Competitive (winner-take-all) prototypes")
    ax[2].plot(errs, marker=".")
    ax[2].set_title("Perceptron (error-correction): mistakes/epoch")
    ax[3].plot(energy)
    ax[3].set_title("Hopfield energy during recall")
    _save(fig, out, "learning_rules")
    return {
        "hebbian_weight_norm_growth": hn[-1] / hn[0],
        "oja_weight_norm": on[-1],
        "oja_vs_svd_pc1_cosine": float(abs(w_oja @ pc1) / np.linalg.norm(w_oja)),
        "perceptron_errors_first_last": [errs[0], errs[-1]],
        "hopfield_recall_exact": bool(np.array_equal(rec, P[0])),
        "hopfield_energy_start_end": [energy[0], energy[-1]],
        "comparison": (
            "Hebbian/Oja/competitive are unsupervised: weights follow input correlations or proximity, with no target. "
            "Plain Hebb grows without bound; Oja's decay term makes it converge to the first principal component. "
            "Error-correction (perceptron, backprop) is supervised: weights move only in proportion to the prediction error. "
            "Hopfield stores patterns with a Hebbian rule and recalls them by descending an energy function (Boltzmann-style, deterministic)."
        ),
    }


def ablations(out, seed):
    """Init, input normalisation, L2 and augmentation on a deliberately small training set (to expose overfitting)."""
    Xtr, ytr = spirals(n_per_class=15, noise=0.4, seed=seed)
    Xva, yva = spirals(n_per_class=200, noise=0.4, seed=seed + 1)

    def run(init="xavier", l2=0.0, scale=1.0, shift=0.0, augment=0, epochs=600):
        rng = np.random.default_rng(seed)
        X, y = Xtr * scale + shift, ytr
        if augment:  # Gaussian jitter copies of each training point
            X = np.vstack([X] + [X + rng.normal(0, 0.05 * scale, X.shape) for _ in range(augment)])
            y = np.concatenate([y] * (augment + 1))
        m = MLP([2, 64, 64, 3], act="tanh", init=init, l2=l2, seed=seed)
        h = train(m, Momentum(0.05), X, y, epochs=epochs, batch_size=16, seed=seed, X_val=Xva * scale + shift, y_val=yva)
        acc = float((m.predict(Xva * scale + shift) == yva).mean())
        return h, {"train_loss": float(h["train"][-1]), "val_loss": float(h["val"][-1]), "val_acc": acc}

    variants = {
        "baseline (xavier, no reg)": {},
        "init: zero": {"init": "zero"},
        "init: he": {"init": "he"},
        "unnormalised input (x*10+5)": {"scale": 10.0, "shift": 5.0},
        "L2 1e-3": {"l2": 1e-3},
        "L2 1e-2": {"l2": 1e-2},
        "augment x4 jitter": {"augment": 4},
    }
    results, hists = {}, {}
    for name, kw in variants.items():
        hists[name], results[name] = run(**kw)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("baseline (xavier, no reg)", "L2 1e-2", "augment x4 jitter"):
        ax[0].plot(hists[name]["train"], "--", label=f"{name} train")
        ax[0].plot(hists[name]["val"], label=f"{name} val")
    ax[0].set_title("Overfitting vs regularisation (small train set)")
    ax[0].set_xlabel("epoch")
    ax[0].legend(fontsize=7)
    names = list(results)
    ax[1].barh(names, [results[n]["val_acc"] for n in names])
    ax[1].set_xlabel("validation accuracy")
    ax[1].set_title("Ablations")
    _save(fig, out, "ablations")
    return results


DEMOS = {"tensors": tensors, "neurons": neurons, "mlp": mlp, "optimizers": optimizers, "svd": svd,
         "learning_rules": learning_rules, "ablations": ablations}


def run(name="all", out_dir="experiments/fundamentals", seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(DEMOS) if name == "all" else [name]
    results = {n: DEMOS[n](out, seed) for n in names}
    path = out / "results.json"
    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}  # single-demo runs keep the others
    path.write_text(json.dumps({**merged, **results}, indent=2), encoding="utf-8")
    return results
