"""Gradient checking and SVD utilities."""
import numpy as np


def gradcheck(model, X, y, eps=1e-5):
    """Max relative error between backprop and central-difference gradients over all parameters."""
    _, analytic = model.loss_and_grads(X, y)
    worst = 0.0
    for p, g in zip(model.params(), analytic):
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            lp = model.loss(X, y)
            p[i] = old - eps
            lm = model.loss(X, y)
            p[i] = old
            num[i] = (lp - lm) / (2 * eps)
        denom = np.maximum(np.abs(num) + np.abs(g), 1e-12)
        worst = max(worst, float((np.abs(num - g) / denom).max()))
    return worst


def svd_compress(X, k):
    """Rank-k reconstruction of centred X plus the explained-variance ratio of every component."""
    mu = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Xk = (U[:, :k] * S[:k]) @ Vt[:k] + mu
    return Xk, S**2 / (S**2).sum()
