"""Classical learning rules (Unit III): Hebbian, Oja, competitive, error-correction, Hopfield energy. NumPy only."""
import numpy as np


def hebbian(X, lr=0.01, epochs=20, seed=0):
    """Plain Hebb: dw = lr * y * x. No decay term, so |w| grows without bound."""
    w = np.random.default_rng(seed).normal(0, 0.1, X.shape[1])
    norms = []
    for _ in range(epochs):
        for x in X:
            w += lr * (x @ w) * x
        norms.append(float(np.linalg.norm(w)))
    return w, norms


def oja(X, lr=0.01, epochs=20, seed=0):
    """Modified Hebb (Oja): dw = lr * y * (x - y w). Normalises itself and converges to the first principal component."""
    w = np.random.default_rng(seed).normal(0, 0.1, X.shape[1])
    norms = []
    for _ in range(epochs):
        for x in X:
            y = x @ w
            w += lr * y * (x - y * w)
        norms.append(float(np.linalg.norm(w)))
    return w, norms


def competitive(X, k=3, lr=0.1, epochs=20, seed=0):
    """Winner-take-all: only the closest prototype moves toward each input."""
    rng = np.random.default_rng(seed)
    # farthest-point start: random starts can share a cluster and leave a "dead" prototype that never wins
    W = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        W.append(X[np.argmax(np.min([((X - w) ** 2).sum(axis=1) for w in W], axis=0))])
    W = np.array(W, dtype=float)
    for _ in range(epochs):
        for x in X[rng.permutation(len(X))]:
            j = np.argmin(((W - x) ** 2).sum(axis=1))
            W[j] += lr * (x - W[j])
    return W


def perceptron(X, y, lr=0.1, epochs=20):
    """Error-correction: weights change only when the prediction is wrong. y in {0,1}."""
    w, b, errors = np.zeros(X.shape[1]), 0.0, []
    for _ in range(epochs):
        e = 0
        for x, t in zip(X, y):
            err = t - int(x @ w + b >= 0)
            if err:
                w += lr * err * x
                b += lr * err
                e += 1
        errors.append(e)
    return w, b, errors


class Hopfield:
    """Energy-based associative memory: Hebbian storage, asynchronous updates never increase the energy."""

    def __init__(self, patterns):
        P = np.asarray(patterns, dtype=float)  # rows in {-1, +1}
        self.W = P.T @ P / P.shape[1]
        np.fill_diagonal(self.W, 0)

    def energy(self, s):
        return float(-0.5 * s @ self.W @ s)

    def recall(self, s, sweeps=5, seed=0):
        s = np.asarray(s, dtype=float).copy()
        rng = np.random.default_rng(seed)
        energies = [self.energy(s)]
        for _ in range(sweeps):
            for i in rng.permutation(len(s)):
                s[i] = 1.0 if self.W[i] @ s >= 0 else -1.0
                energies.append(self.energy(s))
        return s, energies
