"""GD/SGD, Momentum and Nesterov, updating parameter arrays in place."""
import numpy as np


class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    def __init__(self, lr=0.1, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.beta
            v -= self.lr * g
            p += v


class Nesterov(Momentum):
    """Sutskever form: look-ahead folded into the update, so grads are taken at the current params."""

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v_prev = v.copy()
            v *= self.beta
            v -= self.lr * g
            p += -self.beta * v_prev + (1 + self.beta) * v


def train(model, opt, X, y, epochs=100, batch_size=None, seed=0, X_val=None, y_val=None):
    """batch_size=None -> full-batch gradient descent; otherwise mini-batch SGD. Returns loss history."""
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = batch_size or n
    hist = {"train": [], "val": []}
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            b = idx[s : s + bs]
            _, grads = model.loss_and_grads(X[b], y[b])
            opt.step(model.params(), grads)
        hist["train"].append(model.loss(X, y))
        if X_val is not None:
            hist["val"].append(model.loss(X_val, y_val))
    return hist
