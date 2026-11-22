"""Multilayer perceptron with hand-written forward pass, softmax cross-entropy and backprop."""
import numpy as np

ACTS = {
    # name: (f, df/dz expressed via the activation output a)
    "tanh": (np.tanh, lambda a: 1.0 - a**2),
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a: a * (1.0 - a)),
    "relu": (lambda z: np.maximum(z, 0.0), lambda a: (a > 0).astype(a.dtype)),
}


def init_weights(fan_in, fan_out, scheme, rng):
    if scheme == "zero":
        return np.zeros((fan_in, fan_out))
    if scheme == "xavier":
        return rng.normal(0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out))
    if scheme == "he":
        return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
    raise ValueError(f"unknown init {scheme!r}")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, act="tanh", init="xavier", l2=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes, self.act, self.init, self.l2 = list(sizes), act, init, l2
        self.W = [init_weights(a, b, init, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def params(self):
        return self.W + self.b

    def hyperparams(self):
        return {"layer_sizes": self.sizes, "activation": self.act, "init": self.init, "l2": self.l2}

    def n_params(self):
        return sum(p.size for p in self.params())

    def forward(self, X):
        f, _ = ACTS[self.act]
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else f(z))  # last layer = logits
        return acts

    def predict(self, X):
        return self.forward(X)[-1].argmax(axis=1)

    def loss(self, X, y):
        p = softmax(self.forward(X)[-1])
        ce = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        return ce + 0.5 * self.l2 * sum((W**2).sum() for W in self.W)

    def loss_and_grads(self, X, y):
        _, df = ACTS[self.act]
        acts = self.forward(X)
        n = len(y)
        p = softmax(acts[-1])
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean() + 0.5 * self.l2 * sum((W**2).sum() for W in self.W)
        delta = p.copy()
        delta[np.arange(n), y] -= 1.0
        delta /= n
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta + self.l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * df(acts[i])
        return loss, gW + gb
