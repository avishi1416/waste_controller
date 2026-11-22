"""Toy datasets for the fundamentals demos.

shortcut: demos run on synthetic data; swap in real crop embeddings once TODO step 1 produces them.
"""
import numpy as np


def spirals(n_per_class=100, classes=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def low_rank_features(n=300, dim=64, rank=5, noise=0.05, seed=0):
    """High-dimensional features with a few true directions, standing in for CNN embeddings."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, rank)) @ rng.normal(size=(rank, dim)) + noise * rng.normal(size=(n, dim))
