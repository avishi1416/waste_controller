import numpy as np
import pytest

from wastectl.fundamentals.data import low_rank_features, spirals
from wastectl.fundamentals.linalg import gradcheck, svd_compress
from wastectl.fundamentals.mlp import MLP
from wastectl.fundamentals.neurons import GATES, mcculloch_pitts, sigmoid, step, tanh
from wastectl.fundamentals.optimizers import SGD, Momentum, Nesterov, train


def test_mcculloch_pitts_gates():
    X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert mcculloch_pitts(X2, *GATES["AND"]).tolist() == [0, 0, 0, 1]
    assert mcculloch_pitts(X2, *GATES["OR"]).tolist() == [0, 1, 1, 1]
    assert mcculloch_pitts(np.array([[0], [1]]), *GATES["NOT"]).tolist() == [1, 0]


def test_activation_ranges():
    z = np.linspace(-10, 10, 101)
    assert set(np.unique(step(z))) == {0.0, 1.0}
    assert (sigmoid(z) > 0).all() and (sigmoid(z) < 1).all()
    assert np.isclose(tanh(0.0), 0.0) and np.isclose(sigmoid(0.0), 0.5)


@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
def test_gradcheck_backprop_matches_numeric(act):
    X, y = spirals(n_per_class=5, seed=1)
    model = MLP([2, 6, 5, 3], act=act, l2=1e-3, seed=1)
    assert gradcheck(model, X, y) < 1e-6


def test_forward_shapes_and_param_count():
    model = MLP([2, 16, 3])
    acts = model.forward(np.zeros((7, 2)))
    assert [a.shape for a in acts] == [(7, 2), (7, 16), (7, 3)]
    assert model.n_params() == 2 * 16 + 16 + 16 * 3 + 3


@pytest.mark.parametrize("opt,bs", [(SGD(0.5), None), (SGD(0.1), 32), (Momentum(0.1), 32), (Nesterov(0.1), 32)])
def test_optimizers_reduce_loss(opt, bs):
    X, y = spirals(seed=0)
    model = MLP([2, 32, 3], seed=0)
    start = model.loss(X, y)
    hist = train(model, opt, X, y, epochs=30, batch_size=bs)
    assert hist["train"][-1] < 0.8 * start


def test_zero_init_cannot_break_symmetry():
    model = MLP([2, 8, 3], init="zero")
    X, y = spirals(seed=0)
    train(model, SGD(0.5), X, y, epochs=5)
    assert np.allclose(model.W[0], model.W[0][:, :1])  # every hidden unit stays identical


def test_svd_compress():
    X = low_rank_features(rank=5)
    Xfull, ratio = svd_compress(X, k=X.shape[1])
    assert np.allclose(Xfull, X)
    assert np.isclose(ratio.sum(), 1.0)
    assert ratio[:5].sum() > 0.95  # five true directions carry the variance


def test_demo_runner_writes_and_merges(tmp_path):
    import json

    from wastectl.fundamentals.demos import run

    run("neurons", tmp_path)
    run("tensors", tmp_path)
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["neurons"]["mcculloch_pitts_truth_tables"]["NOT"] == [1, 0]
    assert "tensors" in res and (tmp_path / "activations.png").exists()


def test_hebbian_explodes_oja_finds_pc1():
    from wastectl.fundamentals.learning_rules import hebbian, oja

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2)) @ np.array([[3.0, 0.0], [0.0, 0.5]])
    X -= X.mean(axis=0)
    _, hn = hebbian(X, lr=0.001, epochs=5)
    w, on = oja(X, lr=0.001, epochs=20)
    assert hn[-1] > 100 * hn[0]  # unbounded growth
    pc1 = np.linalg.svd(X, full_matrices=False)[2][0]
    assert abs(np.linalg.norm(w) - 1) < 0.05 and abs(w @ pc1) > 0.99


def test_competitive_finds_clusters():
    from wastectl.fundamentals.learning_rules import competitive

    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [5, 5], [0, 5]])
    X = np.vstack([c + 0.3 * rng.normal(size=(50, 2)) for c in centres])
    W = competitive(X, k=3)
    assert all(np.min(np.linalg.norm(W - c, axis=1)) < 0.5 for c in centres)


def test_perceptron_converges_on_separable():
    from wastectl.fundamentals.learning_rules import perceptron

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    _, _, errors = perceptron(X, y, epochs=50)
    assert errors[-1] == 0


def test_hopfield_recalls_and_energy_never_rises():
    from wastectl.fundamentals.learning_rules import Hopfield

    rng = np.random.default_rng(0)
    P = rng.choice([-1.0, 1.0], size=(3, 64))
    net = Hopfield(P)
    noisy = P[0].copy()
    noisy[rng.choice(64, 8, replace=False)] *= -1
    s, E = net.recall(noisy)
    assert np.array_equal(s, P[0]) and all(b <= a + 1e-9 for a, b in zip(E, E[1:]))
