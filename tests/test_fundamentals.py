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
