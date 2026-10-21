import numpy as np
import pytest

from agent.archive import Archive
from agent.warmstart import METHODS, LOCAL_METHODS, make_optimizer
from src.optimizers import OptimizerExhausted

D = 5
BOUNDS = np.array([[-5.0, 5.0]] * D)
SHIFT = np.array([1.5, -2.0, 0.5, 3.0, -1.0])


def f(x):  # shifted, mildly ill-conditioned quadratic
    return float(np.sum((np.arange(1, D + 1)) * (np.asarray(x) - SHIFT) ** 2))


def run(opt, archive, n, tag):
    for _ in range(n):
        try:
            X = opt.ask(1)
        except OptimizerExhausted:
            break
        y = f(X[0])
        opt.tell(X, [y])
        archive.add(X[0], y, tag)


def seeded_archive(n=60, seed=0):
    rng = np.random.default_rng(seed)
    a = Archive(BOUNDS)
    for x in rng.uniform(-5, 5, size=(n, D)):
        a.add(x, f(x), "seed")
    return a


@pytest.mark.parametrize("method", sorted(METHODS))
def test_cold_start_runs(method):
    a = Archive(BOUNDS)
    opt = make_optimizer(method, BOUNDS, a, start="center", seed=0)
    run(opt, a, 25 if method in ("gp_bo", "turbo") else 60, method)
    opt.close()
    assert len(a) > 0
    assert set(opt.state()) >= {"n_told", "exhausted"}


@pytest.mark.parametrize("method", sorted(METHODS))
def test_warm_start_never_loses_incumbent(method):
    a = seeded_archive()
    _, y_inc = a.incumbent()
    opt = make_optimizer(method, BOUNDS, a, seed=1)
    run(opt, a, 25 if method in ("gp_bo", "turbo") else 60, method)
    opt.close()
    assert a.incumbent()[1] <= y_inc


@pytest.mark.parametrize("method", sorted(LOCAL_METHODS))
def test_local_solvers_start_at_incumbent_and_improve(method):
    a = seeded_archive()
    x_inc, y_inc = a.incumbent()
    opt = make_optimizer(method, BOUNDS, a, seed=0)
    X = opt.ask(1)
    assert np.allclose(X[0], x_inc)
    opt.tell(X, [f(X[0])])
    a.add(X[0], f(X[0]), method)
    run(opt, a, 150, method)
    opt.close()
    assert a.incumbent()[1] < 0.5 * y_inc


def test_cma_first_generation_near_incumbent():
    a = seeded_archive()
    x_inc, _ = a.incumbent()
    scale = a.local_scale()
    opt = make_optimizer("cma_es", BOUNDS, a, seed=0)
    X = opt.ask(opt._es.popsize)
    dist = np.linalg.norm(X - x_inc, axis=1)
    assert np.median(dist) < 4 * np.median(scale) * np.sqrt(D)


def test_close_stops_solver_threads():
    a = seeded_archive()
    opts = [make_optimizer(m, BOUNDS, a) for m in sorted(LOCAL_METHODS)]
    for o in opts:
        o.ask(1)  # leave each mid-evaluation
        o.close()
    for o in opts:
        o._thread.join(timeout=5)
        assert not o._thread.is_alive()


def test_archive_helpers():
    a = seeded_archive(100)
    S = a.diverse_starts(3)
    assert S.shape == (3, D)
    assert np.allclose(S[0], a.incumbent()[0])
    s = a.local_scale()
    assert np.all(s > 0) and np.all(s <= 0.3 * 10 + 1e-12)
    assert a.evals_since_improvement() >= 0
