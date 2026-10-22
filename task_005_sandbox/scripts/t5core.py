"""Task 005 campaign core: feasibility, discrete snapping, evaluation wrapper,
Phase A (feasible Sobol) and Phase B (feasibility-aware TuRBO-1).

Exec'd into the persistent kernel namespace; state objects (EV, TR) are only
created once so re-exec'ing this file after edits keeps the campaign state.
"""
import math
import os
import threading
import time
import traceback
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

warnings.filterwarnings("ignore")

SP = "/tmp/claude-1000/-home-kevzhu14-Documents-task-005-sandbox/f418fad7-eaae-41a9-a8ac-f3a2e53bb715/scratchpad"
RES = "/home/kevzhu14/Documents/task_005_sandbox/results/agent_task_005"
LOG = os.path.join(SP, "progress.log")
CHANGELOG = os.path.join(RES, "changelog.md")
D = 11

A = np.zeros((6, 11))
A[0, [1, 4]] = [-0.80, 0.86]
A[1, [0, 3]] = [-0.54, 0.83]
A[2, [0, 4]] = [0.54, -0.86]
A[3, [0, 4]] = [-0.54, 0.86]
A[4, [0, 1, 2]] = [0.54, -0.80, 0.50]
A[5, [3, 4, 5]] = [0.83, -0.86, 0.96]
b = np.array([0.63, 0.62, 0.37, 0.54, 0.50, 0.81])

L9 = np.array([0.0, 0.166667, 1.0])
L10 = np.array([0.0, 0.075697, 0.167331, 0.187251, 0.219124, 0.227092, 0.239044,
                0.286853, 0.298805, 0.418327, 0.657371, 0.697211, 0.984064, 1.0])
CENTER = np.full(D, 0.5)  # strictly feasible interior point


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def changelog(n_eval, change, reason):
    with open(CHANGELOG, "a") as f:
        f.write(f"## Evaluation {n_eval}\n**Change:** {change}\n**Reason:** {reason}\n\n")


def allowed(x):
    """Brief's own definition (total violation <= 1e-10)."""
    return np.maximum(A @ np.asarray(x) - b, 0).sum() <= 1e-10


def feasible_rows(X):
    """Stricter vectorized check used for generating candidates (no violation at all)."""
    X = np.atleast_2d(X)
    return (X @ A.T - b).max(axis=1) <= 0.0


def snap_levels(v, levels):
    # argmin returns the first index on ties, i.e. the lower level, as the brief specifies
    return levels[np.argmin(np.abs(levels[None, :] - np.atleast_1d(v)[:, None]), axis=1)]


def snap(X):
    X = np.array(X, dtype=float, copy=True)
    one = X.ndim == 1
    X = np.atleast_2d(X)
    X[:, 9] = snap_levels(X[:, 9], L9)
    X[:, 10] = snap_levels(X[:, 10], L10)
    return X[0] if one else X


def project(x):
    """Euclidean projection of x onto {box} ∩ {A x <= b}; falls back to bisection
    toward the interior CENTER so the result is always strictly allowed."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if feasible_rows(x)[0]:
        return x
    res = minimize(lambda z: 0.5 * np.sum((z - x) ** 2), x, jac=lambda z: z - x,
                   method="SLSQP", bounds=[(0.0, 1.0)] * D,
                   constraints=[{"type": "ineq", "fun": lambda z: b - 1e-12 - A @ z,
                                 "jac": lambda z: -A}],
                   options={"ftol": 1e-15, "maxiter": 300})
    z = np.clip(res.x, 0.0, 1.0)
    z[6:] = x[6:]
    if feasible_rows(z)[0]:
        return z
    lo, hi = 0.0, 1.0  # t=0 -> CENTER (feasible), t=1 -> z
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if feasible_rows(CENTER + mid * (z - CENTER))[0]:
            lo = mid
        else:
            hi = mid
    out = CENTER + lo * (z - CENTER)
    out[6:] = x[6:]
    return out


class BudgetDone(Exception):
    pass


class Evaluator:
    """Single gateway to harness.evaluate: snaps discrete coords, refuses
    infeasible points, caches exact repeats, tracks the best ALLOWED value."""

    def __init__(self, harness, budget=2000):
        self.h = harness
        self.budget = budget
        self.X, self.Y, self.lab = [], [], []
        self.cache = {}
        self.best_y, self.best_x, self.best_i = np.inf, None, -1
        self.label = None
        self.n_infeasible = 0

    @property
    def n(self):
        return len(self.Y)

    def remaining(self):
        return self.budget - self.n

    def set_label(self, name):
        if name != self.label:
            self.h.set_active_optimizer(name)
            self.label = name

    def __call__(self, x, label):
        x = snap(np.clip(np.asarray(x, dtype=float), 0.0, 1.0))
        if not allowed(x):
            raise ValueError("refusing to evaluate a point that is not allowed")
        key = x.tobytes()
        if key in self.cache:
            return self.cache[key]
        if self.remaining() <= 0:
            raise BudgetDone()
        self.set_label(label)
        y = float(self.h.evaluate(x.tolist()))
        self.cache[key] = y
        self.X.append(x)
        self.Y.append(y)
        self.lab.append(label)
        if not allowed(x):
            self.n_infeasible += 1
        elif y < self.best_y:
            self.best_y, self.best_x, self.best_i = y, x.copy(), self.n
        return y

    def arrays(self):
        return np.array(self.X), np.array(self.Y)


# ----------------------------------------------------------------- Phase A
def feasible_sobol(n, seed):
    """n allowed points from scrambled 11-D Sobol; dims 9/10 mapped to levels
    by equal-probability bins (=> stratified level counts)."""
    sob = qmc.Sobol(d=D, scramble=True, seed=seed)
    out = np.empty((0, D))
    while len(out) < n:
        U = sob.random(256)
        U[:, 9] = L9[np.minimum((U[:, 9] * 3).astype(int), 2)]
        U[:, 10] = L10[np.minimum((U[:, 10] * 14).astype(int), 13)]
        out = np.vstack([out, U[feasible_rows(U)]])
    return out[:n]


# ----------------------------------------------------------------- Phase B
def y_transform(y, mode):
    y = np.asarray(y, dtype=float)
    if mode == "logshift":
        q75, q25 = np.percentile(y, [75, 25])
        delta = 0.1 * max(q75 - q25, 1e-12)
        return np.log(y - y.min() + delta)
    return y


def fit_gp(Xt, yt):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X_t = torch.tensor(Xt, dtype=torch.double)
    Y_t = torch.tensor(yt, dtype=torch.double).unsqueeze(-1)
    covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D,
                                     lengthscale_constraint=Interval(0.005, 4.0)))
    lik = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    model = SingleTaskGP(X_t, Y_t, covar_module=covar, likelihood=lik,
                         outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:  # keep going with the prior-initialised hyperparameters
        log(f"GP fit warning: {e!r}")
    return model


class TurboState:
    def __init__(self):
        self.length_init, self.length_min, self.length_max = 0.8, 0.5 ** 7, 1.6
        self.batch = 4
        self.succ_tol = 3
        self.fail_tol = math.ceil(max(4.0 / self.batch, D / self.batch))  # = 3
        self.n_restart_design = 2 * D
        self.n_cand = 2500
        self.p_cat = 0.2
        self.ymode = "standardize"
        self.max_train = None  # compute guard (criterion 5)
        self.reset(local_idx=[])
        self.restarts = 0
        self.history = []  # per-batch diagnostics

    def reset(self, local_idx):
        self.length = self.length_init
        self.succ = 0
        self.fail = 0
        self.local_idx = list(local_idx)  # indices into EV.X/EV.Y
        self.local_best = np.inf


def turbo_propose(TR, rng):
    import torch
    from botorch.generation.sampling import MaxPosteriorSampling

    Xall, Yall = EV.arrays()
    Xl, yl = Xall[TR.local_idx], Yall[TR.local_idx]
    fix10 = getattr(TR, "fix10", None)
    if fix10 is not None:  # arm mode: centre = best local point on the fixed x10 level
        sel = np.nonzero(Xl[:, 10] == fix10)[0]
        ib = int(sel[np.argmin(yl[sel])])
    else:
        ib = int(np.argmin(yl))
    center = Xl[ib]
    if TR.max_train is not None and len(yl) > TR.max_train:
        keep = np.argsort(np.linalg.norm(Xl - center, axis=1))[:TR.max_train]
        Xl, yl = Xl[keep], yl[keep]
    yt = -y_transform(yl, TR.ymode)  # sampler maximizes
    model = fit_gp(Xl, yt)

    ls = model.covar_module.base_kernel.lengthscale.squeeze().detach().numpy()
    w = ls / ls.mean()
    w = w / np.prod(np.power(w, 1.0 / len(w)))
    lb = np.clip(center - w * TR.length / 2.0, 0.0, 1.0)
    ub = np.clip(center + w * TR.length / 2.0, 0.0, 1.0)

    prob = min(20.0 / D, 1.0)
    cands = np.empty((0, D))
    tries = 0
    while len(cands) < TR.n_cand and tries < 20:
        tries += 1
        m = 3 * TR.n_cand
        sob = qmc.Sobol(d=D, scramble=True, seed=int(rng.integers(1 << 30)))
        pert = lb + (ub - lb) * sob.random(m)
        mask = rng.random((m, D)) <= prob
        none = ~mask.any(axis=1)
        mask[none, rng.integers(0, D, none.sum())] = True
        C = np.tile(center, (m, 1))
        C[mask] = pert[mask]
        # categorical mutations so the discrete coords are not locked to the center's level
        mut9 = rng.random(m) < getattr(TR, "p_cat9", TR.p_cat)
        C[mut9, 9] = rng.choice(L9, mut9.sum())
        mut10 = rng.random(m) < getattr(TR, "p_cat10", TR.p_cat)
        C[mut10, 10] = rng.choice(L10, mut10.sum())
        if fix10 is not None:
            C[:, 10] = fix10
        C = snap(C)
        C = C[feasible_rows(C)]
        # drop exact repeats of already-evaluated points
        if len(C):
            C = C[np.array([c.tobytes() not in EV.cache for c in C], dtype=bool)]
        cands = np.vstack([cands, C])
    cands = np.unique(cands[:TR.n_cand], axis=0)
    with torch.no_grad():
        ts = MaxPosteriorSampling(model=model, replacement=False)
        Xn = ts(torch.tensor(cands, dtype=torch.double), num_samples=TR.batch).numpy()
    return Xn, ls


def turbo_update(TR, ybatch):
    thr = TR.succ_abs if getattr(TR, "succ_abs", None) is not None else 1e-3 * abs(TR.local_best)
    improved = ybatch.min() < TR.local_best - thr
    if improved:
        TR.succ += 1
        TR.fail = 0
    else:
        TR.succ = 0
        TR.fail += 1
    TR.local_best = min(TR.local_best, float(ybatch.min()))
    if TR.succ == TR.succ_tol:
        TR.length = min(2.0 * TR.length, TR.length_max)
        TR.succ = 0
    elif TR.fail == TR.fail_tol:
        TR.length /= 2.0
        TR.fail = 0


def global_best_trace():
    _, Y = EV.arrays()
    return np.minimum.accumulate(Y)


def run_turbo(stop_at, seed=1, label="turbo_feasible"):
    """Background-thread loop. Stops at global eval count `stop_at`, or on the
    pre-registered stagnation criterion (#2), or if STOP flag is set."""
    global TURBO_STATUS
    rng = np.random.default_rng(seed)
    TURBO_STATUS = "running"
    start_n = EV.n
    try:
        while EV.n < stop_at and not STOP.is_set():
            # ---- restart handling (criterion 4)
            if TR.length < TR.length_min and getattr(TR, "on_collapse", "restart") == "stop":
                TURBO_STATUS = f"collapsed at n={EV.n}"
                log(TURBO_STATUS)
                return
            if TR.length < TR.length_min or len(TR.local_idx) == 0:
                if len(TR.local_idx):
                    old_best = TR.local_best
                    TR.restarts += 1
                    n0 = EV.n
                    TR.reset(local_idx=[])
                    changelog(n0, f"TuRBO restart #{TR.restarts}: trust region discarded, fresh "
                                  f"{TR.n_restart_design}-point allowed Sobol design "
                                  f"(global history kept).",
                              f"Trust-region side length fell below 0.5^7 after repeated "
                              f"non-improving batches; that region's best was y={old_best:.6g} "
                              f"while the global best allowed value is y={EV.best_y:.6g} "
                              f"(pre-registered criterion 4).")
                    log(f"restart #{TR.restarts} at n={n0}")
                design = feasible_sobol(TR.n_restart_design, int(rng.integers(1 << 30)))
                for x in design:
                    if EV.n >= stop_at:
                        break
                    EV(x, label)
                    TR.local_idx.append(EV.n - 1)
                TR.local_best = float(np.min(np.array(EV.Y)[TR.local_idx]))
                continue
            t0 = time.time()
            Xn, ls = turbo_propose(TR, rng)
            dt = time.time() - t0
            if dt > 30 and TR.max_train is None:
                TR.max_train = 500
                changelog(EV.n, "Capped the TuRBO GP training set to the 500 points nearest "
                                "the trust-region center.",
                          f"One GP fit + Thompson proposal took {dt:.1f} s (> 30 s threshold, "
                          f"pre-registered criterion 5) with {len(TR.local_idx)} local points.")
            ys = []
            for x in Xn:
                if EV.n >= stop_at:
                    break
                ys.append(EV(x, label))
                TR.local_idx.append(EV.n - 1)
            if not ys:
                break
            turbo_update(TR, np.array(ys))
            TR.history.append(dict(n=EV.n, length=TR.length, local_best=TR.local_best,
                                   gbest=EV.best_y, dt=dt, nloc=len(TR.local_idx),
                                   ls=np.round(ls, 3).tolist()))
            log(f"n={EV.n} L={TR.length:.4f} loc_best={TR.local_best:.6g} "
                f"gbest={EV.best_y:.6g} nloc={len(TR.local_idx)} dt={dt:.1f}s "
                f"batch={np.round(ys, 5).tolist()}")
            # ---- stagnation check (criterion 2)
            nB = EV.n - start_n
            if nB > 300 and TR.restarts >= 1:
                gb = global_best_trace()
                prev = gb[EV.n - 301]
                if prev - gb[-1] <= 1e-6 * max(1.0, abs(gb[-1])):
                    TURBO_STATUS = f"stagnated at n={EV.n}"
                    log(TURBO_STATUS)
                    return
        TURBO_STATUS = f"done at n={EV.n}"
    except BudgetDone:
        TURBO_STATUS = "budget exhausted"
    except Exception:
        TURBO_STATUS = "error: " + traceback.format_exc()
        log(TURBO_STATUS)


def start_bg(fn, *a, **k):
    STOP.clear()
    t = threading.Thread(target=fn, args=a, kwargs=k, daemon=True)
    t.start()
    return t


if "STOP" not in globals():
    STOP = threading.Event()
    TURBO_STATUS = "idle"


def phase_a_design(n, seed):
    """Sobol (filtered to the allowed polytope) for x0..x8; x9/x10 from exactly
    balanced, randomly shuffled level lists (roadmap Phase A)."""
    rng = np.random.default_rng(seed)
    X = feasible_sobol(n, seed)
    X[:, 9] = rng.permutation(np.resize(L9, n))
    X[:, 10] = rng.permutation(np.resize(L10, n))
    assert feasible_rows(X).all()
    return X


# ----------------------------------------------------------------- Phase C2: local CMA-ES
def run_cma(x_start, sigma0, max_evals, label="cma_local", seed=1, popsize=None, tag="cma"):
    """CMA-ES over x0..x8 with x9/x10 fixed at x_start's levels. Infeasible samples are
    projected and the projected point is evaluated; CMA is told f(proj) + 100*||x-proj||.
    Stops on max_evals, sigma < 1e-5, or 150 evals without improvement > 1e-9*max(1,|best|)
    (criterion 7). Results go to CMA_RUNS[tag]."""
    import cma
    global CMA_RUNS
    st = dict(status="running", n0=EV.n, best_y=np.inf, best_x=None, hist=[])
    CMA_RUNS[tag] = st
    try:
        opts = {"bounds": [[0.0] * 9, [1.0] * 9], "verbose": -9, "seed": seed,
                "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 10**9, "tolx": 1e-12,
                "tolstagnation": 10**9, "maxfevals": 10**9}
        if popsize:
            opts["popsize"] = popsize
        es = cma.CMAEvolutionStrategy(np.asarray(x_start[:9], float), sigma0, opts)
        used, last_imp = 0, 0
        while used < max_evals and not STOP.is_set():
            sols = es.ask()
            fit = []
            for s in sols:
                full = np.asarray(x_start, float).copy()
                full[:9] = np.clip(s, 0.0, 1.0)
                p = project(full)
                pen = 100.0 * np.linalg.norm(p[:9] - full[:9])
                n_before = EV.n
                if used >= max_evals:
                    fit.append(np.inf if not fit else max(fit) + 1.0)
                    continue
                y = EV(p, label)
                used += EV.n - n_before
                fit.append(y + pen)
                if y < st["best_y"] - 1e-9 * max(1.0, abs(st["best_y"])):
                    last_imp = used
                if y < st["best_y"]:
                    st["best_y"], st["best_x"] = y, snap(p)
            fit = [f if np.isfinite(f) else max(v for v in fit if np.isfinite(v)) + 1.0 for f in fit]
            es.tell(sols, fit)
            st["hist"].append((EV.n, float(es.sigma), float(min(fit)), st["best_y"]))
            log(f"[{tag}] n={EV.n} used={used} sigma={es.sigma:.4g} gen_best={min(fit):.6g} "
                f"run_best={st['best_y']:.6g} gbest={EV.best_y:.6g}")
            if es.sigma < 1e-5:
                st["status"] = f"sigma<1e-5 at n={EV.n}"
                return
            if used - last_imp >= 150:
                st["status"] = f"no improvement for 150 evals at n={EV.n}"
                return
        st["status"] = f"budget for run used at n={EV.n}"
    except BudgetDone:
        st["status"] = "global budget exhausted"
    except Exception:
        st["status"] = "error: " + traceback.format_exc()
        log(st["status"])


def run_seq(jobs):
    for fn, a, k in jobs:
        if STOP.is_set():
            break
        fn(*a, **k)


if "CMA_RUNS" not in globals():
    CMA_RUNS = {}


def local_cycles(stop_at, L0=0.1, seed=100, label="turbo_local_refine", k_near=300):
    """Repeated local GP trust-region cycles re-centred on the global incumbent;
    a new cycle starts only if the previous one improved the global best by > 1e-3."""
    global TURBO_STATUS, CYCLES
    CYCLES = []
    c = 0
    while EV.n < stop_at and not STOP.is_set():
        X, Y = EV.arrays()
        near = list(np.argsort(np.linalg.norm(X - EV.best_x, axis=1))[:k_near])
        TR.reset(local_idx=near)
        TR.local_best = float(Y[near].min())
        TR.length = L0
        y0, n0 = EV.best_y, EV.n
        run_turbo(stop_at=stop_at, seed=seed + c, label=label)
        CYCLES.append(dict(n0=n0, n1=EV.n, y0=y0, y1=EV.best_y, status=TURBO_STATUS))
        log(f"cycle {c}: n {n0}->{EV.n}, best {y0:.6g} -> {EV.best_y:.6g} ({TURBO_STATUS})")
        c += 1
        if not TURBO_STATUS.startswith("collapsed") or y0 - EV.best_y <= 1e-3:
            break
    TURBO_STATUS = f"local_cycles finished at n={EV.n}"


def arm_cycles(x_center, fix10, stop_at, L0=0.1, seed=500, label="turbo_local_refine", k_near=300):
    """Local GP trust-region cycles restricted to one x10 level (arm); continue while the
    arm's own best improves by > 1e-3."""
    global TURBO_STATUS, ARM
    ARM = dict(fix10=fix10, cycles=[], best_y=None, best_x=None)
    TR.fix10 = fix10
    try:
        c = 0
        while EV.n < stop_at and not STOP.is_set():
            X, Y = EV.arrays()
            on = np.nonzero(X[:, 10] == fix10)[0]
            ia = int(on[np.argmin(Y[on])])
            xc = X[ia] if c > 0 else np.asarray(x_center)
            near = list(np.argsort(np.linalg.norm(X - xc, axis=1))[:k_near])
            if ia not in near:
                near.append(ia)
            TR.reset(local_idx=near)
            TR.local_best = float(Y[ia])
            TR.length = L0
            y0, n0 = float(Y[ia]), EV.n
            run_turbo(stop_at=stop_at, seed=seed + c, label=label)
            X, Y = EV.arrays()
            on = np.nonzero(X[:, 10] == fix10)[0]
            y1 = float(Y[on].min())
            ARM["cycles"].append(dict(n0=n0, n1=EV.n, y0=y0, y1=y1, status=TURBO_STATUS))
            ARM["best_y"], ARM["best_x"] = y1, X[on[np.argmin(Y[on])]]
            log(f"arm x10={fix10} cycle {c}: n {n0}->{EV.n}, arm best {y0:.6g} -> {y1:.6g} ({TURBO_STATUS})")
            c += 1
            if not TURBO_STATUS.startswith("collapsed") or y0 - y1 <= 1e-3:
                break
    finally:
        TR.fix10 = None
    TURBO_STATUS = f"arm_cycles finished at n={EV.n}"


def compass(steps, max_evals, label="compass_polish", min_scale=1 / 32, seed=0, start=None):
    """Compass/pattern search on x0..x8 from the global incumbent with per-coordinate base
    steps; on success keep doubling along that direction; halve the scale after a full
    unsuccessful poll. Strictly lower values only (plateau ties are not moves)."""
    global COMPASS
    rng = np.random.default_rng(seed)
    x, y = (EV.best_x.copy(), EV.best_y) if start is None else (np.asarray(start[0]).copy(), float(start[1]))
    scale, n0 = 1.0, EV.n
    COMPASS = dict(status="running", moves=[], scales=[])
    try:
        while scale >= min_scale and EV.n - n0 < max_evals and not STOP.is_set():
            improved = False
            for j in rng.permutation(9):
                for sgn in (1.0, -1.0):
                    if EV.n - n0 >= max_evals:
                        break
                    xt = x.copy()
                    xt[j] = np.clip(x[j] + sgn * scale * steps[j], 0.0, 1.0)
                    if xt[j] == x[j] or not feasible_rows(xt)[0]:
                        continue
                    yt = EV(xt, label)
                    if yt < y:
                        mult = 2.0
                        x, y = xt, yt
                        while EV.n - n0 < max_evals:  # expand along the successful direction
                            x2 = x.copy()
                            x2[j] = np.clip(x[j] + sgn * mult * scale * steps[j], 0.0, 1.0)
                            if x2[j] == x[j] or not feasible_rows(x2)[0]:
                                break
                            y2 = EV(x2, label)
                            if y2 < y:
                                x, y = x2, y2
                                mult *= 2.0
                            else:
                                break
                        COMPASS["moves"].append((EV.n, int(j), float(sgn * scale * steps[j]), y))
                        log(f"[compass] n={EV.n} move x{j} scale={scale:.4g} -> y={y:.6g}")
                        improved = True
                        break
            if not improved:
                scale *= 0.5
                COMPASS["scales"].append((EV.n, scale))
                log(f"[compass] n={EV.n} no improving poll, scale -> {scale:.4g} (best {y:.6g})")
        COMPASS["status"] = f"done at n={EV.n}, scale={scale:.4g}"
    except BudgetDone:
        COMPASS["status"] = "budget exhausted"
    except Exception:
        COMPASS["status"] = "error: " + traceback.format_exc()
        log(COMPASS["status"])


def quad_features(D9):
    n, d = D9.shape
    iu = np.triu_indices(d)
    Q = np.einsum("ni,nj->nij", D9, D9)[:, iu[0], iu[1]]
    return np.c_[np.ones(n), D9, Q]


def fit_quad(center, radius=0.08, level10=None, ridge=1e-6, max_pts=None):
    """Least-squares quadratic in x0..x8 around `center` on points with the same x10 level
    and x9 in {1/6, 1} (bit-identical there), within an inf-norm radius."""
    X, Y = EV.arrays()
    lv = center[10] if level10 is None else level10
    m = (X[:, 10] == lv) & (X[:, 9] > 0.1) & (np.abs(X[:, :9] - center[:9]).max(axis=1) <= radius)
    idx = np.nonzero(m)[0]
    if max_pts is not None and len(idx) > max_pts:
        idx = idx[np.argsort(np.abs(X[idx, :9] - center[:9]).max(axis=1))[:max_pts]]
    D9 = X[idx, :9] - center[:9]
    F = quad_features(D9)
    coef = np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ Y[idx])
    g = coef[1:10]
    H = np.zeros((9, 9))
    iu = np.triu_indices(9)
    H[iu] = coef[10:]
    H = H + H.T  # diag doubled -> Hessian convention f = c + g.d + 0.5 d'Hd
    return dict(idx=idx, coef=coef, g=g, H=H, c=coef[0], F=F, y=Y[idx])


def quad_predict(model, center, P):
    return quad_features(np.atleast_2d(P)[:, :9] - center[:9]) @ model["coef"]


def axis_range(x, j):
    """Allowed interval of coordinate j with all other coordinates fixed at x."""
    lo, hi = 0.0, 1.0
    rest = b - A @ x + A[:, j] * x[j]
    for a_ij, r in zip(A[:, j], rest):
        if a_ij > 0:
            hi = min(hi, r / a_ij)
        elif a_ij < 0:
            lo = max(lo, r / a_ij)
    return lo, hi


def scan_coord(xc, j, h, skip=None, label="coord_scan"):
    lo, hi = axis_range(xc, j)
    out = []
    for v in np.arange(np.ceil(lo / h) * h, hi + 1e-12, h):
        v = float(np.round(v, 6))
        if skip is not None and skip[0] - h / 2 <= v <= skip[1] + h / 2:
            continue
        x = xc.copy(); x[j] = v
        if not feasible_rows(x)[0]:
            continue
        out.append((v, EV(x, label)))
    return out
