"""Dry run of bo_lib on a synthetic function of my own (no harness, no budget)."""
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
import bo_lib as L


def f(x):
    x = L.snap(x)
    c = np.array([0.3, 0.7, 0.2, 0.4, 0.5, 0.1, 0.8, 0.6, 0.25])
    return float(np.sum((x[:9] - c) ** 2 * np.arange(1, 10)) + np.sin(8 * x[0]) * 0.3
                 + (x[9] - 0.166667) ** 2 + 2 * (x[10] - 0.227092) ** 2)


X = L.initial_design(20, seed=1)
assert all(L.allowed(x) for x in X)
y = np.array([f(x) for x in X])
print("init best", y.min(), L.heavy_tail_check(y))
for it in range(8):
    t = time.time()
    x, info, model = L.propose(X, y, seed=it, verbose=False)
    assert L.allowed(x)
    X = np.vstack([X, x]); y = np.append(y, f(x))
    print(it, round(y[-1], 4), "best", round(y.min(), 4), "t=%.1fs" % (time.time() - t),
          "loo", round(L.loo_spearman(model), 3), round(info["logei"], 3))
# local
c = X[np.argmin(y)]
t = time.time()
x, info, model = L.propose(X, y, seed=99, center=c, halfwidth=0.1, verbose=True)
print("local", f(x), L.allowed(x), "t=%.1fs" % (time.time() - t))
