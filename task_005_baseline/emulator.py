"""Random-forest PCE emulator for the OPV benchmark. Needs only numpy.

train_emulator.py fits the forest with scikit-learn and stores its trees as flat
arrays in opv_emulator_rf.npz, so evaluating it needs neither scikit-learn nor
pickle. Predictions match RandomForestRegressor.predict (checked at training).

    from preprocess import prepare_query
    from emulator import Emulator

    em = Emulator()
    q = prepare_query(x_unit)      # snap HTL/ETL, check constraints
    pce = em.predict(q["x"])       # PCE in %, to be maximized
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_NPZ = HERE / "opv_emulator_rf.npz"


def save_forest(forest, inputs, path=MODEL_NPZ, info=None):
    """Store a fitted sklearn RandomForestRegressor as flat node arrays.

    Tree t occupies nodes roots[t] .. roots[t + 1] - 1. Leaves have left == right == -1.
    """
    left, right, feature, threshold, value, roots = [], [], [], [], [], []
    offset = 0
    for est in forest.estimators_:
        t = est.tree_
        is_leaf = t.children_left < 0
        roots.append(offset)
        left.append(np.where(is_leaf, -1, t.children_left + offset))
        right.append(np.where(is_leaf, -1, t.children_right + offset))
        feature.append(np.where(is_leaf, 0, t.feature))
        threshold.append(np.where(is_leaf, 0.0, t.threshold))
        value.append(t.value[:, 0, 0])
        offset += t.node_count
    np.savez_compressed(
        path,
        inputs=np.array(inputs),
        left=np.concatenate(left).astype(np.int32),
        right=np.concatenate(right).astype(np.int32),
        feature=np.concatenate(feature).astype(np.int8),
        threshold=np.concatenate(threshold).astype(np.float64),
        value=np.concatenate(value).astype(np.float64),
        roots=np.array(roots, dtype=np.int32),
        info=np.array(json.dumps(info or {})),
    )


class Emulator:
    """PCE (%) for snapped inputs in original units (the "x" of prepare_query)."""

    def __init__(self, path=MODEL_NPZ):
        with np.load(path) as z:
            self.inputs = [str(s) for s in z["inputs"]]
            self.left, self.right = z["left"], z["right"]
            self.feature, self.threshold = z["feature"].astype(np.intp), z["threshold"]
            self.value, self.roots = z["value"], z["roots"]
            self.info = json.loads(str(z["info"]))

    def predict(self, X):
        """X: 11 values or an (n, 11) array, columns in self.inputs order.
        Returns a float for one input, else an array of shape (n,)."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        X = np.atleast_2d(X)
        if X.shape[1] != len(self.inputs):
            raise ValueError(f"expected {len(self.inputs)} inputs, got {X.shape[1]}")
        if not np.isfinite(X).all():
            raise ValueError("inputs must be finite")
        # sklearn compares float32-cast inputs against float64 thresholds; do the same.
        X = X.astype(np.float32).astype(np.float64)
        rows = np.arange(len(X))[:, None]
        node = np.tile(self.roots, (len(X), 1))
        while True:
            inner = self.left[node] >= 0
            if not inner.any():
                break
            go_left = X[rows, self.feature[node]] <= self.threshold[node]
            node = np.where(inner, np.where(go_left, self.left[node], self.right[node]), node)
        pce = self.value[node].mean(axis=1)
        return float(pce[0]) if single else pce
