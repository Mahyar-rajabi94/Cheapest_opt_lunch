"""YAHPO Gym `rbv2_*` hyperparameter-optimization tasks, via their ONNX surrogates.

Requires `onnxruntime` (tested with 1.31.0); not in requirements.txt, install
it into `.venv` by hand. `yahpo-gym` itself pins numpy<2, which conflicts with
`ioh`, so its surrogate models are evaluated here directly with the same
input encoding (reproduces yahpo_gym 1.0.2 outputs exactly). The model files
(config_space.json, encoding.json, model.onnx per scenario, from
slds-lmu/yahpo_data v1.0.2) live in `external/yahpo_data/<scenario>/` at the
repository root, which git ignores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.tasks.base import Task

DATA_ROOT = Path(__file__).resolve().parents[3] / "external" / "yahpo_data"

# Model input column order, from yahpo_gym 1.0.2 (yahpo_gym/benchmarks/rbv2.py).
_CONT_NAMES = {
    "rbv2_svm": ["cost", "gamma", "tolerance", "degree", "trainsize", "repl"],
    "rbv2_xgboost": [
        "nrounds", "eta", "gamma", "lambda", "alpha", "subsample", "max_depth",
        "min_child_weight", "colsample_bytree", "colsample_bylevel", "rate_drop",
        "skip_drop", "trainsize", "repl",
    ],
}
_CAT_NAMES = {
    "rbv2_svm": ["task_id", "kernel", "num.impute.selected.cpo"],
    "rbv2_xgboost": ["task_id", "booster", "num.impute.selected.cpo"],
}
# Output columns are rbv2's y_names: acc, bac, auc, brier, f1, logloss, timetrain, timepredict, memory.
_LOGLOSS = 5
# Fixed, not searched: the instance and full fidelity (all training data, first replication).
_FIXED = {"trainsize": 1.0, "repl": 1}

# Best-known logloss (no exact optimum is known): min over 2^20 Sobol points
# + 20 CMA-ES runs of 1e5 evaluations (restarted until the budget is used) in
# the unit box, computed 2026-10-21. svm: all 20 runs reach it (Sobol best
# 0.25905). xgboost: 11 of 20 runs reach it, at the box edge (dart booster,
# nrounds at its maximum); Sobol best is 0.24225.
_F_OPT = {
    ("rbv2_svm", "40981"): 0.25901198387145996,
    ("rbv2_xgboost", "40981"): 0.08344290405511856,
}


class YAHPOTask(Task):
    """Minimize the surrogate logloss of one `rbv2_*` learner (svm or
    xgboost) on one OpenML instance (default 40981).

    The box is [0, 1]^D, one coordinate per searched hyperparameter in
    config_space.json order (task_id, trainsize, repl are fixed). A coordinate
    u decodes as: float, lo + u (hi - lo), interpolated in log space if
    log-scaled; int, the same rounded to the nearest integer; categorical with
    k choices, choice floor(u k) (u = 1 gives the last). Hyperparameters whose
    condition on the decoded parent fails are passed to the model as missing,
    so their coordinates have no effect. `f_opt` is a best-known value.
    """

    def __init__(self, scenario: str, instance: str = "40981"):
        import onnxruntime as ort

        if scenario not in _CONT_NAMES:
            raise ValueError(f"Only {sorted(_CONT_NAMES)} are implemented, got {scenario}.")
        data = DATA_ROOT / scenario
        space = json.loads((data / "config_space.json").read_text())
        self._encoding = json.loads((data / "encoding.json").read_text())
        if instance not in self._encoding["task_id"]:
            raise ValueError(f"{scenario} has no instance {instance}.")
        self._params = [p for p in space["hyperparameters"] if p["name"] not in ("task_id", *_FIXED)]
        self._conditions = {c["child"]: c for c in space["conditions"]}
        self._cont_names = _CONT_NAMES[scenario]
        self._cat_names = _CAT_NAMES[scenario]
        self._fixed = {"task_id": instance, **_FIXED}

        # Sweeps run one process per core, so keep each session single-threaded.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(data / "model.onnx"), options)
        self._inputs = [i.name for i in self._session.get_inputs()]

        self.name = f"yahpo_{scenario}_{instance}"
        self.dim = len(self._params)
        self.bounds = np.tile([0.0, 1.0], (self.dim, 1))
        self.f_opt = _F_OPT[(scenario, instance)]

    def _config(self, u: np.ndarray) -> dict:
        """Decode one unit-box point to a configuration of active hyperparameters."""
        config = {}
        for p, ui in zip(self._params, np.clip(u, 0.0, 1.0)):
            if p["type"] == "categorical":
                k = len(p["choices"])
                config[p["name"]] = p["choices"][min(int(ui * k), k - 1)]
                continue
            lo, hi = p["lower"], p["upper"]
            if p["log"]:
                v = float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo))))
            else:
                v = lo + ui * (hi - lo)
            # Clip: exp(log(hi)) can overshoot hi by rounding.
            config[p["name"]] = int(np.clip(round(v), lo, hi)) if p["type"] == "uniform_int" else min(max(v, lo), hi)
        for child, c in self._conditions.items():
            parent = config[c["parent"]]
            if (parent != c["value"]) if c["type"] == "EQ" else (parent not in c["values"]):
                del config[child]
        return config

    def _logloss(self, configs: list[dict]) -> np.ndarray:
        """Surrogate logloss for a batch of configurations; missing
        hyperparameters encode as yahpo_gym does ("#na#" / 0)."""
        configs = [{**c, **self._fixed} for c in configs]
        x_cat = np.array(
            [[self._encoding[n][c.get(n, "#na#")] for n in self._cat_names] for c in configs], dtype=np.int32
        )
        x_cont = np.array([[c.get(n, 0) for n in self._cont_names] for c in configs], dtype=np.float32)
        out = self._session.run(None, {self._inputs[0]: x_cat, self._inputs[1]: x_cont})[0]
        return out[:, _LOGLOSS].astype(float)

    def _f(self, x: np.ndarray) -> float:
        return float(self._logloss([self._config(x)])[0])
