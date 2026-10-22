"""Organic photovoltaic (OPV) device pool task: find the highest measured
power conversion efficiency (PCE) in a fixed pool of 526 literature devices
(Seifrid et al., J. Mater. Chem. A, 2024) by testing them one at a time.
There is no emulator: each evaluation looks up a measured value.

Needs only numpy. It reads three files from `task_006_baseline/` at the
repository root, written by preprocess.py there: pool_spec.json (feature
order), pool_features.csv (the candidates' min-max scaled features) and
pool_labels.csv (their PCE).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.tasks.base import Task

DATA_ROOT = Path(__file__).resolve().parents[3] / "task_006_baseline"

# Distance ties go to the lowest candidate_id, as in preprocess.nearest_unqueried.
_TIE_TOL = 1e-9


class OPVPoolTask(Task):
    """Maximize measured PCE (%) over a pool of 526 devices, minimized here
    as -PCE.

    The box is [0, 1]^11, one coordinate per pool feature in pool_spec.json
    order (HOMO_D, LUMO_D, Eg_D, HOMO_A, LUMO_A, Eg_A, DA_ratio,
    additive_conc, anneal_temp, HTL_energy, ETL_energy), each min-max scaled
    over the pool. An evaluation maps x to the nearest candidate not yet
    tested in this campaign (Euclidean distance, ties to the lowest
    candidate_id), marks it tested and returns its -PCE, so the same x can
    give different values within a campaign and a campaign has at most 526
    evaluations. `candidates` holds each candidate's point, in candidate_id
    order; `last_candidate` is the candidate_id of the latest evaluation.
    `f_opt` is exact: -PCE of the best candidate.
    """

    pool = True

    def __init__(self):
        self.private_data_paths = tuple(DATA_ROOT / name for name in
                                        ('pool_spec.json', 'pool_features.csv', 'pool_labels.csv'))
        spec = json.loads((DATA_ROOT / "pool_spec.json").read_text())
        names = spec["features"]
        with open(DATA_ROOT / spec["files"]["features"], newline="") as f:
            features = sorted(csv.DictReader(f), key=lambda r: int(r["candidate_id"]))
        with open(DATA_ROOT / spec["files"]["labels"], newline="") as f:
            pce = {int(r["candidate_id"]): float(r["PCE"]) for r in csv.DictReader(f)}

        self._ids = np.array([int(r["candidate_id"]) for r in features])
        self._unit = np.array([[float(r[f"{n}_unit"]) for n in names] for r in features])
        self._pce = np.array([pce[i] for i in self._ids])
        if len(pce) != len(self._ids) or len(self._ids) != spec["n_candidates"]:
            raise ValueError("pool_features.csv, pool_labels.csv and pool_spec.json disagree on the pool.")
        self._tested = np.zeros(len(self._ids), dtype=bool)
        self.last_candidate = None

        self.input_names = names
        self.name = "opv_pool_pce_d11"
        self.dim = len(names)
        self.bounds = np.tile([0.0, 1.0], (self.dim, 1))
        self.n_candidates = len(self._ids)
        self.candidates = self._unit.copy()
        self.f_opt = -float(self._pce.max())

    def nearest_untested(self, x: np.ndarray) -> int:
        """Row of the candidate closest to x among those not yet tested."""
        dist = np.sqrt(((self._unit - x) ** 2).sum(axis=1))
        dist[self._tested] = np.inf
        if self._tested.all():
            raise ValueError(f"all {self.n_candidates} candidates have been tested")
        return int(np.flatnonzero(dist <= dist.min() + _TIE_TOL)[0])

    def _f(self, x: np.ndarray) -> float:
        i = self.nearest_untested(x)
        self._tested[i] = True
        self.last_candidate = int(self._ids[i])
        return -float(self._pce[i])
