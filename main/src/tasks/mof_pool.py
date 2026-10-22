"""Private measured MOF pool from the supplied task_061 archive.

Preserve exact CSV row identity, including any identical feature vectors.
Only anonymous row indices and feature values are public. Minimize -raw_y;
material IDs and full-pool ranks are trusted-side reporting information.
"""
import csv
from numbers import Integral
from pathlib import Path
import time

import numpy as np

from .base import Evaluation, Task

DATA_PATH = Path(__file__).resolve().parents[2] / 'data/mof_pool/data.csv'


class MOFPoolTask(Task):
    pool = True
    indexed_pool = True

    def __init__(self, data_path=None):
        path = Path(data_path) if data_path is not None else DATA_PATH
        self.private_data_paths = (path,)
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if (len(fields) < 3 or fields[0] != 'id' or fields[-1] != 'y'
                    or fields[1:-1] != [f'x{i}' for i in range(len(fields) - 2)]):
                raise ValueError('Expected columns id,x0,x1,...,y.')
            records = list(reader)
        if not records or any(None in row or any(v is None for v in row.values()) for row in records):
            raise ValueError('Empty or malformed candidate table.')
        self._ids = tuple(row['id'] for row in records)
        if not all(self._ids) or len(set(self._ids)) != len(self._ids):
            raise ValueError('Candidate IDs must be nonempty and unique.')
        self.candidates = np.array([[float(row[f]) for f in fields[1:-1]] for row in records])
        self._raw_y = np.array([float(row['y']) for row in records])
        if not np.isfinite(self.candidates).all() or not np.isfinite(self._raw_y).all():
            raise ValueError('Candidate features and labels must be finite.')
        self.candidates.setflags(write=False)
        self.dim = self.candidates.shape[1]
        self.n_candidates = len(records)
        self.bounds = np.column_stack((self.candidates.min(0), self.candidates.max(0)))
        self.name = f'mof_pool_d{self.dim}_n{self.n_candidates}'
        self.f_opt = -float(self._raw_y.max())
        self._tested = np.zeros(self.n_candidates, dtype=bool)
        self.last_candidate = None

    @property
    def available_indices(self):
        return np.flatnonzero(~self._tested)

    def validate_index(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < self.n_candidates:
            raise ValueError('Expected a valid integer candidate index.')
        if self._tested[index]:
            raise ValueError('Candidate was already evaluated.')

    def evaluate_index(self, index):
        self.validate_index(index)
        start = time.perf_counter()
        y = -float(self._raw_y[index])
        self._tested[index] = True
        self.last_candidate = self._ids[index]
        return Evaluation(y=y, g=(), elapsed=time.perf_counter() - start)

    def _f(self, x):
        # The agent uses evaluate_index; exact-vector evaluation is retained for
        # Task compatibility. Duplicate rows select the first unused match.
        point = np.asarray(x, dtype=float)
        if point.shape != (self.dim,) or not np.isfinite(point).all():
            raise ValueError('Expected an exact finite candidate feature row.')
        matches = np.flatnonzero(np.all(self.candidates == point, axis=1) & ~self._tested)
        if not len(matches):
            raise ValueError('Point does not match an unused candidate row.')
        return self.evaluate_index(int(matches[0])).y

    def offline_observation(self, index):
        """Trusted analysis only; never called by the public transport protocol."""
        y = float(self._raw_y[index])
        return {'candidate_id': self._ids[index], 'raw_y': y,
                'rank': 1 + int(np.count_nonzero(self._raw_y < y)),
                'descending_rank': 1 + int(np.count_nonzero(self._raw_y > y))}
