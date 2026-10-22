"""OPV local-solver sensitivity to 10 shared, seeded uniform starting points.

Use the same library calls/options as the repository wrappers. A budgeted
callback replaces their thread/queue adapter; no objective value is fabricated.
The original wrappers and their midpoint baseline remain unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import pybobyqa
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / 'main/src/tasks/opv_pool.py').exists())
sys.path.insert(0, str(REPO / 'main'))
from src.tasks.opv_pool import OPVPoolTask

METHODS = ('bobyqa', 'lbfgsb', 'nelder_mead')
BUDGET = 100


class BudgetExhausted(Exception):
    """Raised before an extra objective call; exits the solver unchanged."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solve(method, seed, x0):
    """Fresh pool state per run; count initialization and all solver probes."""
    np.random.seed(seed)
    task = OPVPoolTask()
    rows = []
    best = float('inf')
    elapsed_sum = 0.0
    started = time.perf_counter()

    def objective(x):
        nonlocal best, elapsed_sum
        if len(rows) >= BUDGET:
            raise BudgetExhausted()
        x = np.asarray(x, dtype=float)
        assert x.shape == (task.dim,) and np.isfinite(x).all()
        assert ((x >= task.bounds[:, 0]) & (x <= task.bounds[:, 1])).all()
        if not rows:
            assert np.array_equal(x, x0), 'Solver altered the shared starting point'
        observation = task.evaluate(x)
        best = min(best, observation.y)
        elapsed_sum += observation.elapsed
        row = {'eval_idx': len(rows) + 1, 'batch_idx': len(rows),
               'optimizer': method + '_random_start'}
        row.update({f'x{i}': float(v) for i, v in enumerate(x)})
        row.update(y=observation.y, candidate_id=int(task.last_candidate),
                   y_best_so_far=best, eval_seconds=observation.elapsed,
                   cumulative_seconds=elapsed_sum)
        rows.append(row)
        return observation.y

    stop_reason = 'solver_returned'
    try:
        if method == 'bobyqa':
            result = pybobyqa.solve(objective, x0.copy(),
                bounds=(task.bounds[:, 0], task.bounds[:, 1]),
                scaling_within_bounds=True, maxfun=20_000, do_logging=False)
            stop_reason = str(result.msg)
        else:
            options = ({'maxfun': 10**9, 'maxiter': 10**9} if method == 'lbfgsb'
                       else {'maxfev': 10**9, 'adaptive': True})
            result = minimize(objective, x0.copy(),
                method='L-BFGS-B' if method == 'lbfgsb' else 'Nelder-Mead',
                bounds=[tuple(b) for b in task.bounds], options=options)
            stop_reason = str(result.message)
    except BudgetExhausted:
        stop_reason = 'budget_exhausted'
    return rows, stop_reason, time.perf_counter() - started


def replay(rows):
    """Audit selection order, exact labels and incumbents without optimizing."""
    task = OPVPoolTask()
    seen = set()
    best = float('inf')
    for i, row in enumerate(rows, 1):
        assert int(row['eval_idx']) == i
        x = np.array([float(row[f'x{j}']) for j in range(task.dim)])
        index = task.nearest_untested(x)
        assert index not in seen
        seen.add(index)
        assert int(row['candidate_id']) == int(task._ids[index])
        assert float(row['y']) == -float(task._pce[index])
        task._tested[index] = True
        best = min(best, float(row['y']))
        assert float(row['y_best_so_far']) == best
    assert 0 < len(rows) <= BUDGET
    return {'status': 'passed', 'evaluations': len(rows),
            'unique_devices': len(seen), 'exact_pool_replay': True,
            'incumbents_verified': True, 'regret': best - task.f_opt}


def midpoint_check():
    """Verify callback plumbing against the actual ask/tell wrappers locally."""
    from src.optimizers.bobyqa import BOBYQA
    from src.optimizers.lbfgsb import LBFGSB
    from src.optimizers.nelder_mead import NelderMead
    from src.optimizers import OptimizerExhausted
    output = {}
    for method, cls in zip(METHODS, (BOBYQA, LBFGSB, NelderMead)):
        direct, _, _ = solve(method, 0, np.full(11, .5))
        task = OPVPoolTask()
        np.random.seed(0)
        optimizer = cls(bounds=task.bounds, seed=0)
        recorded = []
        for i in range(BUDGET):
            try:
                proposed = optimizer.ask(1)
            except OptimizerExhausted:
                break
            observation = task.evaluate(proposed[0])
            optimizer.tell(proposed, [observation.y])
            recorded.append((proposed[0].copy(), observation.y, int(task.last_candidate)))
        assert len(recorded) == len(direct)
        for (x, y, device), row in zip(recorded, direct):
            assert np.array_equal(x, [row[f'x{i}'] for i in range(11)])
            assert y == row['y'] and device == row['candidate_id']
        output[method] = {'evaluations': len(direct), 'matches_repository_wrapper_exactly': True}
    json_write(ROOT / 'midpoint_adapter_check.json', output)
    print('PASS: midpoint callback traces equal existing wrappers for all three methods.', flush=True)


def job(method, seed, x0, source_commit):
    folder = ROOT / 'baselines/opv/opv_pool_pce_d11' / (method + '_random_start') / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    start_time = datetime.now(timezone.utc).isoformat()
    rows, reason, elapsed = solve(method, seed, np.array(x0))
    audit = replay(rows)
    csv_write(folder / 'trace.csv', rows)
    metadata = {'task_name': 'opv_pool_pce_d11', 'task_dim': 11,
        'optimizer_name': method + '_random_start', 'seed': seed,
        'requested_budget': BUDGET, 'evaluations_used': len(rows),
        'initial_point': x0, 'initialization': 'default_rng(seed).uniform(0, 1, size=11)',
        'shared_start_across_local_methods': True,
        'stopped_early': len(rows) < BUDGET, 'stop_reason': reason,
        'source_commit': source_commit, 'runner_sha256': sha(__file__),
        'start_time_utc': start_time, 'end_time_utc': datetime.now(timezone.utc).isoformat(),
        'total_wall_seconds': elapsed,
        'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'Py-BOBYQA')}}
    json_write(folder / 'meta.json', metadata)
    json_write(folder / 'audit.json', audit)
    return method, seed, audit, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    midpoint_check()
    if args.check_only:
        return
    starts = [np.random.default_rng(seed).uniform(0, 1, size=11).tolist() for seed in range(10)]
    source_commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    paths = ['main/src/tasks/opv_pool.py', 'main/src/tasks/base.py',
             *[f'main/src/optimizers/{m}.py' for m in METHODS],
             *[f'task_006_baseline/{f}' for f in ('pool_spec.json', 'pool_features.csv', 'pool_labels.csv')]]
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'opv_pool', 'budget': BUDGET, 'seeds': list(range(10)),
        'methods': list(METHODS), 'starting_points': starts,
        'start_distribution': 'Independent uniform coordinates in [0,1]^11, default_rng(seed)',
        'shared_starts': True, 'initial_evaluation_counts_toward_budget': True,
        'source_commit': source_commit, 'source_sha256': {p: sha(REPO / p) for p in paths},
        'solver_settings': 'Identical to repository wrappers; only x0 changes. No restarts or padding after convergence.',
        'execution': 'Standalone numeric solvers with only objective callback; no LLM campaign is rerun.',
        'wall_time_limit_seconds': 1200}
    json_write(ROOT / 'protocol.json', protocol)
    csv_write(ROOT / 'starting_points.csv', [{'seed': s, **{f'x{i}': v for i, v in enumerate(x)}} for s, x in enumerate(starts)])
    began = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, method, seed, starts[seed], source_commit)
                   for method in METHODS for seed in range(10)]
        for future in as_completed(futures, timeout=1200):
            method, seed, audit, elapsed = future.result()
            results.append({'method': method + '_random_start', 'seed': seed, **audit, 'seconds': elapsed})
            print(f'{len(results)}/30 {method} seed={seed}: {audit["evaluations"]} calls, regret={audit["regret"]:.6g}, {elapsed:.2f}s', flush=True)
    for path, expected in protocol['source_sha256'].items():
        assert sha(REPO / path) == expected, f'Source changed during runs: {path}'
    json_write(ROOT / 'new_runs_summary.json', sorted(results, key=lambda x: (x['method'], x['seed'])))
    print(f'Complete: 30 audited runs in {time.perf_counter() - began:.1f}s.', flush=True)


if __name__ == '__main__':
    main()
