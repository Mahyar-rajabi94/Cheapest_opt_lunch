"""Independent numeric and provenance checks for the updated OPV comparison."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import run_random_starts as experiment

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        assert experiment.sha(experiment.REPO / path) == digest, path
    curves = {}
    summaries = []
    first_points = {}
    for method in sorted((ROOT / 'baselines/opv/opv_pool_pce_d11').iterdir()):
        trajectories = set()
        values = []
        paths = sorted(method.glob('seed_*/trace.csv'))
        assert len(paths) == 10
        for path in paths:
            rows = read_csv(path)
            meta = json.loads(path.with_name('meta.json').read_text())
            seed = int(path.parent.name.split('_')[1])
            assert len(rows) == meta['evaluations_used'] == meta['requested_budget'] == 100
            assert seed == meta['seed']
            audit = experiment.replay(rows)
            xs = tuple(tuple(float(row[f'x{i}']) for i in range(11)) for row in rows)
            trajectories.add(xs)
            if method.name.endswith('_random_start'):
                expected = np.random.default_rng(seed).uniform(0, 1, size=11)
                assert np.array_equal(xs[0], expected)
                assert np.array_equal(meta['initial_point'], expected)
                assert meta['runner_sha256'] == experiment.sha(ROOT / 'run_random_starts.py')
                first_points.setdefault(seed, []).append((xs[0], int(rows[0]['candidate_id'])))
            values.append([float(row['y_best_so_far']) + 15.707252 for row in rows])
            summaries.append({'method': method.name, 'seed': seed, **audit})
        assert len(trajectories) == 10, method.name
        curves[method.name] = values
    assert len(summaries) == 100
    assert all(len(v) == 3 and len(set(v)) == 1 for v in first_points.values())
    curves['agent'] = []
    for seed in range(3):
        path = ROOT / 'runs/opv_pool' / f'seed_{seed}'
        rows = read_csv(path / 'scored_trace.csv')
        assert len(rows) == 100
        assert json.loads((path / 'offline_audit.json').read_text())['status'] == 'passed'
        curves['agent'].append([float(r['regret']) for r in rows])
    points = read_csv(ROOT / 'figures/opv_pool/comparison_curves.csv')
    assert len(points) == 1100
    for row in points:
        index = int(row['eval_idx']) - 1
        values = [v[index] for v in curves[row['method']]]
        assert int(row['seeds']) == len(values)
        assert abs(float(row['mean_regret']) - statistics.mean(values)) < 1e-12
        assert abs(float(row['sample_std']) - statistics.stdev(values)) < 1e-12
    result = {'status': 'passed', 'baseline_runs': 100, 'new_random_start_runs': 30,
              'evaluations_per_run': 100, 'all_baseline_rows_replayed': 10000,
              'shared_initial_points_and_devices_verified': True,
              'distinct_trajectories_per_baseline': 10,
              'independently_recomputed_mean_and_std_points': len(points),
              'source_and_runner_hashes_verified': True, 'baseline_results': summaries}
    (ROOT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: all 10,000 baseline observations replayed; 30 shared-start runs; 1,100 mean/SD points verified.')


if __name__ == '__main__':
    main()
