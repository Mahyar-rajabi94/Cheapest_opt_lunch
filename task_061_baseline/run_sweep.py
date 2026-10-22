"""Sweep GPBO and TuRBO over the MOFs in data.csv, maximizing CO2 uptake.

Run from task_061_baseline, for example::

    python run_sweep.py --budget 100 --seeds 10 --workers 4 --n-init 10

Each continuous proposal is projected onto the nearest unevaluated MOF in
min/max-normalized descriptor space. Only measured table values are used;
the optimizer learns from the selected MOF's actual descriptor coordinates
and negative uptake (the wrappers minimize). Ranks are computed over the
entire input table: rank 1 is highest uptake, and ties share their best rank.

Outputs under --outdir (default: results_sweep):
    bounds.json                 per-descriptor [min, max], in CSV column order
    descriptor_names.json       column names corresponding to bounds.json
    sweep.json                  configuration, including active descriptors
    best_ranks.csv              one final best rank per optimizer and seed
    <optimizer>/seed_<seed>/trace.csv   best rank after every evaluation
    <optimizer>/seed_<seed>/meta.json   completed-run summary

Constant descriptors are saved in bounds.json but omitted from the optimizer
inputs. --n-init overrides the wrappers' dimension-based initial designs so
that short sweeps can fit a GP. Repeating an identical sweep skips completed
runs; incompatible or incomplete existing results require a new --outdir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
TARGET = "pure_uptake_CO2_298.00_15000"
OPTIMIZERS = ("gp_bo", "turbo")
SUMMARY_FIELDS = [
    "optimizer", "seed", "evaluations_used", "best_mof", "best_row_index",
    "best_uptake", "best_rank", "total_mofs", "total_wall_seconds",
]
TRACE_FIELDS = [
    "eval_idx", "optimizer", "seed", "row_index", "mof", TARGET, "rank",
    "best_mof_so_far", "best_uptake_so_far", "best_rank_so_far",
]


@dataclass
class MOFPool:
    names: list[str]
    descriptor_names: list[str]
    descriptors: np.ndarray
    uptake: np.ndarray
    bounds: np.ndarray
    data_sha256: str

    @property
    def active(self) -> np.ndarray:
        return self.bounds[:, 1] > self.bounds[:, 0]

    @property
    def optimizer_bounds(self) -> np.ndarray:
        return self.bounds[self.active]

    @property
    def optimizer_points(self) -> np.ndarray:
        return self.descriptors[:, self.active]


def load_pool(path: str | Path) -> MOFPool:
    """Read the supplied pool, never using uptake as an input descriptor."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("data CSV has duplicate column names")
        if "name" not in fields or TARGET not in fields:
            raise ValueError(f"data CSV must contain 'name' and {TARGET!r}")
        descriptor_names = [
            field for field in fields
            if field not in ("", "name", TARGET) and not field.startswith("Unnamed:")
        ]
        if not descriptor_names:
            raise ValueError("data CSV has no descriptor columns")
        names, descriptors, uptake = [], [], []
        for line, row in enumerate(reader, start=2):
            try:
                values = [float(row[field]) for field in descriptor_names]
                objective = float(row[TARGET])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"non-numeric or missing value at CSV line {line}") from exc
            if not row["name"] or not np.isfinite(values).all() or not np.isfinite(objective):
                raise ValueError(f"missing MOF name or non-finite value at CSV line {line}")
            names.append(row["name"])
            descriptors.append(values)
            uptake.append(objective)
    if not names:
        raise ValueError("data CSV contains no MOFs")
    points = np.asarray(descriptors, dtype=float)
    bounds = np.column_stack((points.min(axis=0), points.max(axis=0)))
    if not np.any(bounds[:, 1] > bounds[:, 0]):
        raise ValueError("at least one descriptor must vary across the MOFs")
    return MOFPool(
        names, descriptor_names, points, np.asarray(uptake), bounds,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


class PoolProjection:
    """Project proposals to distinct table rows using only descriptors."""

    def __init__(self, points: np.ndarray, bounds: np.ndarray):
        self.points = points
        self.lower = bounds[:, 0]
        self.span = bounds[:, 1] - self.lower
        self.normalized = (points - self.lower) / self.span
        self.available = np.ones(len(points), dtype=bool)

    def select(self, proposal: np.ndarray) -> int:
        proposal = np.asarray(proposal, dtype=float)
        if proposal.shape != self.lower.shape or not np.isfinite(proposal).all():
            raise ValueError("optimizer returned an invalid descriptor proposal")
        if not self.available.any():
            raise ValueError("all MOFs have already been evaluated")
        unit = (proposal - self.lower) / self.span
        distance = np.sum((self.normalized - unit) ** 2, axis=1)
        distance[~self.available] = np.inf
        index = int(np.argmin(distance))
        self.available[index] = False
        return index


def _optimizer_classes():
    # Lazy imports allow --help and data validation without loading BoTorch.
    from src.optimizers.gp_bo import GPBO
    from src.optimizers.turbo import TuRBO

    return {"gp_bo": GPBO, "turbo": TuRBO}


def _make_optimizer(name: str, pool: MOFPool, seed: int, n_init: int):
    base = _optimizer_classes()[name]
    projection = PoolProjection(pool.optimizer_points, pool.optimizer_bounds)

    class PoolOptimizer(base):
        """Return actual MOF coordinates from ask, for matching tell calls."""

        def ask(self, n=1):
            if n != 1:
                raise ValueError("the pool sweep evaluates one MOF at a time")
            proposal = super().ask(n=1)[0]
            self.row_index = projection.select(proposal)
            return projection.points[self.row_index:self.row_index + 1].copy()

    optimizer = PoolOptimizer(bounds=pool.optimizer_bounds, seed=seed)
    optimizer.n_init = n_init
    return optimizer


def _init_worker():
    import torch

    torch.set_num_threads(1)


def run_campaign(data: str, name: str, seed: int, budget: int, n_init: int,
                 campaign_dir: str) -> dict:
    """Run one optimizer/seed, preserving raw uptake and best-rank history."""
    import torch

    start = time.perf_counter()
    pool = load_pool(data)
    np.random.seed(seed)
    torch.manual_seed(seed)  # TuRBO also uses Torch's global RNG.
    optimizer = _make_optimizer(name, pool, seed, min(n_init, budget))
    directory = Path(campaign_dir)
    directory.mkdir(parents=True, exist_ok=False)
    sorted_uptake = np.sort(pool.uptake)
    ranks = len(pool.names) - np.searchsorted(sorted_uptake, pool.uptake, side="right") + 1
    best_index = None

    with (directory / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for evaluation in range(1, budget + 1):
            actual = optimizer.ask(n=1)
            index = optimizer.row_index
            uptake = float(pool.uptake[index])
            optimizer.tell(actual, [-uptake])
            if best_index is None or uptake > pool.uptake[best_index]:
                best_index = index
            writer.writerow({
                "eval_idx": evaluation, "optimizer": name, "seed": seed,
                "row_index": index, "mof": pool.names[index], TARGET: uptake,
                "rank": int(ranks[index]),
                "best_mof_so_far": pool.names[best_index],
                "best_uptake_so_far": float(pool.uptake[best_index]),
                "best_rank_so_far": int(ranks[best_index]),
            })
            stream.flush()

    summary = {
        "optimizer": name, "seed": seed, "evaluations_used": budget,
        "best_mof": pool.names[best_index], "best_row_index": best_index,
        "best_uptake": float(pool.uptake[best_index]),
        "best_rank": int(ranks[best_index]), "total_mofs": len(pool.names),
        "total_wall_seconds": time.perf_counter() - start,
    }
    metadata = dict(summary, requested_budget=budget, n_init=min(n_init, budget),
                    target=TARGET, direction="max", data_sha256=pool.data_sha256)
    (directory / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data.csv")
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--budget", type=_positive_int, default=100,
                        help="distinct MOFs evaluated per optimizer/seed (default: 100)")
    parser.add_argument("--seeds", type=_positive_int, default=10,
                        help="run seeds 0 through seeds-1 (default: 10)")
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--n-init", type=_positive_int, default=10,
                        help="initial design size before fitting a GP (default: 10)")
    parser.add_argument("--outdir", type=Path, default=Path("results_sweep"))
    args = parser.parse_args()
    if args.n_init < 2:
        parser.error("--n-init must be at least 2 for GP fitting")
    if len(args.optimizers) != len(set(args.optimizers)):
        parser.error("--optimizers must not contain duplicates")
    return args


def run_sweep(args) -> int:
    pool = load_pool(args.data)
    if args.budget > len(pool.names):
        raise ValueError(f"--budget exceeds the pool of {len(pool.names)} distinct MOFs")
    configuration = {
        "data_sha256": pool.data_sha256, "target": TARGET, "direction": "max",
        "budget": args.budget, "seeds": args.seeds, "n_init": min(args.n_init, args.budget),
        "optimizers": args.optimizers, "total_mofs": len(pool.names),
        "descriptor_names": pool.descriptor_names, "bounds": pool.bounds.tolist(),
        "active_descriptors": [column for column, active in zip(pool.descriptor_names, pool.active)
                               if active],
        "projection": "nearest unevaluated MOF by min/max-normalized Euclidean distance",
        "rank": "descending competition rank; 1 = highest uptake; ties share rank",
    }
    manifest = args.outdir / "sweep.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != configuration:
            raise ValueError("existing sweep has a different configuration; use a new --outdir")
    elif args.outdir.exists() and any(args.outdir.iterdir()):
        raise ValueError("output directory is not empty; use a new --outdir")

    todo, completed = [], []
    for name in args.optimizers:
        for seed in range(args.seeds):
            directory = args.outdir / name / f"seed_{seed}"
            if directory.exists():
                meta = directory / "meta.json"
                if not meta.exists() or not (directory / "trace.csv").exists():
                    raise ValueError(f"incomplete existing run: {directory}; use a new --outdir")
                metadata = json.loads(meta.read_text())
                if (metadata["evaluations_used"] != args.budget
                        or metadata["data_sha256"] != pool.data_sha256):
                    raise ValueError(f"incompatible existing run: {directory}; use a new --outdir")
                completed.append({field: metadata[field] for field in SUMMARY_FIELDS})
                print(f"{name} seed={seed} skipped (complete)", flush=True)
            else:
                todo.append((str(args.data.resolve()), name, seed, args.budget,
                             min(args.n_init, args.budget), str(directory)))

    if todo:
        try:
            _optimizer_classes()
        except ModuleNotFoundError as exc:
            raise ValueError(f"missing dependency {exc.name!r}; install with "
                             "python -m pip install -r requirements.txt") from exc
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(configuration, indent=2) + "\n")
    (args.outdir / "bounds.json").write_text(json.dumps(pool.bounds.tolist(), indent=2) + "\n")
    (args.outdir / "descriptor_names.json").write_text(
        json.dumps(pool.descriptor_names, indent=2) + "\n")
    print(f"Pool: {len(pool.names)} MOFs, {len(pool.descriptor_names)} descriptors, "
          f"{int(pool.active.sum())} varying; budget={args.budget}, "
          f"initial design={min(args.n_init, args.budget)}", flush=True)

    failed = []

    def finished(job, result=None, error=None):
        _, name, seed, *_ = job
        if error is not None:
            failed.append((name, seed))
            print(f"{name} seed={seed} FAILED: {error!r}", file=sys.stderr, flush=True)
        else:
            completed.append(result)
            print(f"{name} seed={seed} best={result['best_mof']} "
                  f"rank={result['best_rank']}/{result['total_mofs']} "
                  f"({result['total_wall_seconds']:.1f}s)", flush=True)

    if todo and args.workers == 1:
        _init_worker()
        for job in todo:
            try:
                result = run_campaign(*job)
            except Exception as exc:
                finished(job, error=exc)
            else:
                finished(job, result=result)
    elif todo:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(run_campaign, *job): job for job in todo}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    finished(job, error=exc)
                else:
                    finished(job, result=result)

    summary_path = args.outdir / "best_ranks.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(completed, key=lambda row: (row["optimizer"], row["seed"])))
    print(f"Saved {summary_path}; completed={len(completed)}, failed={len(failed)}", flush=True)
    return 1 if failed else 0


def main() -> int:
    args = _parse_args()
    try:
        return run_sweep(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
