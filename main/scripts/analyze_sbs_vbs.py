#!/usr/bin/env python3
"""Aggregate multi-task optimizer results using SBS and VBS.

SBS (single best solver) is the one solver with the lowest mean final
normalized regret over the available tasks. VBS (virtual best solver) is an
oracle that chooses the best solver separately for every task.

The default horizon is 100 evaluations, matching the agent/LLM campaigns.
Longer traces are truncated and early-stopped traces are padded with their
last incumbent, which represents performance at *at most* the budget.

Outputs are written to ``main/analysis/sbs_vbs``:
  - task_summary.csv: per-task SBS/VBS results and task scaling factors
  - solver_summary.csv: aggregate solver scores, coverage, and SBS ranking
  - trajectories.csv: mean normalized/raw trajectories by task and solver
  - sbs_vbs_summary.png: aggregate SBS versus VBS curves
  - solver_ranking.png: final solver ranking

Dionysus tasks are deliberately excluded from the analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "analysis" / "sbs_vbs"
EXCLUDED_TASKS = {"dionysus_delaney", "dionysus_freesolv"}

AGENT_METHODS = {
    "llm_openai_pb_s0",
    "llm_openai_nopb_s0",
    "rule_based",
    "claude_agent_run",
    "claude_agent_run_hptuning_tweak",
}

DISPLAY = {
    "bobyqa": "BOBYQA",
    "cma_es": "CMA-ES",
    "de": "Differential evolution",
    "gp_bo": "GP-BO",
    "lbfgsb": "L-BFGS-B",
    "nelder_mead": "Nelder-Mead",
    "random_search": "Random search",
    "sobol": "Sobol",
    "turbo": "TuRBO",
    "llm_openai_pb_s0": "OpenAI LLM (playbook)",
    "llm_openai_nopb_s0": "OpenAI LLM (no playbook)",
    "rule_based": "Rule-based router",
    "claude_agent_run": "Claude agent",
    "claude_agent_run_hptuning_tweak": "Claude agent (HP tweak)",
}


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def task_f_opt(task_dir: Path) -> float:
    metas = sorted(task_dir.glob("*/seed_*/meta.json")) + sorted(task_dir.glob("*/meta.json"))
    for path in metas:
        try:
            meta = load_json(path)
            if "task_f_opt" in meta:
                return float(meta["task_f_opt"])
        except (OSError, ValueError, TypeError):
            continue
    raise RuntimeError(f"No task_f_opt found under {task_dir}")


def trace_paths(method_dir: Path) -> list[Path]:
    direct = method_dir / "trace.csv"
    if direct.exists():
        return [direct]
    return sorted(method_dir.glob("seed_*/trace.csv"))


def load_values(path: Path) -> np.ndarray:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return np.array([], dtype=float)
    col = "merit" if "merit" in rows[0] else "y"
    vals = np.array([float(row[col]) for row in rows], dtype=float)
    return np.minimum.accumulate(vals)


def pad_or_truncate(values: np.ndarray, budget: int) -> np.ndarray:
    if len(values) == 0:
        return values
    if len(values) >= budget:
        return values[:budget]
    return np.pad(values, (0, budget - len(values)), mode="edge")


def method_type(name: str) -> str:
    return "agent/LLM" if name in AGENT_METHODS else "standard"


def task_data(task_dir: Path, budget: int) -> tuple[float, dict[str, list[np.ndarray]]]:
    f_opt = task_f_opt(task_dir)
    raw_by_method: dict[str, list[np.ndarray]] = {}
    for method_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        paths = trace_paths(method_dir)
        if not paths:
            continue
        traces = [load_values(path) for path in paths]
        traces = [pad_or_truncate(t, budget) for t in traces if len(t)]
        if traces:
            raw_by_method[method_dir.name] = traces
    return f_opt, raw_by_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--outdir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.budget < 1:
        raise SystemExit("--budget must be positive")

    tasks = sorted(
        p for p in RESULTS.iterdir()
        if p.is_dir() and p.name not in EXCLUDED_TASKS
    )
    if not tasks:
        raise SystemExit(f"No task directories found under {RESULTS}")

    # Load raw trajectories first so the scale is independent of any one solver.
    loaded = {}
    for task_dir in tasks:
        f_opt, raw_by_method = task_data(task_dir, args.budget)
        all_raw = np.concatenate([np.concatenate(v) for v in raw_by_method.values()])
        # Robust task scale: handles zero/negative optima and prevents a single
        # extreme noisy observation from dominating cross-task averages.
        scale = max(abs(f_opt), float(np.percentile(np.abs(all_raw - f_opt), 95)), 1e-12)
        loaded[task_dir.name] = (f_opt, scale, raw_by_method)

    trajectories = []
    task_rows = []
    solver_curves: dict[str, list[np.ndarray]] = defaultdict(list)
    solver_task_names: dict[str, list[str]] = defaultdict(list)

    for task, (f_opt, scale, raw_by_method) in loaded.items():
        norm_by_method = {}
        for method, traces in raw_by_method.items():
            raw_curve = np.mean(np.stack(traces), axis=0) - f_opt
            norm_curve = raw_curve / scale
            norm_by_method[method] = norm_curve
            solver_curves[method].append(norm_curve)
            solver_task_names[method].append(task)
            for eval_idx, (raw, norm) in enumerate(zip(raw_curve, norm_curve), start=1):
                trajectories.append({
                    "task": task,
                    "solver": method,
                    "solver_label": DISPLAY.get(method, method),
                    "solver_type": method_type(method),
                    "eval_idx": eval_idx,
                    "mean_raw_regret": raw,
                    "mean_normalized_regret": norm,
                    "n_runs": len(traces),
                    "f_opt": f_opt,
                    "task_scale": scale,
                })

        if not norm_by_method:
            continue
        final_scores = {m: float(curve[-1]) for m, curve in norm_by_method.items()}
        per_task_best = min(final_scores, key=final_scores.get)
        vbs_curve = np.min(np.stack(list(norm_by_method.values())), axis=0)
        # The global SBS is selected below, after all solver aggregates are
        # available. Keep the per-task best here for reporting only.
        task_rows.append({
            "task": task,
            "budget": args.budget,
            "n_solvers": len(norm_by_method),
            "per_task_best_solver": per_task_best,
            "per_task_best_solver_label": DISPLAY.get(per_task_best, per_task_best),
            "per_task_best_final_normalized_regret": vbs_curve[-1],
            "vbs_final_normalized_regret": vbs_curve[-1],
            "vbs_final_raw_regret": vbs_curve[-1] * scale,
            "task_scale": scale,
            "available_solvers": ";".join(sorted(norm_by_method)),
        })

    # Global SBS: one solver selected using its mean final normalized regret
    # over every task on which it has a result. VBS is the task-wise oracle.
    solver_summary = []
    for solver, curves in solver_curves.items():
        matrix = np.stack(curves)
        solver_summary.append({
            "solver": solver,
            "solver_label": DISPLAY.get(solver, solver),
            "solver_type": method_type(solver),
            "n_tasks": matrix.shape[0],
            "task_coverage": matrix.shape[0] / len(tasks),
            "mean_final_normalized_regret": float(matrix[:, -1].mean()),
            "median_final_normalized_regret": float(np.median(matrix[:, -1])),
            "mean_auc_normalized_regret": float(np.mean(matrix.mean(axis=1))),
        })
    solver_summary.sort(key=lambda row: row["mean_final_normalized_regret"])
    for rank, row in enumerate(solver_summary, start=1):
        row["sbs_rank"] = rank
    global_sbs = solver_summary[0]["solver"]

    # Aggregate curves. SBS uses the selected solver where available; VBS uses
    # the best available solver independently for each task and evaluation.
    sbs_norm_curves = []
    vbs_norm_curves = []
    for task, (f_opt, scale, methods) in loaded.items():
        normalized = {
            method: np.mean(np.stack(traces), axis=0) - f_opt
            for method, traces in methods.items()
        }
        normalized = {method: curve / scale for method, curve in normalized.items()}
        if global_sbs in normalized:
            sbs_norm_curves.append(normalized[global_sbs])
        if normalized:
            vbs_norm_curves.append(np.min(np.stack(list(normalized.values())), axis=0))

        # Add the global SBS result to the per-task summary now that the
        # global solver has been selected.
        row = next(item for item in task_rows if item["task"] == task)
        if global_sbs in normalized:
            row["global_sbs_final_normalized_regret"] = float(normalized[global_sbs][-1])
            row["global_sbs_final_raw_regret"] = float(normalized[global_sbs][-1] * scale)
        else:
            row["global_sbs_final_normalized_regret"] = ""
            row["global_sbs_final_raw_regret"] = ""

    sbs_mean = np.mean(np.stack(sbs_norm_curves), axis=0)
    vbs_mean = np.mean(np.stack(vbs_norm_curves), axis=0)
    global_sbs_row = next(row for row in solver_summary if row["solver"] == global_sbs)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "task_summary.csv", task_rows)
    write_csv(args.outdir / "solver_summary.csv", solver_summary)
    write_csv(args.outdir / "trajectories.csv", trajectories)

    with (args.outdir / "definition.json").open("w") as f:
        json.dump({
            "excluded_tasks": sorted(EXCLUDED_TASKS),
            "n_tasks": len(tasks),
            "budget": args.budget,
            "normalization": "max(abs(f_opt), 95th percentile(abs(y - f_opt)), 1e-12)",
            "global_sbs": global_sbs,
            "global_sbs_label": DISPLAY.get(global_sbs, global_sbs),
            "global_sbs_tasks_with_results": len(sbs_norm_curves),
            "vbs_tasks": len(vbs_norm_curves),
        }, f, indent=2)

    plot_summary(args.outdir, sbs_mean, vbs_mean, global_sbs, len(sbs_norm_curves), len(vbs_norm_curves), args.budget)
    plot_ranking(args.outdir, solver_summary, len(tasks))

    print(f"Analyzed {len(tasks)} tasks; excluded: {', '.join(sorted(EXCLUDED_TASKS))}")
    print(f"Global SBS: {global_sbs} ({DISPLAY.get(global_sbs, global_sbs)})")
    print(f"VBS coverage: {len(vbs_norm_curves)}/{len(tasks)} tasks")
    print(f"Wrote outputs to {args.outdir}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(outdir: Path, sbs: np.ndarray, vbs: np.ndarray, sbs_name: str, n_sbs: int, n_vbs: int, budget: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    x = np.arange(1, budget + 1)
    ax.plot(x, np.maximum(sbs, 1e-12), label=f"SBS: {DISPLAY.get(sbs_name, sbs_name)}", lw=2.4, color="#2a78d6")
    ax.plot(x, np.maximum(vbs, 1e-12), label="VBS (oracle)", lw=2.4, color="#e34948")
    ax.set_yscale("log")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Mean normalized simple regret")
    ax.set_title(f"Single Best Solver vs. Virtual Best Solver ({budget}-evaluation horizon)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.01, -0.2, f"SBS coverage: {n_sbs} tasks; VBS coverage: {n_vbs} tasks. Dionysus tasks excluded.", transform=ax.transAxes, fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(outdir / "sbs_vbs_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_ranking(outdir: Path, rows: list[dict], n_tasks: int) -> None:
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(rows) + 1)), dpi=150)
    values = [max(float(row["mean_final_normalized_regret"]), 1e-12) for row in rows]
    labels = [f"{row['solver_label']} ({int(row['n_tasks'])}/{n_tasks} tasks)" for row in rows]
    colors = ["#e87ba4" if row["solver_type"] == "agent/LLM" else "#2a78d6" for row in rows]
    ax.barh(np.arange(len(rows)), values, color=colors)
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.set_xscale("log")
    ax.set_xlabel("Mean final normalized regret (lower is better)")
    ax.set_title("Solver ranking used to select the global SBS")
    ax.grid(True, axis="x", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "solver_ranking.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
