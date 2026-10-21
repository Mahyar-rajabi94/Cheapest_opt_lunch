"""Plot agent runs against single-method baselines on one task (dev use only).

Makes two figures from trace.csv files:
  regret_vs_eval.png   best-so-far regret per evaluation, agents vs every baseline
  method_timeline.png  which method each agent run was executing, by evaluation

    python scripts/plot_agent_runs.py \
        --baselines results/rosenbrock_d10_i1 --override results_agent/baselines/rosenbrock_d10_i1 \
        --agent "v2 scripted=results_agent/v2_scripted_s{0..4}" \
        --agent "v1 Claude agent=../task_001_sandbox_claud/results/agent_task_001" \
        --out results_agent/plots

An --agent value may contain a {a..b} seed range; the first run is drawn bold,
the others thin. --override replaces baseline methods (e.g. rerun after a fix).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Paired hues: same family = same hue, solid vs dashed.
METHODS = {
    "random_search": ("Random search", "#2bb38a", "-"),
    "sobol": ("Sobol", "#1aa36f", "--"),
    "cma_es": ("CMA-ES", "#e9a21b", "-"),
    "de": ("Differential evolution", "#e9a21b", "--"),
    "turbo": ("TuRBO", "#e98aaa", "-"),
    "gp_bo": ("GP-BO", "#e98aaa", "--"),
    "bobyqa": ("BOBYQA", "#1f8a2e", "-"),
    "nelder_mead": ("Nelder-Mead", "#6c55d6", "-"),
    "lbfgsb": ("L-BFGS-B", "#e0504f", "-"),
    "probes": ("Probes", "#9a9a9a", "-"),
    "race": ("Race", "#7f8fa6", "-"),
}
AGENT_COLORS = ["#111111", "#b8009a", "#00a3c4", "#1f6fd6", "#f06a35"]
REGRET_FLOOR = 1e-12


def canon(name: str) -> str:
    """Map optimizer labels from v1/v2 traces to METHODS keys."""
    if name.startswith("probe"):
        return "probes"
    if name.startswith("race"):  # race candidates interleave; show the race as one block
        return "race"
    name = name.removesuffix("_local")
    return {"sobol_search": "sobol", "differential_evolution": "de"}.get(name, name)


def load(run_dir: Path, f_opt: float | None):
    with open(run_dir / "trace.csv") as fh:
        rows = list(csv.DictReader(fh))
    if f_opt is None:
        f_opt = json.load(open(run_dir / "meta.json")).get("task_f_opt")
    best = np.array([float(r["y_best_so_far"]) for r in rows])
    regret = np.maximum(best - f_opt, REGRET_FLOOR)
    methods = [canon(r["optimizer"]) for r in rows]
    return regret, methods


def segments(methods):
    """Run-length encode the method column: [(method, start_idx, end_idx_excl)]."""
    out, start = [], 0
    for i in range(1, len(methods) + 1):
        if i == len(methods) or methods[i] != methods[start]:
            out.append((methods[start], start, i))
            start = i
    return out


def expand(path: str):
    m = re.search(r"\{(\d+)\.\.(\d+)\}", path)
    if not m:
        return [Path(path)]
    return [Path(path[:m.start()] + str(s) + path[m.end():])
            for s in range(int(m.group(1)), int(m.group(2)) + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True, help="results/<task>: <method>/seed_0/trace.csv")
    ap.add_argument("--override", default=None, help="same layout; methods here replace --baselines")
    ap.add_argument("--agent", action="append", default=[], help="LABEL=run_dir (may use {a..b})")
    ap.add_argument("--f-opt", type=float, default=None)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--out", default="results_agent/plots")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    base_dirs = {p.name: p / "seed_0" for p in Path(a.baselines).iterdir() if (p / "seed_0").is_dir()}
    if a.override:
        base_dirs.update({p.name: p / "seed_0" for p in Path(a.override).iterdir() if (p / "seed_0").is_dir()})
    f_opt = a.f_opt
    if f_opt is None:
        f_opt = json.load(open(next(iter(base_dirs.values())) / "meta.json"))["task_f_opt"]

    agents = []  # (label, color, [(regret, methods, run_name)])
    for i, spec in enumerate(a.agent):
        label, path = spec.split("=", 1)
        runs = [(*load(d, f_opt), d.name) for d in expand(path)]
        agents.append((label, AGENT_COLORS[i % len(AGENT_COLORS)], runs))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: regret vs evaluation.
    fig, ax = plt.subplots(figsize=(13, 7.2))
    for key in METHODS:
        if key not in base_dirs:
            continue
        regret, _ = load(base_dirs[key], f_opt)
        name, color, ls = METHODS[key]
        ax.plot(np.arange(1, len(regret) + 1), regret, color=color, ls=ls, lw=1.4, label=name)
    for label, color, runs in agents:
        for j, (regret, methods, _) in enumerate(runs):
            x = np.arange(1, len(regret) + 1)
            lead = j == 0
            ax.plot(x, regret, color=color, lw=3 if lead else 1, alpha=1 if lead else 0.45,
                    label=f"{label}" + (f" (seed 0 bold, {len(runs)} seeds)" if len(runs) > 1 else "") if lead else None,
                    zorder=5 if lead else 4)
            if lead:
                sw = [s for _, s, _ in segments(methods)]
                ax.scatter(x[sw], regret[sw], s=40, color=color, edgecolor="white", lw=1.2, zorder=6)
    ax.set_yscale("log")
    ax.set_xlim(-60, a.budget + 60)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Regret (log scale)")
    ax.grid(True, color="#dddddd", lw=0.8)
    ax.set_title("Regret vs. evaluation: agents against every individual baseline", loc="left", fontsize=13)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              title="Paired hue = same family\n(solid vs. dashed)", title_fontsize=9)
    fig.text(0.01, 0.01, "Dots mark each agent's method switches (seed-0 run). Runs shorter than "
             f"{a.budget} evals stopped early (converged or exhausted). Regret floored at {REGRET_FLOOR:g}. "
             "Dev task, not an official result.", fontsize=8, color="#777777")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "regret_vs_eval.png", dpi=150)
    plt.close(fig)

    # Figure 2: method timeline per agent run.
    seed_of = lambda name: re.sub(r".*?(\d+)$", r"seed \1", name)
    rows = [(f"{label}" + (f" · {seed_of(name)}" if len(runs) > 1 else ""), methods)
            for label, _, runs in agents for _, methods, name in runs]
    fig, ax = plt.subplots(figsize=(13, 0.75 * len(rows) + 1.6))
    used = set()
    for r, (row_label, methods) in enumerate(rows):
        y = len(rows) - 1 - r
        for m, s, e in segments(methods):
            name, color, _ = METHODS.get(m, (m, "#cccccc", "-"))
            used.add(m)
            ax.barh(y, e - s, left=s, height=0.62, color=color, edgecolor="white", lw=1)
            if e - s >= 130:
                ax.text((s + e) / 2, y, name, ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")
        if len(methods) < a.budget:
            ax.text(len(methods) + 12, y, "stopped", va="center", fontsize=8, color="#777777")
    ax.set_yticks(range(len(rows)), [lab for lab, _ in reversed(rows)])
    ax.set_xlim(0, a.budget)
    ax.set_xlabel("Evaluation")
    ax.grid(True, axis="x", color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Which method each agent was running, by evaluation", loc="left", fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHODS[m][1]) for m in METHODS if m in used]
    ax.legend(handles, [METHODS[m][0] for m in METHODS if m in used], loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.text(0.01, 0.01, "Segment color matches that method's line in regret_vs_eval.png. "
             "Short segments are unlabeled; see the legend.", fontsize=8, color="#777777")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "method_timeline.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out / 'regret_vs_eval.png'} and {out / 'method_timeline.png'}")


if __name__ == "__main__":
    main()
