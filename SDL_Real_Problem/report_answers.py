"""Write each problem's found answers, in physical units, into README.md.

For every problem, three solutions are reported: the best-known reference
optimum (data/reference_optimum.json), the best point any baseline optimizer
evaluated, and the best point the v2 agent evaluated (over its seeds). Each
is decoded into what it means in the lab: a recipe, an instrument
calibration, or a cell layout.

The tables replace the text between <!-- ANSWERS:<problem> --> and
<!-- /ANSWERS:<problem> --> in README.md.

Usage (from the repository root): python SDL_Real_Problem/report_answers.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load(problem: str, module: str):
    for name in ("model", "task"):  # each problem folder has its own model.py / task.py
        cached = sys.modules.get(name)
        if cached is not None and Path(cached.__file__).parent != ROOT / problem:
            del sys.modules[name]
    sys.path.insert(0, str(ROOT / problem))
    spec = importlib.util.spec_from_file_location(f"{problem}_{module}", ROOT / problem / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


def best_point(trace_paths):
    """(y, x, source) of the lowest y over every row of the given traces."""
    best = None
    for path in trace_paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                y = float(row["y"])
                if best is None or y < best[0]:
                    xs = [float(row[k]) for k in row if re.fullmatch(r"x\d+", k)]
                    best = (y, np.array(xs), f"{path.parent.parent.name} {path.parent.name}")
    return best


def candidates(problem: str, task, x_ref):
    task_dir = next(p for p in (ROOT / problem / "results").iterdir() if p.is_dir())
    base = best_point(task_dir.glob("*/seed_*/trace.csv"))
    agent = best_point((ROOT / problem / "agent_runs" / "v2_scripted").glob("seed_*/trace.csv"))
    return [
        ("Best-known optimum", task.f_opt, np.asarray(x_ref, float), "reference search"),
        ("Best baseline", base[0], base[1], base[2]),
        ("v2 agent", agent[0], agent[1], agent[2].replace("v2_scripted ", "")),
    ]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def blend():
    p = "01_nonnewtonian_blend"
    task = load(p, "task").make_task()
    M = task.model
    ref = json.loads((ROOT / p / "data" / "reference_optimum.json").read_text())
    tgt = load(p, "model").target_viscosity_Pa_s(np.array([0.1, 1.0, 10.0, 100.0]))
    rows = []
    for name, y, x, src in candidates(p, task, list(ref["x_opt"].values())):
        t = M.objective_terms(x)
        eta = M.flow_curve(x, np.array([0.1, 1.0, 10.0, 100.0]))[0]
        rows.append([
            f"{name} ({src})", f"{y:.4f}", f"{x[0]:.3f}", f"{x[1]:.2f}", f"{x[2]:.2f}", f"{x[3]:.2f}", f"{x[4]:.1f}",
            f"{x[5]:.2f}", f"{x[6]:.0f}", f"{100 * (10 ** t['fit_log_rmse'] - 1):.1f}%", f"{t['cost_usd_per_kg']:.2f}",
            " / ".join(f"{e:.3g}" for e in eta),
        ])
    rows.append(["Target spec", "–", "", "", "", "", "", "5.0 ± 0.5", "", "", "", " / ".join(f"{e:.3g}" for e in tgt)])
    return table(
        ["Solution", "f", "Carbopol %", "Sepineo %", "Solagum %", "PEG %", "glycerol %", "pH", "mix (s)",
         "flow-curve error", "cost ($/kg)", "η at 0.1 / 1 / 10 / 100 s⁻¹ (Pa·s)"], rows)


def viscometer():
    p = "02_viscometer_calibration"
    task = load(p, "task").make_task()
    ref = json.loads((ROOT / p / "data" / "reference_optimum.json").read_text())
    br = load(p, "build_reference")
    rows = []
    for name, y, x, src in candidates(p, task, ref["x_opt"]):
        errs = [100 * (br.predict_viscosity(x[:3], h, d) / mu - 1) for _, mu, _, h, d in task.standards]
        rows.append([f"{name} ({src})", f"{y:.3f}%", f"{10 ** x[0]:.4g}", f"{x[1]:.4f}", f"{x[2]:.4f}",
                     f"{np.mean(np.abs(errs)):.1f}%", f"{np.max(np.abs(errs)):.1f}%"])
    c = br.REPO_CONSTANTS
    th = [c["log10_k"], c["p"], c["h_c"]]
    errs = [100 * (br.predict_viscosity(th, h, d) / mu - 1) for _, mu, _, h, d in task.standards]
    rows.append(["Lab's current constants (rheology_constants.py)", f"{br.projected(th, task.standards)[0]:.3f}%",
                 f"{10 ** th[0]:.4g}", f"{th[1]:.4f}", f"{th[2]:.4f}", f"{np.mean(np.abs(errs)):.1f}%", f"{np.max(np.abs(errs)):.1f}%"])
    return table(["Solution", "RMS drag error", "k", "p", "h_c (mm)",
                  "mean error vs certified μ", "worst standard"], rows)


def robot():
    p = "03_robot_cell_layout"
    task = load(p, "task").make_task()
    M = load(p, "model")
    ref = json.loads((ROOT / p / "data" / "reference_optimum.json").read_text())
    rows = []
    for name, y, x, src in candidates(p, task, ref["x_opt"]):
        r = M.evaluate(x)
        reach = "yes" if r["reach_deficit"] == 0 else "no"
        rows.append([
            f"{name} ({src})", f"{y:.1f}", f"({x[0]:.3f}, {x[1]:.3f}) m, {math.degrees(x[2]):.0f}°",
            f"({x[3]:.3f}, {x[4]:.3f}) m", f"{math.degrees(x[5]):.0f}°",
            f"{math.degrees(x[6]):.0f}° / {math.degrees(x[7]):.0f}°", f"{math.degrees(x[8]):.0f}°", reach,
            f"{r['cycle_time']:.1f}", f"{1000 * r['min_clearance']:.0f}",
        ])
    return table(["Solution", "f (s)", "arm base (x, y), yaw", "CNC (x, y)", "lid open", "grasp tilt x / y",
                  "pick azimuth", "reaches all targets", "cycle time (s)", "min clearance (mm)"], rows)


def main():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for problem, fn in [("01", blend), ("02", viscometer), ("03", robot)]:
        md = fn()
        print(f"--- {problem}\n{md}\n")
        pattern = re.compile(rf"(<!-- ANSWERS:{problem} -->\n).*?(\n<!-- /ANSWERS:{problem} -->)", re.S)
        text = pattern.sub(lambda m: m.group(1) + md + m.group(2), text)
    readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
