"""Build the data for the robot cell layout problem.

    python build_data.py validate    # FK/IK checks against sdl_dt, default layout terms
    python build_data.py dataset     # data/simulated_layouts.csv (2048 Sobol layouts, every term)
    python build_data.py reference   # data/reference_optimum.json (heavy multi-start search)

`validate` needs a checkout of sdl_dt (path below or --sdl-dt); the others only
need data/cell_geometry.json (written by extract_geometry.py).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402

LO, HI = M.BOUNDS[:, 0], M.BOUNDS[:, 1]


def default_layout(pick_az=None, flags=None) -> np.ndarray:
    g = M.GEO["default_layout"]
    az = math.radians(345.0) if pick_az is None else pick_az
    fl = [0.75, 0.75, 0.25, 0.75, 0.75, 0.75] if flags is None else flags
    return np.array([*g["base_xy"], g["base_yaw"], *g["cnc_xy"], g["lid_open"], *g["grasp_tilt"], az, *fl])


def validate(sdl_dt: str) -> dict:
    sys.path.insert(0, str(Path(sdl_dt) / "src"))
    from are312 import formulation as F

    rng = np.random.default_rng(0)
    report = {}
    # FK: our URDF chain vs sdl_dt's tool_pose (TCP), random arm poses and joint vectors.
    errs = []
    for _ in range(50):
        bx, by, yaw = rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.1), rng.uniform(-math.pi, math.pi)
        q = rng.uniform(-3, 3, 6)
        ours = (M.fk(M._trans(bx, by, M.PLATE) @ M._rz(yaw), q[None])[0]["tool0"] @ M._trans(0, 0, M.TCP))[0]
        ref = F.tool_pose(q, F.Pose(xyz=(bx, by, M.PLATE), rpy=(0, 0, yaw)))
        errs.append(np.abs(ours - ref).max())
    report["fk_max_abs_error_vs_sdl_dt"] = float(max(errs))
    # IK: every branch, FK(IK(T)) == T wherever the branch reports the target reachable.
    q = rng.uniform(-3, 3, (1000, 6))
    q[:, 2] = rng.uniform(-3.1, 3.1, 1000)
    base = M._trans(-0.08, -0.22, M.PLATE) @ M._rz(1.3)
    targets = M.fk(base, q)[0]["tool0"] @ M._trans(0, 0, M.TCP)
    worst, recovered = 0.0, np.full(1000, np.inf)
    for flags in itertools.product([1, -1], repeat=3):
        qs, deficit = M.ik(np.linalg.inv(base), targets, *flags)
        back = M.fk(base, qs)[0]["tool0"] @ M._trans(0, 0, M.TCP)
        err = np.abs(back - targets).reshape(1000, -1).max(1)
        worst = max(worst, float(err[deficit == 0].max()))
        recovered = np.minimum(recovered, err)
    report["ik_max_roundtrip_error_on_reachable"] = worst
    report["ik_every_target_recovered_by_some_branch"] = float(recovered.max())
    # Targets: our handle grasp / cup seat / cup on balance vs sdl_dt's.
    tilt = M.GEO["default_layout"]["grasp_tilt"]
    report["handle_grasp_max_abs_error"] = float(
        max(np.abs(M.handle_grasp(a, tilt) - F._handle_grasp(a, tuple(tilt))).max() for a in np.linspace(0, 1.2217, 7))
    )
    report["cup_grasp_max_abs_error"] = float(
        max(np.abs(M.cup_grasp(a) - F._cup_grasp(a)).max() for a in np.linspace(0, 6.28, 9))
    )
    cnc = F.FORMULATION_CNC_POSE
    report["cup_on_scale_max_abs_error"] = float(
        np.abs(M._trans(cnc.xyz[0], cnc.xyz[1], 0) @ M._CUP_ON_SCALE - F._cup_on_scale(cnc)).max()
    )
    # Default layout: best over the 64 branch pairs and 24 pick azimuths.
    best = None
    for flags in itertools.product([0.75, 0.25], repeat=6):
        for az in np.radians(np.arange(0, 360, 15)):
            r = M.evaluate(default_layout(az, list(flags)))
            if best is None or r["objective"] < best[0]["objective"]:
                best = (r, flags, az)
    r, flags, az = best
    report["default_layout"] = {
        "x": default_layout(az, list(flags)).tolist(),
        "pick_azimuth_deg": math.degrees(az),
        **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()},
        "sdl_dt_plan_lid_cycle_best_effort": "min_clearance -0.0355 m, duration 36.3 s (58 s per call)",
    }
    # Timing.
    t = time.perf_counter()
    for _ in range(50):
        M.evaluate(default_layout(az, list(flags)))
    report["ms_per_eval_reachable_layout"] = (time.perf_counter() - t) / 50 * 1000
    X = LO + np.random.default_rng(1).random((500, 15)) * (HI - LO)
    t = time.perf_counter()
    for x in X:
        M.evaluate(x)
    report["ms_per_eval_random_layout"] = (time.perf_counter() - t) / 500 * 1000
    (HERE / "data" / "validation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report


def dataset(n: int = 2048):
    from scipy.stats import qmc

    u = qmc.Sobol(15, scramble=True, seed=0).random(n)
    X = LO + u * (HI - LO)
    keys = ["objective", "feasible", "cycle_time", "reach_deficit", "footprint_overlap", "min_clearance",
            "collision", "ik_jumps", "manip_shortfall", "penalty"]
    with open(HERE / "data" / "simulated_layouts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(M.NAMES + keys)
        for x in X:
            r = M.evaluate(x)
            w.writerow([f"{v:.6g}" for v in x] + [r[k] for k in keys])
    print(f"wrote {n} layouts")


def _cma_run(args):
    import cma

    seed, x0, sigma, budget = args
    es = cma.CMAEvolutionStrategy(
        x0, sigma, {"bounds": [0, 1], "maxfevals": budget, "verbose": -9, "seed": seed, "tolfun": 1e-9}
    )
    es.optimize(lambda u: M.objective(LO + np.clip(u, 0, 1) * (HI - LO)))
    return float(es.result.fbest), (LO + es.result.xbest * (HI - LO)).tolist(), int(es.result.evaluations)


def _polish(x, budget=3000):
    """Nelder-Mead on the continuous coordinates with the branch flags fixed."""
    from scipy.optimize import minimize

    x = np.asarray(x, float)
    flags = x[9:]

    def f(c):
        return M.objective(np.concatenate([np.clip(c, LO[:9], HI[:9]), flags]))

    res = minimize(f, x[:9], method="Nelder-Mead", options={"maxfev": budget, "xatol": 1e-6, "fatol": 1e-6})
    return float(res.fun), np.concatenate([np.clip(res.x, LO[:9], HI[:9]), flags]).tolist()


def _map(fn, items, workers):
    if workers <= 1:  # in-process: no worker processes (low-memory machines)
        out = []
        for i, item in enumerate(items):
            out.append(fn(item))
            print(f"  {i + 1}/{len(items)}: {out[-1][0]:.4f}", flush=True)
        return out
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


def reference(restarts: int = 32, budget: int = 20000, workers: int = 8):
    rng = np.random.default_rng(2026)
    starts = [(default_layout() - LO) / (HI - LO)] + [rng.random(15) for _ in range(restarts - 1)]
    jobs = [(i + 1, s, 0.3, budget) for i, s in enumerate(starts)]
    t = time.perf_counter()
    runs = _map(_cma_run, jobs, workers)
    runs.sort(key=lambda r: r[0])
    print("CMA-ES restarts (best 5):", [round(r[0], 4) for r in runs[:5]], f"{time.perf_counter() - t:.0f}s", flush=True)
    polished = _map(_polish, [r[1] for r in runs[:8]], workers)
    best_f, best_x = min(polished + [(r[0], r[1]) for r in runs], key=lambda p: p[0])
    r = M.evaluate(best_x)
    out = {
        "f_opt": best_f,
        "x_opt": best_x,
        "x_names": M.NAMES,
        "terms": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()},
        "how_found": f"{restarts} CMA-ES restarts (sigma 0.3 in the unit box, {budget} evaluations each, "
        f"first start at sdl_dt's default layout, the rest uniform random), then Nelder-Mead polish "
        f"of the 9 continuous coordinates of the 8 best with their branch flags fixed; best of all. "
        f"At most {restarts * budget + 8 * 3000} evaluations in total. Best-known, not proven optimal.",
        "cma_finals": sorted(round(r[0], 6) for r in runs),
    }
    (HERE / "data" / "reference_optimum.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["validate", "dataset", "reference"])
    p.add_argument("--sdl-dt", default="C:/Users/mrast/Github_Mahdi/sdl_dt")
    p.add_argument("--restarts", type=int, default=32)
    p.add_argument("--budget", type=int, default=20000)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.what == "validate":
        validate(a.sdl_dt)
    elif a.what == "dataset":
        dataset()
    else:
        reference(a.restarts, a.budget, a.workers)
