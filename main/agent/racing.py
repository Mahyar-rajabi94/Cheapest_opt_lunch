"""Portfolio racing by successive halving (design §5). UCB: deferred to v2.1.

All candidates are warm-started from the same archive state. Each rung
splits its budget equally among the surviving candidates; evaluation is
interleaved in chunks so no candidate benefits from going first. After each
rung the worse half (by the best value each candidate found itself) is
dropped. The winner keeps its internal state and becomes the active method.
"""

from __future__ import annotations

import math

import numpy as np


def successive_halving(ctl, candidates: list[str], budget: int) -> dict:
    cands = list(dict.fromkeys(candidates))
    n = len(cands)
    rungs = max(1, math.ceil(math.log2(n)))
    per_rung = budget // rungs
    _, y0 = ctl.archive.incumbent() if len(ctl.archive) else (None, np.inf)
    alive = {m: {"opt": ctl._new_opt(m, start="incumbent" if len(ctl.archive) else "center"),
                 "best": np.inf, "evals": 0, "done": False} for m in cands}
    history = []
    chunk = max(1, ctl.dim)
    for r in range(rungs):
        if ctl.ev.left <= 0:
            break
        share = per_rung // len(alive)
        target = {m: alive[m]["evals"] + share for m in alive}
        progressing = True
        while progressing and ctl.ev.left > 0:
            progressing = False
            for m, c in alive.items():
                if c["done"] or c["evals"] >= target[m]:
                    continue
                n_run = min(chunk, target[m] - c["evals"])
                done, ev = ctl.run_optimizer(c["opt"], n_run, f"race:{m}",
                                             check_converged=False)
                c["evals"] += done
                own = c["opt"].y_
                if len(own):
                    c["best"] = float(np.min(own))
                if ev in ("exhausted", "budget") or done == 0:
                    c["done"] = True
                progressing = True
        ranking = sorted(alive, key=lambda m: alive[m]["best"])
        history.append({"rung": r, "share": int(share),
                        "scores": {m: float(f"{alive[m]['best']:.6g}") for m in ranking}})
        if len(alive) > 1:
            keep = ranking[: max(1, len(alive) // 2)]
            for m in list(alive):
                if m not in keep:
                    alive[m]["opt"].close()
                    del alive[m]
    winner = min(alive, key=lambda m: alive[m]["best"])
    return {
        "candidates": cands,
        "winner": winner,
        "winner_opt": alive[winner]["opt"],
        "winner_finished": alive[winner]["done"],
        "start_best": float(f"{y0:.10g}") if np.isfinite(y0) else None,
        "rungs": history,
    }
