# Offline dry run of the machinery on a synthetic function (NOT the task harness; no budget used).
import numpy as np, time
ns = {}
exec(open("/tmp/claude-1000/-home-kevzhu14-Documents-task-005-sandbox/f418fad7-eaae-41a9-a8ac-f3a2e53bb715/scratchpad/t5core.py").read(), ns)
ns["LOG"] = "/tmp/claude-1000/-home-kevzhu14-Documents-task-005-sandbox/f418fad7-eaae-41a9-a8ac-f3a2e53bb715/scratchpad/dryrun/dry.log"
ns["CHANGELOG"] = "/tmp/claude-1000/-home-kevzhu14-Documents-task-005-sandbox/f418fad7-eaae-41a9-a8ac-f3a2e53bb715/scratchpad/dryrun/dry_changelog.md"
class Fake:
    def set_active_optimizer(self, n): pass
    def evaluate(self, x):
        x = np.asarray(x); return float(np.sum((x[:9]-0.3)**2 * np.arange(1,10)) + (x[9]-0.17)**2 + (x[10]-0.42)**2)
ns["EV"] = ns["Evaluator"](Fake(), budget=300)
ns["TR"] = ns["TurboState"]()
X0 = ns["feasible_sobol"](110, 0)
print("design feasible:", all(ns["allowed"](x) for x in X0), "x9 counts", np.unique(X0[:,9], return_counts=True)[1], "x10 counts", np.unique(X0[:,10], return_counts=True)[1])
for x in X0: ns["EV"](x, "a")
ns["TR"].reset(local_idx=list(range(110))); ns["TR"].local_best = min(ns["EV"].Y)
# projection test
rng = np.random.default_rng(1); bad = rng.random((200, 11)); P = np.array([ns["project"](x) for x in bad])
print("projection all allowed:", all(ns["allowed"](p) for p in P))
t = time.time(); ns["run_turbo"](stop_at=200, seed=3); print("status", ns["TURBO_STATUS"], "time", round(time.time()-t,1))
print("best", ns["EV"].best_y, "n", ns["EV"].n, "infeasible", ns["EV"].n_infeasible)
