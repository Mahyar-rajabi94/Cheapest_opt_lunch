# Task definition: psa_v10_blackbox_run1

**Status: Confirmed by user** (2026-10-22; revision 2: budget set, discrete-materials search space added; no materials list supplied)

## Problem

- **Objective:** the scalar returned by `BlackBox()(x)` from the user-supplied
  module `C:\Users\rahim\Desktop\psa_v10_blackbox.py`. **Minimize** (`f.direction == "min"`).
- **Search space (user-stated):** a discrete set of materials. Each material is
  described by 8 features scaled to [0, 1]. The evaluator accepts any point in
  [0, 1]^8 and returns the value of the material that point maps to, so
  proposing continuous points is allowed. The list of materials is not supplied
  to the agent and stays sealed inside the evaluator.
- **Constraints:** none beyond the box.
- **Noise:** none. The evaluator documentation states it is deterministic.
- **Budget:** **2000 evaluations**. Every `harness.evaluate(x)` call counts as one,
  including a call that lands on a material already evaluated through a different `x`.
- **Execution limits:** none stated. Standard library only, so each evaluation is cheap.

## Python calling contract (public interface)

```python
from psa_v10_blackbox import BlackBox
f = BlackBox()
f.dim, f.bounds, f.direction      # 8, [[0.0, 1.0]] * 8, "min"
y = f(x)                          # x: 8 numbers in [0, 1] -> one float
```

Out-of-bounds, non-finite or wrongly sized inputs raise `ValueError`. The module
has no budget or logging of its own, so it is wrapped in the project's
`BlackBoxHarness` (`src/agent_harness/harness.py`), which enforces the budget and
writes `trace.csv` and `meta.json`:

```python
harness = BlackBoxHarness(objective=BlackBox(), bounds=[[0.0, 1.0]] * 8,
                          budget=2000, outdir=OUTDIR, task_id="psa_v10_blackbox")
harness.set_active_optimizer("<label>")
y = harness.evaluate(x)
harness.save()
```

Optimizer proposals are clipped to [0, 1] before evaluation so the evaluator's
bound check never rejects a point.

## Assumptions and unknowns

- Because the space is a finite set of materials, the objective is
  piecewise-constant over the box: flat within the region that maps to one
  material, with jumps between regions. The agent's earlier look at
  `BlackBox.evaluate` agrees with this. The embedded data has not been decoded
  or inspected and will not be.
- Consequences for planning: there are no useful gradients; many distinct `x`
  can return the same material, so the same value can recur; small steps
  often change nothing.
- The number of materials, how they are spread over the box, and the best value
  are unknown.

## Output

- Result directory: `main/agent_runs/psa_v10_blackbox_run1/` (new, empty).
  At completion it holds exactly `roadmap_initial.md`, `changelog.md`, `trace.csv`
  and `meta.json`.
- The campaign script lives outside it: `main/scripts/campaign_psa_v10_run1.py`.
- Environment: the repo has no `.venv` yet. A project `.venv` will be created with
  `--system-site-packages` so it reuses the numpy, scipy, cma and botorch that are
  already installed, instead of reinstalling torch.
