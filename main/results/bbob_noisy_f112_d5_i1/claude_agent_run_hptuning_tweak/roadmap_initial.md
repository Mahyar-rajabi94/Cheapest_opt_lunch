# Roadmap — initial step

## Understood problem

- Objective: sealed black-box, 5-dimensional, bounds `[-5, 5]` per
  coordinate, **minimize**.
- Noise: stated to be **noisy and heavy-tailed** — repeated calls at the
  same `x` can return materially different values, and rare evaluations
  can be far off (outliers). No noise model (e.g. variance, distribution
  family) is given beyond this qualitative description.
- Budget: **100** `evaluate()` calls total, hard-enforced by the harness.
  This is a very small budget relative to the 5-dimensional, noisy,
  black-box setting (for reference, the project's own GP-BO wrapper
  documents its practical regime as "budgets of order 10 x dim", i.e.
  ~50 here; we have roughly 2x that, still tight).
- The function is treated as a sealed black box; no attempt will be made
  to identify it, and no source beyond the documented harness/optimizer
  interfaces has been or will be read.

## Key assumption driving the first step

Heavy-tailed noise with rare large outliers directly threatens *any*
method that treats a single `evaluate()` return value as ground truth —
this includes local closed-loop solvers (BOBYQA, Nelder-Mead, L-BFGS-B,
which only see one y per queried x) and also affects global methods: a
single unlucky outlier at a good x can throw off CMA-ES's generation
ranking, DE's greedy replacement, or (worse, since it enters the model as
training data) a GP's posterior fit if the GP's fitted/assumed noise
scale under-estimates how heavy the tails really are. Before committing
real evaluations to any optimizer, it is worth directly measuring how
large the noise is at a single point, since the stated budget (100) is
too small to spend on this question later if it turns out method choice
should change. This is exploratory evidence-gathering, not yet
optimization.

## Step 1 — noise characterization probe

- **Method:** repeated direct evaluation (no optimizer library) at a
  single fixed point: the midpoint of the box, `x0 = [0, 0, 0, 0, 0]`
  (the same box-midpoint convention used by this project's own local
  solvers, e.g. CMA-ES/BOBYQA/Nelder-Mead/L-BFGS-B, as a neutral,
  assumption-free starting point — not chosen based on any prior
  knowledge of the function).
- **Hyperparameters:** none to tune — this is a fixed-point repeated
  sampling probe, not an optimizer. The only choice is *how many* repeats
  to spend on it (see step length below) and *where* (box midpoint,
  justified above).
- **Why this suits the problem:** the only way to learn the actual scale
  and tail behavior of the stated noise (rather than guessing from the
  qualitative description) is to evaluate the same `x` more than once.
  This directly informs whether later optimization needs explicit
  robustness measures (e.g. replicating candidate evaluations before
  trusting an apparent improvement, favoring methods whose internal
  model infers/absorbs an observation-noise term such as a GP's
  likelihood noise, or being skeptical of any single very good value).

## Step length and checkpoint

- **Evaluations for this step:** 3 (`evaluate()` calls at `x0`, repeated).
- **Cumulative evaluation count at pause:** 3 / 100.
- **Why this length:** 3 repeats is the minimum that lets us see both a
  typical spread (from the closest pair) and whether a third draw
  surfaces an outlier, while spending well under 5% of the total budget
  on pure characterization before any real search begins. There is no
  value in a larger probe here: the budget is scarce, and 3 draws is
  already an initial per-x noise estimate useful to every method
  considered next (if the next step's chosen method also happens to
  revisit points, like CMA-ES's own objective-noise handling, these 3
  points are retained as real data, not wasted — but the step is sized
  for the measurement goal, not for warm-starting a specific method).

## What I want to learn, and early-stop conditions

- The approximate scale (e.g. range, rough spread) of `y` values at a
  fixed `x`, and whether one of the 3 draws looks like a clear outlier
  relative to the other two (consistent with the stated heavy tails).
- This evidence will inform the choice of method and settings for step 2
  (e.g. a global, population/model-based method over a closed-loop local
  solver that cannot see repeated evaluations; and whether to budget
  explicit replication of promising points later in the campaign).
- No early-stop condition applies within this 3-call step — it is already
  minimal; all 3 calls will be made before the first checkpoint.
