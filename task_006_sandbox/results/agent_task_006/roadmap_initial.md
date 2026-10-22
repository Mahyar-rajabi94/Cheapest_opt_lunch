# Roadmap (pre-registered, written before any evaluation) — Task 006

## What I know going in

- Minimize a black box over x in [0, 1]^11 (d = 11).
- Hard budget: 100 evaluations, about 9 per dimension. That is very small for
  11-D, so every evaluation has to be chosen by a model, not by a
  population-based search.
- No random noise, but the function is **not repeatable**: f(x) depends on
  which points were already evaluated. From the outside that looks like a
  combination of (i) noise or nuisance variation tied to the history and/or
  (ii) a systematic drift or penalty tied to the history (for example to the
  evaluation index or to how close x is to earlier points). I don't know
  which, so the plan measures it early and branches on what it finds.

## Method choice and why

| Phase | Evals (planned) | Method (harness label) | Purpose |
|---|---|---|---|
| A1 | 1–22 | `sobol_init`: the box centre (eval 1), then 21 scrambled Sobol points (seed 6) | Space-filling design so a GP fit means something (~2(d+1) points). |
| A2 | 23–24 | `replicate_probe`: re-evaluate the centre (eval 23) and the best point so far (eval 24) | Measure the size of the history effect directly, from two same-x pairs. |
| B | 25–~72 | `gp_bo_global`: GP-BO (BoTorch SingleTaskGP with an ARD kernel, dimension-scaled lengthscale prior, a learned Gaussian noise term, and standardized outputs) | Global, sample-efficient search. This is the right regime for GP-BO (about 10·d evals). |
| C | ~73–100 | `trbo_local`: trust-region BO around the incumbent (TuRBO-style box scaled by the ARD lengthscales, LogEI inside the box) | Local refinement of the best basin. |

Reasoning:

- **Why not CMA-ES, DE, or Nelder–Mead as the main engine:** CMA-ES's default
  population size in 11-D is 11. That gives only ~9 generations, which is not
  enough to adapt a covariance matrix. DE and NM are worse still at this budget.
- **Why not BOBYQA for the local phase:** in 11-D it needs 2d+1 = 23
  evaluations just to build its first quadratic model, which uses up nearly
  all of phase C before it takes a single model-driven step. It is also
  fragile when repeated evaluations disagree (the trust region collapses on
  inconsistent data).
- **Why GP-BO with a learned noise term:** at ~100 evals this is the most
  sample-efficient general tool. The learned noise variance soaks up the
  history-dependence if it acts like noise, instead of letting the GP
  interpolate it exactly.
- **Why a local trust-region phase at the end:** global EI in 11-D keeps
  spending evaluations on boundary and corner exploration. The last ~25–30
  evals are better spent refining the best basin found.
- **Acquisition details:** I use analytic LogEI by default and switch to
  qLogNEI (q = 1) if the replicate probes show a material history effect
  (rule D1). I optimize the acquisition with optimize_acqf (10 restarts,
  1024 raw samples).
- **Output warping:** the GP is fit on y as-is, unless the initial design is
  strongly right-skewed (rule D2).

## Pre-registered deviation rules (quantitative)

Let s = standard deviation of y over the 22 phase-A1 points. Let
δ = max |y_repeat − y_original| over the two replicate pairs.

- **D1 – history effect size (decided at eval 24):**
  - δ < 0.05·s: treat as effectively deterministic. Use LogEI with best_f =
    the best observed y.
  - 0.05·s ≤ δ ≤ 0.5·s: switch to qLogNEI (q = 1), which handles noise properly.
  - δ > 0.5·s: the history effect is comparable to the signal. Check for a
    trend against evaluation index (|Spearman ρ| ≥ 0.4 over evals 1–24). If
    there is one, add the normalized evaluation index as a 12th GP input and
    optimize the acquisition at the next index. If there is no trend, use
    TuRBO with Thompson sampling, which needs only rankings, not exact
    levels.
- **D2 – output skew (decided at eval 22):** if (max − median) > 5 × (median
  − min) of the phase-A1 values, fit the GP on log(y − min(y) + 0.01·IQR)
  instead of raw y.
- **D3 – GP usefulness check (at eval 45):** compute the leave-one-out
  Spearman correlation between GP predictions and observed values. If it is
  < 0.3, the global GP is not informative. Switch the rest of phase B to
  TuRBO (Thompson sampling, batch 1–4).
- **D4 – stagnation in phase B:** if the best observed y has not improved by
  at least 1% of the phase-A1 range (max − min) over 12 consecutive BO
  evaluations, start phase C early. Phase C starts no later than eval 75
  regardless.
- **D5 – trust-region control in phase C:** the initial side length is 0.4 in
  normalized units. Halve it after 3 consecutive non-improving evals and
  double it (capped at 0.8) after 2 consecutive improving evals. If the side
  length falls below 0.02, or there are 8 consecutive failures, restart the
  trust region on the best point lying > 0.3 (Euclidean) from the incumbent
  and in the top 10% of observed values. If there is no such point, spend 4
  evals on global LogEI before re-centring.
- **D6 – history-proximity effect (checked at evals 45 and 70):** if the GP's
  out-of-sample residuals (observed − predicted) correlate with the distance
  to the nearest previously evaluated point (|Spearman ρ| ≥ 0.4 over ≥ 20
  points), adapt the search. If points close to the history come out worse,
  enforce a minimum distance of 0.1 from previous points. If they come out
  better, bias the trust region toward re-sampling close to the incumbent.
- **D7 – robustness:** if the GP fit or acquisition optimization raises
  errors twice in a row, fall back to the project's `TuRBO` class for the
  next 8 evals, then retry the GP.

Every deviation that actually happens will be logged to `changelog.md` with
the evaluation number and the evidence that triggered it.
