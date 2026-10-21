---
name: probing
description: Use when deciding whether to probe, and when interpreting probe features (P1-P5) to choose a strategy.
---

# Probing the landscape

## When to probe
- First round on any problem with budget ≥ 20·dim: `probe` with the default
  set and budget.
- Skip, or use `probe_ids: ["P1", "P2", "P3"]` only, when the budget is
  under 20·dim or evaluations are expensive. Each probe evaluation is then
  too valuable.
- Re-probing later is rarely worth it. Read progress curves instead.

## What each probe measures
| Probe | Feature | Reading |
|---|---|---|
| P1 | `log10_range`, `skew`, `frac_extreme` | Spread of values over the box. Large range and high skew mean a few huge values and a narrow good region. |
| P2 | `fdc_spearman` | Correlation between value and distance to the best sample. High (> 0.5) means values improve steadily toward the best region (global structure). Near 0 means no global trend. |
| P2 | `top10_spread_vs_uniform` | How spread out the best 10% of samples are, relative to random points. Under 0.5: good points cluster in one region. Near 1: good points are everywhere (many basins or flat). |
| P3 | `r2_quad_sep_rank`, `r2_quad_full_rank` | How well a quadratic explains the ranking of values. Above 0.85: bowl-shaped at box scale. Under 0.5: no simple global shape. |
| P4 | `rel_gain_median` | How much short local runs improved from their starts (1 = reached the best endpoint value). |
| P4 | `endpoint_y_spread` | How different the local runs' end values are, relative to their total improvement. Small (< 0.3) means they reached the same level: one basin. Large (> 0.6) means different basins. |
| P4 | `endpoint_contraction` | Distance between endpoints divided by distance between starts. Small means the runs converge toward one point. |
| P5 | `cosine_min` | Agreement of finite-difference gradients at two step sizes. Above 0.9: smooth and noise-free at small scales, so gradient methods are reliable. Under 0.5: rough or noisy, so avoid `lbfgsb`. |

## Reading the combination
- **Smooth single basin:** P5 cosine > 0.9, P4 gain > 0.5, P4 y_spread < 0.3.
  → `polish` with `lbfgsb` (or `bobyqa`) from the incumbent now. Population
  methods only waste budget here.
- **Probably one basin but ill-conditioned:** P5 > 0.9, P3 quadratic
  R² > 0.8, but P4 shows modest gain (< 0.6) and y_spread 0.3–0.7. Short
  local runs simply didn't finish on a stretched valley. → `cma_es` from the
  incumbent (it learns the scaling), then polish when it converges. Or race
  `cma_es` vs `bobyqa`.
- **Several basins with global structure:** P4 y_spread > 0.6 and P2 fdc > 0.4.
  → `cma_es` with restarts (growing population), polishing each basin's
  best. See stall-recovery.
- **Weak structure:** P4 y_spread > 0.6, P2 fdc < 0.35, P3 R² < 0.5.
  → Large-population `cma_es` (`popsize_factor` 2–4) from `center`, or race
  `cma_es`, `de`, `turbo`. Expect several restarts.
- **Rough or noisy:** P5 cosine < 0.5. → No `lbfgsb`. Use `cma_es` or `bobyqa`.

Details and the evidence behind these thresholds: `references/feature_thresholds.md`.
