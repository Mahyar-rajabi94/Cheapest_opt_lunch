# Tasks

What each registered task (`src/tasks/registry.py`) tests, and whether its baselines show it. Adding a task: `ADDING_TASKS.md`.

This file names task identities. Keep it in `main/`; never copy it into a sandbox.

## Dimensions

| Dimension | Tasks |
| --- | --- |
| Baseline / sanity | `sphere10` |
| Ill-conditioning | `bent_cigar10`, `rosenbrock*`, `constrained_ellipsoid10` |
| Dimension scaling (10/20/40) | `rosenbrock10`, `rosenbrock20`, `rosenbrock40` |
| Multimodality | `rastrigin10`, `cec_rastrigin30`, `weierstrass10`, `constrained_rastrigin10` |
| Plateaus / zero gradient | `step_ellipsoid10`, `noncont_rastrigin30` |
| Nondifferentiability | `sharp_ridge10` |
| Wide box ([-100, 100]), D ≥ 30 | `cec_rastrigin30`, `noncont_rastrigin30`, `schaffer30`, `schaffer50` |
| Inequality constraints | `pymoo_g04`, `pymoo_g07`, `constrained_*` |
| Finding any feasible point | `pymoo_g07` |
| Non-uniform box | `pymoo_g04`, `pymoo_g07` |
| Observation noise | `noisy_rosenbrock_*`, `ng_noisy_*` |

## Per Task

Folder: `results_linux/<folder>/`. "Random" is the median random-search regret over 500 uniform points; "center" is the regret at the box center, where deterministic optimizers (`lbfgsb`, `nelder_mead`, `bobyqa`) start. Winners are median regret over seeds of the baselines in `results_linux/`, at 100 and 2000 evaluations (GP-BO, TuRBO, BOBYQA capped at 100).

### Unconstrained, Noiseless

| Key | Folder | Source | D | Box | Tests | Winner @100 | Winner @2000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sphere10` | `sphere_d10_i1` | BBOB f1 | 10 | [-5, 5] | Separable convex bowl; every local method should solve it. | BOBYQA (0) | L-BFGS-B, CMA-ES (~1e-11) |
| `bent_cigar10` | `bent_cigar_d10_i1` | BBOB f12 | 10 | [-5, 5] | Condition 1e6: one gentle, nine steep directions; gradient methods win. | L-BFGS-B (44) | L-BFGS-B (9e-6) |
| `rosenbrock10` | `rosenbrock_d10_i1` | BBOB f8 | 10 | [-5, 5] | Curved, ill-conditioned valley; unimodal. | L-BFGS-B (76) | L-BFGS-B (1e-7) |
| `rosenbrock20` | `rosenbrock_d20_i1` | BBOB f8 | 20 | [-5, 5] | Same valley at 2x dimension. | L-BFGS-B (806) | CMA-ES (18) |
| `rosenbrock40` | `rosenbrock_d40_i1` | BBOB f8 | 40 | [-5, 5] | Same valley at 4x dimension; BOBYQA's lead at small budgets grows with D. | BOBYQA (1.3e4) | L-BFGS-B (5.5) |
| `rastrigin10` | `rastrigin_d10_i1` | BBOB f15 | 10 | [-5, 5] | Rotated, regular multimodality; local methods trap, BO leads early. | GP-BO (98) | CMA-ES (15) |
| `weierstrass10` | `weierstrass_d10_i1` | BBOB f16 | 10 | [-5, 5] | Rugged, repetitive multimodality; center is worse than a random point (104 vs 78), so local methods from the center fail. | TuRBO (21) ≈ random (22) | CMA-ES (6.7) |
| `step_ellipsoid10` | `step_ellipsoid_d10_i1` | BBOB f7 | 10 | [-5, 5] | Plateaus: L-BFGS-B and Nelder–Mead never leave the center (745). | GP-BO (13) | CMA-ES (0.76) |
| `sharp_ridge10` | `sharp_ridge_d10_i1` | BBOB f13 | 10 | [-5, 5] | Nondifferentiable ridge; in practice finite-difference L-BFGS-B still wins, ranking ≈ `bent_cigar10`. | BOBYQA (23) | L-BFGS-B (0.22) |
| `cec_rastrigin30` | `cec2017_f5_d30` | CEC-2017 F5 | 30 | [-100, 100] | Rastrigin at D=30 on a wide box. | L-BFGS-B ≈ GP-BO (~365) | CMA-ES (233) |
| `noncont_rastrigin30` | `cec2017_f8_d30` | CEC-2017 F8 | 30 | [-100, 100] | F5 with rounded steps: plateaus on top of multimodality; stalls L-BFGS-B (518 vs 303 on F5). | GP-BO (361) | CMA-ES (236) |
| `schaffer30` | `cec2017_f6_d30` | CEC-2017 F6 | 30 | [-100, 100] | Expanded Schaffer F6, nonseparable needle-like multimodality. | all ≈ random (14.0–14.9) | all ≈ random (13.7–14.8) |
| `schaffer50` | `cec2017_f6_d50` | CEC-2017 F6 | 50 | [-100, 100] | Same at D=50. | all ≈ random (23.9–25.3) | all ≈ random (23.4–25.3) |

### Constrained

Feasibility rate is the fraction of 500 uniform points that satisfy every constraint. The box center is infeasible on every constrained task.

| Key | Folder | Source | D | Constraints | Feasible | Tests | Winner @100 | Winner @2000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pymoo_g04` | `pymoo_g04` | G-suite G04 | 5 | 6 nonlinear, 2 active | 27% | Optimum on the constraint boundary; non-uniform box [27, 102]. | Nelder–Mead (128) | CMA-ES (0.006), Nelder–Mead (0.013) |
| `pymoo_g07` | `pymoo_g07` | G-suite G07 | 10 | 8, 6 active | 0% | Feasible region is tiny: random search, CMA-ES, DE find no feasible point in 100 evaluations; random search finds none in 2000, so the global metric excludes this task. | BOBYQA (1.6e3, 10/10 seeds feasible) | CMA-ES (1.2) |
| `constrained_ellipsoid10` | `bbob_constrained_f20_d10_i1` | bbob-constrained f20 | 10 | 3 linear | 18% | Ill-conditioned objective with linear constraints. | TuRBO (1.0e3) | CMA-ES (44) |
| `constrained_diffpowers10` | `bbob_constrained_f38_d10_i1` | bbob-constrained f38 | 10 | 3 linear | 8% | Flat objective near the optimum. | CMA-ES (1.2e3) | CMA-ES (0.13) |
| `constrained_rastrigin10` | `bbob_constrained_f44_d10_i1` | bbob-constrained f44 | 10 | 3 linear | 17% | Multimodal objective with linear constraints. | CMA-ES (1.6e3) | CMA-ES (266) |

### Noisy

Optimizers see the observed `y`; scoring uses the noise-free value of the best observed point. Rank correlation is Spearman between observed and noise-free values over 500 uniform points.

| Key | Folder | Source | D | Noise | Rank corr. | Tests | Winner @100 | Winner @2000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `noisy_rosenbrock_gauss5` | `bbob_noisy_f110_d5_i1` | bbob-noisy f110 | 5 | Gaussian, multiplicative, vanishes at the optimum | 0.72 | Moderate noise. | CMA-ES (467) | CMA-ES (4.7) |
| `noisy_rosenbrock_unif5` | `bbob_noisy_f111_d5_i1` | bbob-noisy f111 | 5 | Uniform, severe, mostly upward | 0.24 | Severe noise; observed values at the center span 2 to 1e6. | BOBYQA (485) | CMA-ES (603) |
| `noisy_rosenbrock_cauchy5` | `bbob_noisy_f112_d5_i1` | bbob-noisy f112 | 5 | ~+1000 offset, rare Cauchy outliers | 1.00 | In practice near-noiseless: SD at the center is 1.0 over 200 calls. Behaves like a 5-D noiseless Rosenbrock. | BOBYQA (9.0) | CMA-ES (1.2) |
| `ng_noisy_rosenbrock20` | `nevergrad_noisy_rosenbrock_d20` | nevergrad, noise level 10 | 20 | 10·N(0,1)·(f(x+N(0,I)) − f(x)), symmetric | 0.11 | Noise swamps the signal (SD at center 1.7e5 vs true regret 4.4e3). | Nelder–Mead (4.4e3 = center) | Nelder–Mead (center) |
| `ng_noisy_cigar20` | `nevergrad_noisy_cigar_d20` | nevergrad, noise level 10 | 20 | Same formula | 0.12 | Same; SD at center 2.0e8 vs true regret 1.9e7. | Nelder–Mead (1.9e7 = center) | L-BFGS-B, Nelder–Mead (center) |

COCO's noise stream repeats by call index across seeds, so seed spread on the `noisy_rosenbrock_*` tasks understates the true variance.

## Gaps

Tasks that add little beyond the set as it stands, at the 100/2000 budgets:

| Task(s) | Issue |
| --- | --- |
| `schaffer30`, `schaffer50` | No signal: every method, including random search, ends within ~1 of the center and random-sample regret. Every method scores r ≈ 1 in the global metric. `schaffer50` duplicates `schaffer30`. |
| `ng_noisy_rosenbrock20`, `ng_noisy_cigar20` | Degenerate: observed and true values are nearly uncorrelated, so the "winner" is whoever never leaves the start point (Nelder–Mead's simplex collapses at the center; L-BFGS-B stops). A method that explores selects a noise outlier, e.g. CMA-ES seed 0 picks a point observed at −9.6e6 whose true regret is 4.3e5. Ranks reward not moving, not noise handling. |
| `noisy_rosenbrock_cauchy5` | Not noisy in practice (rank corr. 1.00); repeats noiseless Rosenbrock at D=5. |
| `constrained_*` (3 tasks) | Same winner (CMA-ES at 2000) and similar ranks on all three; the objective varies, but the baselines do not separate them. |
| `sharp_ridge10` | Ranking matches `bent_cigar10`; the nondifferentiability does not change which method wins. |

Tasks that each add a distinct behavior: `step_ellipsoid10` and `noncont_rastrigin30` (gradient methods stall), `weierstrass10` (start point worse than random), `pymoo_g07` (feasibility search), `pymoo_g04` (boundary optimum, Nelder–Mead competitive), `noisy_rosenbrock_unif5` (severe noise flips the B=100 winner to BOBYQA), the `rosenbrock*` scaling series, and BO-favoring early budgets on `rastrigin10` and `step_ellipsoid10`.
