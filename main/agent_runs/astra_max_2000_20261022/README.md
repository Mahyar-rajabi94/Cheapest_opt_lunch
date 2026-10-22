# Astra max mathematical suite — 2,000 evaluations

**Partial release: 29/30 campaigns finished.**
1 campaign has an execution failure, retained and explicitly labeled.
A normal early stop is permitted by the canonical prompt; the budget is a maximum.

Each task has **three figures with all nine existing standalone baselines**, in PNG, PDF, and SVG.
Open the links below directly on GitHub, or open [index.html](index.html) locally for the complete gallery.
[summary.csv](summary.csv) contains unrounded agent and baseline regrets.
[statistics.json](statistics.json) and [campaign_status.json](campaign_status.json) record this export's state.

These campaigns use **GPT-6 Astra, max reasoning, one independent run per task**,
source commit `591e3de6a553c026976e3f785eceb8cac8b6690b`, and noise seed 0.
The repository may have newer code; this report retains the exact experiment version.
No baselines were rerun. All 1,890 baseline runs come from `main/results_full` at that commit.

| Task | Status | Calls | Final regret | Figures (PNG) | Decision record |
| --- | --- | ---: | ---: | --- | --- |
| `rosenbrock10` | completed_early | 1859 | 0 | [Final](figures/rosenbrock_d10_i1/01_final_regret.png) · [Convergence](figures/rosenbrock_d10_i1/02_convergence.png) · [Timeline](figures/rosenbrock_d10_i1/03_decision_timeline.png) | [Log](analysis/rosenbrock_d10_i1/changelog.md) |
| `pymoo_g04` | completed_early | 499 | 1.2126911e-06 | [Final](figures/pymoo_g04/01_final_regret.png) · [Convergence](figures/pymoo_g04/02_convergence.png) · [Timeline](figures/pymoo_g04/03_decision_timeline.png) | [Log](analysis/pymoo_g04/changelog.md) |
| `noisy_rosenbrock_gauss5` | complete | 2000 | 2.4067424 | [Final](figures/bbob_noisy_f110_d5_i1/01_final_regret.png) · [Convergence](figures/bbob_noisy_f110_d5_i1/02_convergence.png) · [Timeline](figures/bbob_noisy_f110_d5_i1/03_decision_timeline.png) | [Log](analysis/bbob_noisy_f110_d5_i1/changelog.md) |
| `mixint_rosenbrock10` | completed_early | 1986 | 0 | [Final](figures/bbob_mixint_f8_d10_i1/01_final_regret.png) · [Convergence](figures/bbob_mixint_f8_d10_i1/02_convergence.png) · [Timeline](figures/bbob_mixint_f8_d10_i1/03_decision_timeline.png) | [Log](analysis/bbob_mixint_f8_d10_i1/changelog.md) |
| `rosenbrock20` | complete | 2000 | 23.637874 | [Final](figures/rosenbrock_d20_i1/01_final_regret.png) · [Convergence](figures/rosenbrock_d20_i1/02_convergence.png) · [Timeline](figures/rosenbrock_d20_i1/03_decision_timeline.png) | [Log](analysis/rosenbrock_d20_i1/changelog.md) |
| `rosenbrock40` | complete | 2000 | 74.076607 | [Final](figures/rosenbrock_d40_i1/01_final_regret.png) · [Convergence](figures/rosenbrock_d40_i1/02_convergence.png) · [Timeline](figures/rosenbrock_d40_i1/03_decision_timeline.png) | [Log](analysis/rosenbrock_d40_i1/changelog.md) |
| `rastrigin10` | complete | 2000 | 15.919331 | [Final](figures/rastrigin_d10_i1/01_final_regret.png) · [Convergence](figures/rastrigin_d10_i1/02_convergence.png) · [Timeline](figures/rastrigin_d10_i1/03_decision_timeline.png) | [Log](analysis/rastrigin_d10_i1/changelog.md) |
| `bent_cigar10` | complete | 2000 | 0.00021911866 | [Final](figures/bent_cigar_d10_i1/01_final_regret.png) · [Convergence](figures/bent_cigar_d10_i1/02_convergence.png) · [Timeline](figures/bent_cigar_d10_i1/03_decision_timeline.png) | [Log](analysis/bent_cigar_d10_i1/changelog.md) |
| `sphere10` | completed_early | 261 | 0 | [Final](figures/sphere_d10_i1/01_final_regret.png) · [Convergence](figures/sphere_d10_i1/02_convergence.png) · [Timeline](figures/sphere_d10_i1/03_decision_timeline.png) | [Log](analysis/sphere_d10_i1/changelog.md) |
| `step_ellipsoid10` | complete | 2000 | 0.012701368 | [Final](figures/step_ellipsoid_d10_i1/01_final_regret.png) · [Convergence](figures/step_ellipsoid_d10_i1/02_convergence.png) · [Timeline](figures/step_ellipsoid_d10_i1/03_decision_timeline.png) | [Log](analysis/step_ellipsoid_d10_i1/changelog.md) |
| `sharp_ridge10` | complete | 2000 | 3.7307505e-05 | [Final](figures/sharp_ridge_d10_i1/01_final_regret.png) · [Convergence](figures/sharp_ridge_d10_i1/02_convergence.png) · [Timeline](figures/sharp_ridge_d10_i1/03_decision_timeline.png) | [Log](analysis/sharp_ridge_d10_i1/changelog.md) |
| `weierstrass10` | complete | 2000 | 0.0071653881 | [Final](figures/weierstrass_d10_i1/01_final_regret.png) · [Convergence](figures/weierstrass_d10_i1/02_convergence.png) · [Timeline](figures/weierstrass_d10_i1/03_decision_timeline.png) | [Log](analysis/weierstrass_d10_i1/changelog.md) |
| `sep_rastrigin10` | completed_early | 1964 | 1.7053026e-12 | [Final](figures/sep_rastrigin_d10_i1/01_final_regret.png) · [Convergence](figures/sep_rastrigin_d10_i1/02_convergence.png) · [Timeline](figures/sep_rastrigin_d10_i1/03_decision_timeline.png) | [Log](analysis/sep_rastrigin_d10_i1/changelog.md) |
| `schwefel10` | complete | 2000 | 0.23687679 | [Final](figures/schwefel_d10_i1/01_final_regret.png) · [Convergence](figures/schwefel_d10_i1/02_convergence.png) · [Timeline](figures/schwefel_d10_i1/03_decision_timeline.png) | [Log](analysis/schwefel_d10_i1/changelog.md) |
| `gallagher101_10` | complete | 2000 | 2.0436835 | [Final](figures/gallagher101_d10_i1/01_final_regret.png) · [Convergence](figures/gallagher101_d10_i1/02_convergence.png) · [Timeline](figures/gallagher101_d10_i1/03_decision_timeline.png) | [Log](analysis/gallagher101_d10_i1/changelog.md) |
| `gallagher21_10` | complete | 2000 | 0.69185691 | [Final](figures/gallagher21_d10_i1/01_final_regret.png) · [Convergence](figures/gallagher21_d10_i1/02_convergence.png) · [Timeline](figures/gallagher21_d10_i1/03_decision_timeline.png) | [Log](analysis/gallagher21_d10_i1/changelog.md) |
| `schaffer30` | complete | 2000 | 0.98499301 | [Final](figures/cec2017_f6_d30_b5/01_final_regret.png) · [Convergence](figures/cec2017_f6_d30_b5/02_convergence.png) · [Timeline](figures/cec2017_f6_d30_b5/03_decision_timeline.png) | [Log](analysis/cec2017_f6_d30_b5/changelog.md) |
| `schaffer50` | complete | 2000 | 2.0831809 | [Final](figures/cec2017_f6_d50_b5/01_final_regret.png) · [Convergence](figures/cec2017_f6_d50_b5/02_convergence.png) · [Timeline](figures/cec2017_f6_d50_b5/03_decision_timeline.png) | [Log](analysis/cec2017_f6_d50_b5/changelog.md) |
| `cec_rastrigin30` | complete | 2000 | 200.98071 | [Final](figures/cec2017_f5_d30/01_final_regret.png) · [Convergence](figures/cec2017_f5_d30/02_convergence.png) · [Timeline](figures/cec2017_f5_d30/03_decision_timeline.png) | [Log](analysis/cec2017_f5_d30/changelog.md) |
| `noncont_rastrigin30` | complete | 2000 | 87.091978 | [Final](figures/cec2017_f8_d30/01_final_regret.png) · [Convergence](figures/cec2017_f8_d30/02_convergence.png) · [Timeline](figures/cec2017_f8_d30/03_decision_timeline.png) | [Log](analysis/cec2017_f8_d30/changelog.md) |
| `pymoo_g07` | completed_early | 556 | -7.4591e-10 | [Final](figures/pymoo_g07/01_final_regret.png) · [Convergence](figures/pymoo_g07/02_convergence.png) · [Timeline](figures/pymoo_g07/03_decision_timeline.png) | [Log](analysis/pymoo_g07/changelog.md) |
| `constrained_ellipsoid10` | execution_failure | 735 | 6.1919669 | [Final](figures/bbob_constrained_f20_d10_i1/01_final_regret.png) · [Convergence](figures/bbob_constrained_f20_d10_i1/02_convergence.png) · [Timeline](figures/bbob_constrained_f20_d10_i1/03_decision_timeline.png) | [Log](analysis/bbob_constrained_f20_d10_i1/changelog.md) |
| `constrained_diffpowers10` | complete | 2000 | 5.7784746e-09 | [Final](figures/bbob_constrained_f38_d10_i1/01_final_regret.png) · [Convergence](figures/bbob_constrained_f38_d10_i1/02_convergence.png) · [Timeline](figures/bbob_constrained_f38_d10_i1/03_decision_timeline.png) | [Log](analysis/bbob_constrained_f38_d10_i1/changelog.md) |
| `constrained_rastrigin10` | complete | 2000 | 1.881881e-08 | [Final](figures/bbob_constrained_f44_d10_i1/01_final_regret.png) · [Convergence](figures/bbob_constrained_f44_d10_i1/02_convergence.png) · [Timeline](figures/bbob_constrained_f44_d10_i1/03_decision_timeline.png) | [Log](analysis/bbob_constrained_f44_d10_i1/changelog.md) |
| `noisy_rosenbrock_unif5` | complete | 2000 | 16.732181 | [Final](figures/bbob_noisy_f111_d5_i1/01_final_regret.png) · [Convergence](figures/bbob_noisy_f111_d5_i1/02_convergence.png) · [Timeline](figures/bbob_noisy_f111_d5_i1/03_decision_timeline.png) | [Log](analysis/bbob_noisy_f111_d5_i1/changelog.md) |
| `noisy_rosenbrock_cauchy5` | complete | 2000 | 0.046326004 | [Final](figures/bbob_noisy_f112_d5_i1/01_final_regret.png) · [Convergence](figures/bbob_noisy_f112_d5_i1/02_convergence.png) · [Timeline](figures/bbob_noisy_f112_d5_i1/03_decision_timeline.png) | [Log](analysis/bbob_noisy_f112_d5_i1/changelog.md) |
| `ng_noisy_rosenbrock20` | complete | 2000 | 820.66346 | [Final](figures/nevergrad_noisy_rosenbrock_d20_n0.1/01_final_regret.png) · [Convergence](figures/nevergrad_noisy_rosenbrock_d20_n0.1/02_convergence.png) · [Timeline](figures/nevergrad_noisy_rosenbrock_d20_n0.1/03_decision_timeline.png) | [Log](analysis/nevergrad_noisy_rosenbrock_d20_n0.1/changelog.md) |
| `ng_noisy_cigar20` | complete | 2000 | 9993.6474 | [Final](figures/nevergrad_noisy_cigar_d20_n0.1/01_final_regret.png) · [Convergence](figures/nevergrad_noisy_cigar_d20_n0.1/02_convergence.png) · [Timeline](figures/nevergrad_noisy_cigar_d20_n0.1/03_decision_timeline.png) | [Log](analysis/nevergrad_noisy_cigar_d20_n0.1/changelog.md) |
| `mixint_rastrigin10` | complete | 2000 | 5.0129201 | [Final](figures/bbob_mixint_f15_d10_i1/01_final_regret.png) · [Convergence](figures/bbob_mixint_f15_d10_i1/02_convergence.png) · [Timeline](figures/bbob_mixint_f15_d10_i1/03_decision_timeline.png) | [Log](analysis/bbob_mixint_f15_d10_i1/changelog.md) |
| `mixint_schaffer10` | pending | — | — | — | — |

## Files and verification

- `analysis/<task_name>/`: scored evaluation trace, public metadata, first-block plan,
  complete decision log, and decision intervals (including within-block method segments).
- `figures/<task_name>/`: the three figures in each format and machine-readable `result.json`.
- `provenance/<task_key>/`: runtime/source manifest, 24 preflight isolation checks,
  integrity audit, scoring metadata, public task input, CLI exit status, and agent's final response.
- `provenance/canonical_prompt.txt`: exact historical prompt, preserved as data.
- `cohort.json`, runtime requirement files, and `source_snapshot.tar.gz`: frozen experiment provenance.
- `baseline-sha256.jsonl`: hashes of the 3,780 existing baseline metadata/trace files.
- `verification.json`: independent checks against original journals, scored traces,
  baseline means, decision coverage, and figure existence. Zero verification failures
  means artifact consistency; it does not erase the labeled optimizer execution failure.
- `execution_metrics.csv`: per-campaign elapsed time (including setup) and CLI-reported token use.
- `ARTIFACT_HASHES.json`: SHA-256 hashes of all published payload files except itself.

Original evaluator journals and full CLI event logs remain in the local experiment
archive; they are not duplicated in Git. Absolute machine paths in copied manifests
are replaced with `{EXPERIMENT_REPO}` and `{HOME}`. Numerical traces are unchanged.
The scripts under `provenance/cohort_tools/` document the reporting and verification
implementation; they expect the original `runs/paper-2000-20261022/` layout and raw
journals. They are preserved source, not a standalone entry point for a fresh clone.

## Execution and isolation


`supervise.py` launches the repository's `main/scripts/run_agent.py` in separate
processes. The agent sees only its public workspace, optimizer libraries,
dimension, bounds, noise/constraint metadata, budget, and its observations.
The objective code, task identity, reference optimum, baselines, other campaigns,
and this report are outside its filesystem permissions. Native sandbox checks
run before each campaign. A separate evaluator owns the durable budget and journal.

The agent chooses one block at a time, including its optimizer, hyperparameters,
and block length. The supervisor gives no optimization suggestions and never
retries campaigns to select a better result. A budget is a maximum: an explicit
agent decision to stop early is retained and reported with the actual call count.
Failures are preserved separately from successful early stops.

The launch ceiling began at four, increased to eight, twelve, and sixteen after
checking resources. `watch.py` adjusts the number of new launches according to
available memory, without interrupting active campaigns. `supervision_events.jsonl`
records these changes. Numerical library threads are limited to one in the launcher.
`control.json` controls the launch limit and pause flag; a pause affects new launches.
The supervisor also pauses new launches if tracked source code changes.

These are tested native isolation controls; they are not a formal proof against
operating-system or tool vulnerabilities. `private/isolation_checks.json` and
`private/integrity_audit.json` preserve the actual checks for each campaign.

## Scoring and comparisons

The baseline source is `main/results_full` at the pinned commit. There are 1,890
existing runs across the 30 tasks: random search, Sobol, CMA-ES, differential
evolution, GP-BO, TuRBO, BOBYQA, Nelder–Mead, and L-BFGS-B. Stochastic methods
have ten seeds; the three deterministic local optimizers have one. All have a
2,000-evaluation allowance. No baseline was rerun for this cohort.

Regret is the scored incumbent value minus the task's stored reference optimum.
Baseline convergence shows arithmetic means across seeds. Early-stopped baseline
incumbents are carried forward to the allowed horizon. Values at or below 1e-12
are drawn at 1e-12 on log axes; raw values remain in the numerical outputs.

For noisy tasks, the incumbent is selected by its lowest observed objective, but
scored using that point's noise-free objective. The true values remain private
throughout optimization. The plotting code combines public observations with
`private/scored_trace.csv` only in the trusted `analysis/` directory, never in an
agent workspace. A noisy task's scored trajectory need not be monotone.

The constrained Ellipsoid campaign has an **execution failure**, distinct from a
normal early stop. At 735 calls, the agent's adapter deadlocked while starting a
second COBYQA solver with the first suspended; the suspended call held SciPy's
process-wide lock. Its attempted process interrupt was denied by the sandbox.
The saved 735 observations remain valid and are plotted, with the execution
failure labeled. No retry or supervisor intervention replaced this outcome.
See `annotations.json` and the campaign's final decision for the evidence.

For constrained tasks, an incumbent must satisfy the repository's shared
feasibility rule (total positive constraint violation at most 1e-10). Regret is
undefined until a feasible point is found. Baseline means skip seeds without a
feasible incumbent, and final plots report the feasible-seed count. The agent
receives objective and constraint values separately; existing baseline wrappers
use a fixed scalar penalty. This interface difference matters when interpreting
the constrained comparisons.

The pinned public prompt says each constraint must be nonpositive with a 1e-10
tolerance; it does not specify that the tolerance applies to the *sum* of positive
violations. G04's agent interpreted it per constraint and flagged two rejected
candidates. Inspection confirms their summed violations were approximately
1.60e-10 and 2.00e-10, so both correctly fail the shared evaluator/baseline rule.
The saved G04 incumbent and reported regret follow that rule. The agent's claim
that feasibility was strictly zero-tolerance is not supported by this inspection.
This prompt ambiguity is retained and disclosed for the pinned cohort; neither
the campaign nor its observations were altered after the fact.

There is one agent run per task, versus multiple seeds for stochastic baselines.
These are individual agent outcomes, not estimates over repeated agent runs.
The agent and baseline NumPy/SciPy versions match, but platform and backend
differences may still affect numerical outcomes.

The canonical agent may use methods beyond the nine standalone baselines,
including SciPy SLSQP, SciPy COBYQA, and campaign-local surrogate methods.
For example, G04 selected SLSQP only after its initial 64-point Sobol block.
There is no standalone SLSQP baseline in the supplied comparison set. These
comparisons include differences in available methods and hyperparameters;
they do not isolate the benefit of LLM method selection from those differences.

Decision boundaries use the evaluator's timestamped document snapshots, including
continuations of the same method. A logged block can contain multiple method
labels, such as repeated noise probes followed by CMA-ES. Those labels appear
together under the single decision; their exact intervals are retained in
`method_segments` in `analysis/*/decisions.json`. Internal label changes do not
count as additional LLM decisions. Each entry retains the decision text and
exact evaluation interval.

