# Optimization agent instructions

These are reusable project instructions for optimization campaigns. The user
describes the problem in their terminal conversation; you write its task
definition from that conversation and any clarification answers. A task brief
is an output of planning, not a prerequisite supplied by the user. Never require
the user to create, complete, or select a task Markdown file.

`agent_briefs/task_001.md` and `agent_briefs/task_template.md` are optional input
examples. Do not read or use them unless the user explicitly supplies them as
context. Never inherit an example's budget, domain, or evaluator for a new task.
For repository maintenance, follow the maintenance request; do not launch a
campaign or apply its approval gate to ordinary edits.

Your purpose is to find the best feasible observed objective value within the
agreed budget, in the user's specified optimization direction. Select, switch,
and hybridize methods when justified by the problem and collected evidence;
switching is a strategy, not a goal in itself. Document your reasoning and actual
observations so the research can examine how those decisions affect performance.

## 1. Understand the problem and clarify what matters

Use the user's description and supplied public interface to understand the task.
Accept what is stated or unambiguously implied; do not ask the user to repeat it.
If there is enough information for a valid run, go straight to the task definition.
Ask focused questions only where uncertainty could change the problem, invalidate
execution, or materially improve the optimization plan. Explain why the answer
matters and help the user notice consequential assumptions they may have missed.
Do not turn this into a questionnaire or delay for speculative concerns.

Understand these essentials:

- The objective and whether to minimize or maximize it.
- The decision variables, their types (continuous, discrete, or mixed), allowed
  values or bounds, and any constraints on feasible candidates.
- Whether to select from a fixed pool or generate new points. Continuous-valued
  features in a table do not by themselves permit generating new candidates.
- The evaluation budget, how it is counted, and any stated execution limits.
- The supplied Python script/module and its public usage example: how to submit
  a candidate, receive its objective value, and save the recorded results.

Explore task-specific details only when relevant: for example, which table rows
are eligible, whether repetitions help with noisy measurements, or whether prior
observations may be reused. Use information already supplied. Unknown smoothness
or noise can remain unknown; record it and choose an appropriate strategy. Make
routine operational choices, such as a fresh run name, yourself and include them
in the summary. Do not silently invent facts about the objective or feasible set.

Resolve conflicts with the evaluator or available methods before execution;
do not silently change the problem to fit an optimizer. Require a real Python
evaluator, not an invented objective. Do not spend evaluations during planning,
including through test runs or imports that trigger evaluations. Use public
interface documentation; do not inspect sealed source or hidden outcomes.

## 2. Write the definition and obtain confirmation

Write a concise `task_definitions/<run_id>.md` from the conversation, creating
the directory if needed and using a fresh name. Capture the understood problem,
Python calling example, relevant assumptions/unknowns, and output destination.
Keep reusable agent instructions out of this generated definition.

Mark the draft **Awaiting user confirmation**, link it in a short summary, and
ask the user to confirm. **Wait for explicit confirmation of that definition
before any evaluation.** Update and present the draft again if the user corrects
it; an initial request to optimize is not confirmation of your interpretation.

After confirmation, mark the definition **Confirmed by user** and preserve it.
Include it in `roadmap_initial.md` with a reference to its file and the optimizer
plan, then follow the campaign protocol below. The definition stays outside the
result directory, preserving the four campaign files. Later changes to the
problem require a separate revision and reconfirmation; routine optimizer
choices and switches follow the roadmap/changelog protocol without more approval.

## 3. Available optimization tools

The project provides wrappers with an `ask(n)` / `tell(X, y)` / `best()`
interface. See `src/optimizers/*.py` for examples and docstrings:

- `src.optimizers.random_search.RandomSearch`
- `src.optimizers.sobol_search.SobolSearch`
- `src.optimizers.cma_es.CMAES` (pycma)
- `src.optimizers.differential_evolution.DifferentialEvolution`
- `src.optimizers.gp_bo.GPBO` (BoTorch, analytic LogEI)
- `src.optimizers.turbo.TuRBO` (BoTorch, trust-region BO)
- `src.optimizers.bobyqa.BOBYQA` (Py-BOBYQA)
- `src.optimizers.nelder_mead.NelderMead` (scipy)
- `src.optimizers.lbfgsb.LBFGSB` (scipy, finite-difference gradients)

Check the selected wrapper's domain and direction conventions against the
confirmed task. If a minimization-only optimizer is used for maximization,
transform observations only at the optimizer interface; keep the evaluator's
raw values and direction in the trace unchanged.

You may also use, install, or wrap scipy, Ax, BoTorch, HEBO, Optuna, SMAC3,
cmaes, pycma, PyPop7, EvoX, pymoo, PRIMA, DFO-LS, PyNomad, or other helpful
tools. Use this project's `.venv`, never system/base Python. Check dependency
availability before relying on it; a copied sandbox may need environment setup.

- The PyPI package literally named `prima` is an unrelated placeholder. For
  Powell's solvers, use `pdfo` or Py-BOBYQA (package `Py-BOBYQA`, import
  `pybobyqa`).
- The background-thread wrapper in `src/optimizers/bobyqa.py` can become
  extremely slow past roughly 500 evaluations. Beyond that, prefer another
  method or another way of driving the solver.
- `ioh` requires numpy >= 2 in the supplied setup, which can conflict with
  packages requiring numpy < 2. Check numpy compatibility before installing.
- The original run environment included numpy, scipy, cma, ioh, botorch, torch,
  gpytorch, pybobyqa, and pdfo. Use `.venv/bin/pip install <package>` when needed.

## 4. Required campaign protocol

Use the confirmed output directory wherever `<outdir>` appears below. Do not
change the existing evaluator, optimizer library, baseline runner, budget
enforcement, or trace/metadata format as part of defining a task.

**(a) Before the first evaluation**, write `<outdir>/roadmap_initial.md` with:

- The confirmed problem definition, including assumptions and any explicitly
  unknown properties. Reference the supplied evaluator and permitted data.
- Intended methods, sequence, and approximate allocation of the total budget.
- Why these methods suit the stated problem and any permitted prior data.
  With no prior observations, reason from stated properties alone.
- Explicit quantitative criteria for deviating from the plan, such as a
  plateau over a specified number of evaluations or trust-region collapse.

Once written, do not edit this file again. It is a historical record of the
pre-registered plan. Create an empty `<outdir>/changelog.md` at this point; it
remains empty if there are no deviations. Keep campaign scripts outside the
result directory.

**(b) Run the campaign** through the supplied evaluation interface only. With
the existing harness, call `harness.set_active_optimizer(name)` before the first
`harness.evaluate(x)` and on every method switch. Use short descriptive labels.
Every trace row must come from a real evaluation. Surrogate predictions may
guide selection but must never be recorded as evaluated outcomes. Respect the
confirmed domain, eligible pool, constraints, budget, and stopping rules.

**(c) Whenever you actually deviate from the roadmap** (switch methods, change
a committed hyperparameter, hybridize, restart, etc.), append an entry to
`<outdir>/changelog.md`:

```text
## Evaluation <N>
**Change:** <what changed>
**Reason:** <observed evidence, with specific values/evaluation counts>
```

Append only; never edit or remove earlier entries. Hybridizing and switching
optimizers are encouraged: studying those decisions is central to this project.

**(d) Finish and verify.** On budget exhaustion or early stopping, use the
evaluator's save mechanism (`harness.save()` for the existing harness). Confirm
that `<outdir>/trace.csv` and `<outdir>/meta.json` exist. The harness records
points, raw objective values, running best, optimizer labels, and timing, and
automatically saves when its budget is exhausted. Never hand-construct or edit
these files. Report the best observed result, evaluations used, and limitations.

## 5. Experiment integrity and completion

- Never exceed the confirmed budget or reset its counter by constructing
  another harness instance, restarting a process, or changing the implementation.
- Do not fabricate, estimate, interpolate, or hand-edit evaluation results.
- Treat black-box objectives as sealed. Do not search for, identify, or reverse
  engineer the function; inspect evaluator implementation or files under
  `src/tasks/`; or use baseline results, other agents' campaigns, or sibling
  workspaces. Use only the public interface and permitted observations.
- If the selected output directory already contains a campaign, stop before
  evaluation and arrange a fresh directory/workspace. Do not inspect previous
  traces or roadmaps, reset an old campaign, or overwrite historical files.
- At completion, the run directory contains exactly `roadmap_initial.md`,
  `changelog.md`, `trace.csv`, and `meta.json`. The first two are agent-written;
  the last two are evaluator-written. No additional result files are required.
