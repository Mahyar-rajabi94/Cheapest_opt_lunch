# Agentoptimizer

Research project: an LLM agent that selects, switches and hybridizes
black-box optimizers under a budget, instead of committing to a single
fixed method.

## Structure

```
Agentoptimizer_2/
├── main/                 the actual project: optimizer library, tasks,
│                         docs, baseline campaign results, and generated
│                         comparison figures (main/figures/)
├── sandbox/              reusable blind-test sandbox — the current way
│                         to run an agent on any black-box problem
├── Agent.md              task-agnostic system prompt for an agent run in
│                         sandbox/ (a copy also lives in sandbox/Agent.md,
│                         since an agent scoped to that folder can't see
│                         the repo root)
└── archive_results/      superseded one-sandbox-per-task copies
                          (task_001_sandbox_claud/, task_002_sandbox/,
                          etc.), kept for the historical record only
```

**`main/`** is everything: the optimizer library (`src/optimizers/`),
task wrappers (`src/tasks/`), the campaign runner (`scripts/`), all
literature surveys and the optimizer/benchmark catalogues (`docs/`), the
baseline results already generated (`results/`), and the comparison plots
generated from them (`figures/`, via `scripts/make_comparison_plots*.py`).

**`sandbox/`** is a single reusable blind-test environment — edit, clear,
and reuse the same folder for each new problem, rather than making a new
per-task copy. Before a run:

1. Edit `sandbox/objective.py` — set `DIM`, `BOUNDS`, `DIRECTION`,
   `NOISY`, and implement `fn(x)` for the problem at hand.
2. Clear `sandbox/results/agent_run/` if it was just used for a different
   problem.
3. Start a **brand-new agent session scoped to `sandbox/` only** — never
   open an editor or terminal on `main/` or this repo's root in the same
   session (see below for why).

Then give that fresh session `sandbox/Agent.md` as its system prompt,
plus a short task-specific prompt stating only the generic parameters
(dimension, bounds, direction, noise, budget) — never the function's
identity. `sandbox/README.md` has the full checklist and an example
prompt to copy.

**`archive_results/`** holds the old one-sandbox-per-task copies
(`task_001_sandbox_claud/`, `task_001_sandbox_openai/`,
`task_002_sandbox/`, `task_003_sandbox/`, `task_004_sandbox/`) built
before `sandbox/` + `Agent.md` replaced that pattern with one reusable
folder. They're kept only for reference — nothing in the current workflow
depends on them.

## Task code → true identity

The blind-test sandbox (and everything handed to an agent) must refer to
a task only by generic, non-identifying parameters — never a name like
"Rosenbrock". This mapping is exactly what must never leak into
`sandbox/` or an agent's prompt; it exists only here, for the human
reading results from `main/`.

| Task code  | Function (BBOB id)     | dim | True optimum (`f_opt`) |
|------------|-------------------------|-----|-------------------------|
| `task_001` | Rosenbrock (f8)         | 10  | 149.15                  |
| `task_002` | Rastrigin (f15)         | 10  | 1000.0                  |
| `task_003` | Bent Cigar (f12)        | 10  | -621.11                 |
| `task_004` | Sphere (f1)             | 10  | 79.48                   |

All four use BBOB instance 1. `f_opt` values are specific to that instance
(a different instance would shift them).

## Before running an agent on `sandbox/`, two things matter more than anything else

1. **Start a brand-new conversation, not a continuation of any chat that
   has discussed this project.** Any session with that context already
   knows which task is which. The blind framing only works in a
   conversation that has never seen that discussion.

2. **Scope the agent to `sandbox/` only — never open an editor or
   terminal on `main/`, `archive_results/`, or this repo's root in the
   same session.** If the agent's file tools can reach `main/` as a
   sibling directory, the same leak happens through the filesystem
   instead of through conversation history. Likewise, keep the true
   function's identity out of `sandbox/objective.py`'s comments/names and
   out of the task-specific prompt — state only dimension, bounds,
   direction, noise, and budget.

With both respected, see `sandbox/README.md` for the exact steps and the
example first message to give the fresh agent.

Results land in `sandbox/results/agent_run/`. Scoring them against the
true optimum, and generating comparison plots against the baselines in
`main/results/<task>_d10_i1/`, has to happen from `main/` afterward —
copy the finished run into that task's results folder (e.g. as
`claude_agent_run/`), then adapt/run one of
`main/scripts/make_comparison_plots*.py` to write figures into
`main/figures/<task>_d10_i1/`.
