# Agentoptimizer

An LLM chooses optimization methods, hyperparameters, and evaluation block lengths
for a black-box problem. It plans **one block at a time**, inspects its observations,
and decides again. The total evaluation budget is enforced by a separate evaluator.

[AGENTS.md](AGENTS.md) is the canonical agent prompt used for the iterative Astra
experiments. [CLAUDE.md](CLAUDE.md) imports it; there is only one maintained prompt.
The automated isolated runner below drives either Codex CLI or Claude Code CLI
(`--agent codex|claude`); both use Codex's native sandbox for every agent command.

[Published Astra max results: 30 mathematical tasks, 2,000-evaluation allowance](main/agent_runs/astra_max_2000_20261022/README.md).

## Install once

Use Python 3.11+ and an installed `codex` CLI, authenticated for `--agent codex`.
For `--agent claude`, also install Claude Code CLI and sign in (`claude auth login`).
Run these commands from the repository root. Keep two Python environments: the agent gets optimizer
libraries; only the private evaluator gets objective backends such as `ioh`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r main/requirements.txt
python3.11 -m venv .agent-venv
.agent-venv/bin/python -m pip install -r main/requirements-agent.txt
codex login
```

The native runner has been checked on macOS with Codex CLI
`0.162.0-alpha.17.2` and `0.163.0`; `--agent claude` with Claude Code CLI `2.1.308`
(sandboxed by Codex `0.163.0`). It also has a Linux execution path, which needs validation
on the target machine. It fails before inference if its isolation checks fail.
Other CLI versions must support the named permission profiles used by the runner.

## Run the agent

First check setup and isolation. This spends **zero objective evaluations** and
makes **no model call**:

```bash
.venv/bin/python main/scripts/run_agent.py \
  --task step_ellipsoid10 --budget 100 --check-only
```

Then start a fresh campaign:

```bash
.venv/bin/python main/scripts/run_agent.py \
  --task step_ellipsoid10 --budget 100 \
  --model gpt-6-astra --effort max
```

The defaults are `gpt-6-astra` and `max`, so both flags can be omitted.
Use `--effort xhigh` to retain the reasoning setting used in earlier experiments.

`--budget` is the number of **objective evaluations**, not model tokens or optimizer
iterations. It is supplied in the public task input and enforced privately.
Initialization, repeated points, finite differences and line searches all count.
The agent chooses the next block's method, settings and length; continuations also
require a new decision. It can preserve a live optimizer process between blocks.

To run Claude instead, add `--agent claude` (default model `claude-opus-5-5`,
same `max` default effort; `--claude` overrides the CLI path). Claude Code runs headless
with no built-in tools, settings, memory, CLAUDE.md, skills or claude.ai connectors:
its only tools are the same sandboxed `run_command`/`write_stdin` MCP executor.

```bash
.venv/bin/python main/scripts/run_agent.py \
  --task step_ellipsoid10 --budget 100 --agent claude
```

For the other two recent experiments, use `--task sharp_ridge10` or
`--task weierstrass10`. Every invocation creates a new opaque `runs/run-<id>/`;
there is no resume/reset option. Runs may be launched from separate terminals in
parallel. `--runs-dir`, `--agent-python` and `--codex` override local paths.

The native runner accepts every task in [the task registry](main/src/tasks/registry.py).
The agent sees the same box as the baselines: discrete, categorical and conditional
structure stays hidden inside the objective. Constrained tasks return `(y, g)`,
feasible iff every `g <= 0`. Noisy tasks set `noisy` and are reseeded with `--seed`
(default 0); the agent sees only observations, and the private journal keeps the
noise-free values. See [TASKS.md](main/TASKS.md) and [ADDING_TASKS.md](main/ADDING_TASKS.md).

## What the agent can access

```text
Fresh Codex CLI context + canonical prompt + public task description
    -> native OS sandbox: optimizer libraries and this run's workspace
    -> public harness.evaluate(x): request file containing x
    -> separate trusted evaluator: objective, durable budget and journal
    -> numeric response returned to the agent
```

No VM is required. A fresh conversation or a different working directory alone
is insufficient: the runner applies explicit deny-by-default filesystem rules.
Agent commands can read their public workspace and optimizer runtime, and write
scripts in `work/`, temporary files, requests and decision logs. They cannot read
the repository, objective implementation, private evaluator state, previous runs,
or siblings. Network access, browser/app tools, memories, personal configuration,
and permission escalation are disabled for the campaign. The only execution tools
are the trusted `workspace.run_command` and `workspace.write_stdin` MCP tools,
which enforce the native sandbox on every command.

The CLI retains the credentials needed for model inference; those credentials
are not exposed to agent commands. The agent sees bounds, dimension, direction,
noise status, budget, and its own observations. It does not receive task identity,
optimum, baseline results or this README.

Preflight attempts forbidden reads/writes, a symlink escape, network access and
process inspection on macOS. Checks are recorded before the first evaluation.
These are tested isolation controls, not a formal guarantee against OS or tool
vulnerabilities. Never run the optimization agent directly in the repository.

## Outputs and supervision

The launcher prints the run directory immediately. While it runs, inspect its
private CLI event log from your supervisor terminal; do not send benchmark hints
back to the agent.

```text
runs/run-<id>/
  workspace/
    AGENTS.md                       generated copy of the canonical prompt
    PUBLIC_EVALUATOR.md              public problem and actual budget
    src/agent_harness/harness.py     transport client, no objective implementation
    work/                           agent-written scripts
    results/agent_run/
      roadmap_initial.md            FIRST BLOCK ONLY; immutable after recording
      changelog.md                  every subsequent decision, including continuations
      trace.csv                     actual observations, written by evaluator
      meta.json                     public run state, written by evaluator
  private/
    manifest.json                   source/prompt hashes, model and package versions
    isolation_checks.json           preflight results
    evaluations.sqlite              authoritative evaluations and decision snapshots
    cli_events.jsonl                CLI tool calls and messages (Codex or Claude stream-json)
    agent_final.txt                 final agent response
    integrity_audit.json            journal/artifact consistency checks
    scoring.json                   task identity and optimum, for offline analysis
    scored_trace.csv               y_best_so_far in the baseline convention (noise-free, feasible)
```

The historical filename `roadmap_initial.md` remains for compatibility; it is not
a full campaign roadmap. The evaluator snapshots decisions before evaluations,
rejects edits to the initial record and requires an append-only changelog.
Reconstructing the client does not reset the evaluation counter. On a crash or an
unknown evaluation outcome, preserve the run directory and inspect the private
journal; do not retry that campaign as though it had zero evaluations.

## Baselines and figures

Run all standalone optimizers at the same budget into a **new** results directory:

```bash
.venv/bin/python main/scripts/run_sweep.py \
  --task step_ellipsoid10 --budget 100 --seeds 10 --workers 4 \
  --outdir runs/baselines-100
```

Stochastic baselines use ten seeds; deterministic methods use one. Current sweeps
use the requested budget for every method. Historical `main/results_linux/` runs
may have GP-BO, TuRBO and BOBYQA capped at 100; inspect their metadata. The newest
full-budget benchmark is also preserved under `main/`. Numerical results can
vary by platform and library version; use matched environments for formal comparisons.

After the agent finishes, replace `<id>` below with its printed run ID:

```bash
.agent-venv/bin/python main/scripts/make_comparison_plots.py \
  --results runs/baselines-100 --task step_ellipsoid_d10_i1 \
  --agent-run runs/run-<id>/workspace/results/agent_run \
  --outdir runs/run-<id>/figures
```

This produces baseline convergence, a final-regret comparison, and an agent
method/decision timeline. The supplied agent's budget sets the comparison horizon;
longer baseline traces are truncated and early-stopped incumbents are carried
forward within their allowed budget. Historical prefix comparisons are not fresh
baseline reruns configured for the smaller budget. Scoring happens outside the
agent's workspace, using the private optimum from baseline metadata.

## Repository layout and checks

- `AGENTS.md`, `CLAUDE.md`: one canonical agent and its Claude import.
- `main/src/tasks/`: private objective implementations and registry.
- `main/src/optimizers/`: shared optimizer wrappers.
- `main/src/agent_harness/harness.py`: public evaluation client.
- `main/src/blind_runner/`: native execution, private evaluator and orchestration.
- `main/scripts/`: agent runner, baseline runner and plotting entry points.
- `archive_results/`: preserved historical agent artifacts; old source is in Git history.
- `task_005_baseline/`, `poly_results/`: polymer data/model utilities and existing figures.
- `runs/`: ignored generated workspaces, logs and new results.

The numbered local harnesses, copied sandbox implementations and old planning
briefs have been retired. A new supported task reuses the same public client.

The historical PSA campaign in `main/scripts/campaign_psa_v10_run1.py` retains its
original in-process harness in `_psa_v10_legacy_harness.py` for reproducibility.
Use `run_agent.py` for new isolated agent campaigns on supported tasks.

```bash
.venv/bin/python -m unittest discover -s main/tests -v
```

The integrity tests use fake objectives to check budget exhaustion, duplicate
request handling, crash recovery, configuration immutability and prospective logs.
