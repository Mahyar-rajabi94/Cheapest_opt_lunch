# CLAUDE.md — cheaplunch optimization agent (main/)

Context for Claude Code sessions. Start sessions from this `main/` folder.

> **Why this file is in `main/` and not the repo root:** Claude Code also
> loads CLAUDE.md from parent folders. A root-level file would be loaded by
> blind agent runs in `task_001_sandbox_*` and leak the project context
> (including what Task 001 is). Never add a CLAUDE.md at the repo root or in
> the sandbox folders.

## Project
We are building an **adaptive optimization agent** that picks, switches and
combines black-box optimizers for a new problem under an evaluation budget
(the "no free lunch" motivation: no single optimizer wins everywhere).

- Team split: **Mohamad owns the agent.** Teammates own the benchmark suite
  and problem set. The agent's needs from them are in `design/agent_design.md` §8.
- Repo: github.com/Mahyar-rajabi94/Cheapest_opt_lunch. Active branch: `agent-v2`.

## Key docs (read these first)
- `design/agent_design.md`: v2 architecture (round loop, probes, racing,
  archive + warm starts, problem intake).
- `design/roadmap.md`: phased plan.
- `design/v2_status.md`: **current state, dev results, next steps.**
- `design/notes/`: earlier review and decision log (from the claude.ai project).

## Code map
```
agent/
  archive.py        all evaluations; incumbent, top_k, local_scale, diverse_starts
  warmstart.py      make_optimizer(method, bounds, archive, start=...): warm starts for all 9 optimizers
  controller.py     round loop, executes Actions, builds StateReport, budget, fallback
  schema.py         StateReport / Action (pydantic); action menu generated from here
  intake.py         problem intake + defaults (round size, probe budget, allowed methods)
  probes.py         landscape probes P1-P5
  racing.py         successive-halving race
  evaluator.py      budget-enforced evaluation, trace.csv/meta.json
  decision_log.py   decisions.jsonl
  dev_tasks.py      DEV-ONLY BBOB smoke tasks (not official results)
  playbook/         markdown strategy knowledge (ROLE.md + skills/) for the LLM
  playbook_loader.py system prompt assembly, version hash, leak check
  policies/         scripted.py (rules), fixed.py (single method), llm.py (Anthropic/OpenAI)
src/optimizers/     9 ask/tell wrappers (random, sobol, cma_es, de, gp_bo, turbo, bobyqa, nelder_mead, lbfgsb)
scripts/            run_agent.py, eval_dev.py, check_playbook.py, run_campaign.py
tests/              pytest
```

## Commands
```
python -m pytest tests                       # 41 tests, ~40 s
python scripts/run_agent.py --task task001 --policy scripted --seed 0
python scripts/run_agent.py --task task001 --policy llm --provider anthropic   # or openai; needs API key
python scripts/run_agent.py --task dev_f15_d10 --policy llm --no-playbook      # ablation
python scripts/eval_dev.py --seeds 5 --jobs 2  # dev comparison table
python scripts/check_playbook.py               # must pass before committing playbook changes
```
Default LLM models are `claude-sonnet-4-6` / `gpt-5`. Override with `--model`
or `AGENT_ANTHROPIC_MODEL` / `AGENT_OPENAI_MODEL`.

## Rules
- **No benchmark identities in the playbook or in anything the strategist
  sees** (function names, suite names, optimum values). `check_playbook.py`
  and `tests/test_controller.py::test_report_never_contains_task_identity` enforce this.
- **Dev tasks are for development only.** Never report them as results;
  official evaluation uses the benchmark team's suite.
- **Never lose progress on a switch:** every optimizer change goes through
  `make_optimizer(...)` with the archive (warm start), never a cold midpoint start.
- **The controller is deterministic and owns all numerics.** The LLM only maps
  StateReport → Action. Invalid actions fall back; runs must never crash.
- Keep the trace format (`trace.csv`, `meta.json`) compatible with `scripts/run_campaign.py`.
- Commit on feature branches, not `master`.

## Known issues / next steps (see design/v2_status.md)
1. LLM policy is only tested offline. Run it live with both providers, ± playbook, and compare to scripted.
2. Probe P4 is noisy (the landscape guess flips between seeds) and reads ill-conditioning as multi-basin. Use P3 R² to tell them apart; more or longer P4 runs.
3. No pinned `requirements.txt` yet.
4. `results/` baselines for BOBYQA and TuRBO came from buggy versions (BOBYQA maxfun memory bug, TuRBO sign bug). Rerun them.
5. Deferred: P6/P7 probes, UCB racing, racing/intake skills, random policy.
