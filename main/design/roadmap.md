# Agent v2 — Implementation Roadmap

**Branch:** `agent-v2` (from `master`, with `fix-turbo-sign` and `agent-design-v2` merged in)
**Design:** [`agent_design.md`](agent_design.md)
**Owner:** Mohamad Moosavi (agent). The benchmark suite and problem set are owned by teammates (see design §8).

Principle: **build from the bottom up, and keep every layer testable without an LLM.**
The LLM strategist comes last. By then the controller, probes and racing
already work end to end under a scripted policy, so the LLM's added value
can be measured.

---

## Target code layout

```
main/
  agent/                      # new package: everything agent v2
    archive.py                # Archive (design §6.1)
    warmstart.py              # from_archive() adapters for the 9 optimizers (§6.2)
    intake.py                 # Intake schema + defaults (§7)
    schema.py                 # StateReport, Action (+ JSON schema export) (§3.2–3.3)
    controller.py             # round loop, action execution, budget (§3)
    decision_log.py           # decisions.jsonl writer (§3.4)
    probes/                   # P1–P7 (§4)
    racing.py                 # successive halving, UCB (§5)
    policies/
      scripted.py             # rule-based baseline (no LLM)
      random_policy.py        # random valid actions (ablation floor)
      llm.py                  # LLM strategist (structured output)
    playbook/                 # markdown strategy knowledge (no task identities)
      ROLE.md
      skills/<topic>/SKILL.md (+ references/)
  src/optimizers/             # existing wrappers (+ state exposure, warm start)
  scripts/run_agent.py        # CLI: run agent on a task with a policy
  tests/                      # pytest
```

---

## Phase 0 — Setup (½ day)

- [ ] `main/requirements.txt` with pinned versions (numpy, scipy, cma, botorch,
      torch, pybobyqa, ioh, pydantic, pyyaml, pytest, LLM SDK).
- [ ] `pytest` skeleton plus a smoke test that imports every optimizer.
- [ ] **Dev smoke tasks:** a small local set (e.g. BBOB f1, f8, f10, f15, f21 at
      d=5,10) for development only. Results from these are **not** official.
      The official suite comes from the benchmark team.
- [ ] Decide the LLM provider and model, and how API keys are handled
      (env var, never committed).

**Done when:** `pytest` passes on a clean `.venv`.

---

## Phase 1 — Archive and warm starts (M1) (2–3 days)

Fixes the biggest Task 001 failure (progress lost when switching methods).

- [x] TuRBO sign fix (`fix-turbo-sign`, merged).
- [ ] `agent/archive.py`: `incumbent`, `top_k`, `local_scale`,
      `diverse_starts`, and tag bookkeeping.
- [ ] `from_archive(bounds, archive, start=..., scale=..., seed=...)` for all 9
      optimizers (design §6.2 table). Midpoint start only as an explicit
      `start="center"`.
- [ ] Expose internal state needed for reports: CMA-ES `sigma`, TuRBO TR
      length, BOBYQA trust radius, and a converged/exhausted flag on every
      wrapper (`state()` method).
- [ ] BOBYQA: replace the thread wrapper with a subprocess or a synchronous
      driver (it is slow past ~500 evals today), or cap it and document why.
- [ ] Tests: warm-started optimizers never make the incumbent worse, and the
      first proposals sit near the incumbent at the expected scale.
- [ ] Regenerate the TuRBO baseline on the team's reference machine.

**Done when:** all warm-start tests pass, and a hand-scripted
"CMA-ES 800 → BOBYQA from incumbent" run on Task 001 beats both methods run alone.

---

## Phase 2 — Controller skeleton and problem intake (M2) (3–4 days)

- [ ] `agent/intake.py`: schema from design §7.1, defaults by `eval_cost` /
      `noise` / `batch_size`, saved as `intake.yaml`.
- [ ] `agent/schema.py`: `StateReport` and `Action` (pydantic), with JSON schema
      export (this also feeds the action-menu section of `ROLE.md`).
- [ ] `agent/controller.py`: round sizing, action validation, execution of
      `continue / switch / restart / polish / stop`, budget accounting, and
      early round end on events.
- [ ] `agent/decision_log.py`: `decisions.jsonl` with each action's
      expectation and whether it held.
- [ ] `policies/scripted.py` and `policies/random_policy.py`.
- [ ] `scripts/run_agent.py --task --policy --seed --budget`, writing the existing
      trace format plus `decisions.jsonl` and `intake.yaml`.

**Done when:** the scripted policy runs end to end on all dev tasks, never
goes over budget, and the trace and decision log are complete.

---

## Phase 3 — Landscape probes (M3) (3–4 days)

- [ ] `agent/probes/`: P1 space-filling stats, P2 fitness–distance correlation (FDC)
      and top-10% spread, P3 quadratic fit R², P4 multi-start local runs, P5
      finite-difference gradient consistency, P6 separability, P7 noise.
- [ ] All probe evaluations go into the archive (no budget wasted). Probe
      budget scales with dim and budget (reduced set when budget < 20·dim).
- [ ] `probe` action wired into the controller, with features in `StateReport.probes`.
- [ ] Validation: the features separate the dev tasks' landscape types
      (smooth unimodal vs ill-conditioned vs multimodal).

**Done when:** within ≤300 evals on Task 001, probes report single-basin and
smooth, and the scripted policy using them reaches regret 1e-6 in ≤1277 evals.

---

## Phase 4 — Portfolio racing (M4) (2 days)

- [ ] `agent/racing.py`: successive halving (default) and UCB. Interleaved
      evaluation, 30% budget cap, per-candidate tags (`race:<method>`).
- [ ] `race` action in the controller, with race status in the `StateReport`.
- [ ] Tests: on dev tasks the race keeps the method that is best in hindsight
      in most seeds.

**Done when:** a race on Task 001 picks a local solver (L-BFGS-B or BOBYQA), and on
a multimodal dev task it picks a population method.

---

## Phase 5 — Playbook v0 (2–3 days, can overlap Phases 3–4)

- [ ] `ROLE.md` (≤1 page): goal, inputs, action menu (generated from
      `schema.py`), core principles.
- [ ] Skills: `intake`, `probing` (+ feature thresholds), `method-selection`
      (+ one method card per optimizer), `racing`, `stall-recovery`.
- [ ] Each rule comes with its reason, numeric criteria, and 1–2 worked examples.
- [ ] Leak check: a script that fails CI if the playbook names a benchmark
      function or suite (Rosenbrock, BBOB fN, Rastrigin, …).
- [ ] Playbook version stamped into every run's metadata.

**Done when:** the playbook is reviewed by a teammate and passes the leak check.

---

## Phase 6 — LLM strategist (M5) (3–5 days)

- [ ] `policies/llm.py`: one call per round. Input: `ROLE.md` + relevant skills +
      `StateReport`. Output: `Action` via forced tool call / structured
      output. On an invalid action → one retry, then fall back to the scripted policy (logged).
- [ ] Cost controls: token budget per run, and optionally a cheaper model for
      routine rounds with a larger one only on events.
- [ ] Record full prompts and responses (separate from `decisions.jsonl`) so
      runs can be replayed.
- [ ] Blind runs on Task 001 through a harness that doesn't leak the identity
      (needs the benchmark team's subprocess harness, design §8.2).

**Done when:** on Task 001 the LLM agent reaches regret 1e-6 in ≤1277 evals,
median over ≥5 replicates.

---

## Phase 7 — Evaluation on the team suite (ongoing)

- [ ] Run on the benchmark team's suite: LLM agent vs scripted policy vs
      random policy vs v1 agent vs best fixed single method vs per-task best
      method in hindsight.
- [ ] Playbook ablations: none vs v0 vs later versions.
- [ ] Report: fraction of the gap closed, area under the regret curve, evals to target, tokens,
      wall-clock time.

---

## Dependencies on the benchmark team

| Needed by | Item |
|---|---|
| Phase 2 | Agreed task API and intake metadata fields (design §8.1) |
| Phase 6 | Leak-proof subprocess harness (design §8.2) |
| Phase 7 | Official suite + reference results + agreed metrics (design §8.4–8.5) |

Until these exist, development runs on the dev smoke tasks only.

## Decisions needed

1. LLM provider and model(s), and the API budget.
2. Fix BOBYQA's driver (subprocess) vs cap its use.
3. Round size: fixed fraction vs adaptive.
4. Whether `Action` may set free hyperparameters or only presets.
