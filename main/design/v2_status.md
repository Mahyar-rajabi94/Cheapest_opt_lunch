# Agent v2 — Status after day 1 (2026-10-21)

Branch `agent-v2`. Design: `agent_design.md`. Plan: `roadmap.md`.

## What's built
| Block | Status | Code |
|---|---|---|
| 1. Archive + warm starts | done | `agent/archive.py`, `agent/warmstart.py`, wrapper kwargs + `state()`/`close()` |
| 2. Controller + intake | done | `agent/controller.py`, `agent/intake.py`, `agent/schema.py`, `agent/evaluator.py`, `agent/decision_log.py` |
| 3. Probes | P1–P5 done; P6, P7 deferred | `agent/probes.py` |
| 4. Racing | successive halving done; UCB deferred | `agent/racing.py` |
| 5. Playbook v0 | ROLE + probing, method-selection, stall-recovery; racing/intake skills deferred | `agent/playbook/`, `agent/playbook_loader.py` |
| 6. LLM strategist | Anthropic + OpenAI behind one adapter; tested offline only (no API key in the build environment) | `agent/policies/llm.py` |
| Baselines | scripted (rules), fixed single method | `agent/policies/scripted.py`, `agent/policies/fixed.py` |

Tests: `python -m pytest tests` (41 tests).

## Bugs found and fixed along the way
- **TuRBO** maximized instead of minimizing (GP fit on y; Thompson sampling picks the max). Fixed on `fix-turbo-sign`.
- **BOBYQA** wrapper passed `maxfun=10**9`. Py-BOBYQA allocates lists of length maxfun (~8 GB). That was the real cause of the documented "slow past ~500 evals": memory paging. With `maxfun=20000`, BOBYQA solves the Task 001 problem alone in 1,187 evals. **The stored 500-eval BOBYQA baseline in `main/results/` is invalid and should be rerun**, like TuRBO's.

## Dev smoke results (NOT official — dev BBOB tasks, d=10, budget 2,000)
Median over 5 seeds (deterministic methods: 1 run). "t(1e-6)" = evals to reach regret 1e-6, "-" = not reached by all runs.

| Task (type) | scripted v2 | CMA-ES | L-BFGS-B | BOBYQA |
|---|---|---|---|---|
| Task 001 problem (curved valley) | 3e-8, t(1e-6) 1,252 (range 845–1,705) | 4.21 | 1e-7, t 1,277 | 3e-13, t 1,187 |
| separable bowl | 3e-13, t 176 | t 1,098 | t 23 | t 23 |
| ill-conditioned bowl | 28.1 | 39.0 | **3.33** | 32.6 |
| multimodal, global structure | **15.9** | 16.7 | 120 | 63.7 |
| multimodal, weak structure | 2.78 | 9.48 | 15.7 | **2.26** |

Reading:
- The scripted policy is never the worst and avoids the disasters of single
  methods (L-BFGS-B on multimodal, CMA-ES on smooth single-basin). It is
  close to the best method on 3/5 tasks.
- It loses to the best single method where the probes mislead it. On the
  ill-conditioned bowl, short local runs don't finish, so it reads as
  multi-basin → CMA-ES. In general the probe budget (~250 evals) is a fixed
  cost on easy problems.
- Classification from P4 (3 short local runs) is noisy: it flips between
  seeds on the ill-conditioned and weak-structure tasks.
- The v1 Claude agent needed 1,925 evals to reach 1e-6 on Task 001.

## Next (day 2+)
1. Run the LLM policy with real keys:
   `python scripts/run_agent.py --task task001 --policy llm --provider anthropic` (and `--provider openai`), 5 seeds, plus `--no-playbook`.
2. Compare LLM vs scripted with `scripts/eval_dev.py` (add an `llm` policy entry).
3. Probe robustness: more P4 starts or longer runs; use the P3 quadratic R² to tell ill-conditioning from multimodality.
4. Rerun the TuRBO and BOBYQA baselines in `main/results/` on the reference machine.
5. Deferred items: P6/P7, UCB racing, racing/intake skills, random policy.
6. Benchmark team: subprocess harness (leak-proof) and official suite.
