# Agent Design v2 — Adaptive Optimization Agent

**Owner (agent):** Mohamad Moosavi
**Status:** Draft, 2026-10-21
**Related work owned by others:** benchmark suite and problem set (see §8, "Interface with the benchmark team")

---

## 1. Motivation

Task 001 (two blind agent runs plus baselines, commit `75a251c`) showed that the
current agent is a *planner*, not an *adaptive agent*:

- Both agents wrote a fixed pipeline before seeing any data
  (space-filling scan → CMA-ES → local polish) and then mostly followed it.
- The Claude agent reached the optimum (regret 3.8e-9) but needed 1925 evals
  to reach 1e-6. Plain L-BFGS-B from the box midpoint needed only 1277. About
  1250 evals went to exploration that the problem did not need.
- The OpenAI agent's final value is identical to standalone BOBYQA (155.626).
  Its last phase restarted from the box midpoint and dropped the CMA-ES
  incumbent, so the first 1500 evals did nothing for the final result.

The design below fixes this. The agent works in a closed loop:
**ask about the problem → probe → decide → act → observe → decide again**, and
it never loses progress when it switches methods.

Out of scope for v2: memory or meta-learning across tasks (a case library).
We may revisit it once the closed loop works.

---

## 2. Architecture overview

```
            ┌───────────────────────────────────────────────┐
 user ────► │ 0. Problem intake (§7)                        │
            └───────────────┬───────────────────────────────┘
                            ▼
            ┌───────────────────────────────────────────────┐
            │ LLM strategist (reasons, picks actions)       │
            │   reads: StateReport  →  emits: Action        │
            └───────▲───────────────────────┬───────────────┘
                    │ every round            │
            ┌───────┴───────────────────────▼───────────────┐
            │ Controller (pure Python, deterministic)       │
            │  - runs optimizers for the round's budget     │
            │  - runs probes (§4) and races (§5)            │
            │  - builds StateReport                         │
            └───────┬───────────────────────▲───────────────┘
                    │ ask/tell               │
            ┌───────▼───────────────────────┴───────────────┐
            │ Shared Archive (§6)  ◄──► optimizer wrappers  │
            └───────┬───────────────────────────────────────┘
                    │ evaluate(x)
            ┌───────▼───────────────────────────────────────┐
            │ Harness / Task (owned by benchmark team, §8)  │
            └───────────────────────────────────────────────┘
```

The roles are kept separate on purpose:

- **LLM strategist:** strategy only. It never sees or proposes single points.
- **Controller:** all numerics. It is deterministic given a seed, so results
  can be reproduced.
- **Archive:** the single source of truth for every evaluation.

---

## 3. Component 1 — Round-based decision loop

### 3.1 Rounds
- The budget is split into rounds. Default round size:
  `k = max(5·dim, round(0.05·budget))`, so 100 evals for d=10 with budget 2000.
- The controller may end a round early on an event: an optimizer exhausts or
  converges, a trust region collapses, or a race finishes.
- After each round the controller sends a `StateReport` to the strategist and
  waits for an `Action`.

### 3.2 StateReport (what the LLM sees)
This is compact, structured JSON with no raw point dumps:

```json
{
  "round": 7, "evals_used": 700, "evals_left": 1300,
  "problem": { "...intake fields, §7..." },
  "incumbent": {"y": 156.37, "found_at": 640, "by": "cma_es"},
  "progress": {
    "best_curve_log10_regret_proxy": [...],   // best-so-far, subsampled
    "improvement_last_round": 0.8,
    "stall_evals": 60
  },
  "active": [{"method": "cma_es", "evals": 550, "sigma": 0.03,
              "rel_improvement_per_100": 0.004}],
  "probes": { "...landscape features, §4..." },
  "race": null,
  "history": [{"round": 6, "action": "continue", "rationale_short": "..."}]
}
```

Notes:

- "Regret proxy" means `y − best_so_far`, or a log-scaled improvement. The
  true optimum is **never** exposed.
- `history` holds the agent's own last N decisions, so it can stay consistent
  from round to round.

### 3.3 Action menu (what the LLM can do)
The menu is closed, and each action is validated by the controller:

| Action | Args | Effect |
|---|---|---|
| `continue` | – | Run the active method(s) for another round |
| `switch` | `method, hyperparams?` | Stop the current method and start a new one, warm-started from the archive (§6) |
| `restart` | `method, start: incumbent \| diverse \| random, scale?` | Fresh instance (e.g. IPOP-style larger population, or a new region) |
| `probe` | `probe_ids, budget` | Run landscape probes (§4) |
| `race` | `candidates[], budget, rule` | Portfolio race (§5) |
| `polish` | `method` | Local refinement from the incumbent until it converges or the budget ends |
| `stop` | `reason` | Save and finish early |

Each `Action` includes a short `rationale` and a falsifiable expectation, e.g.
`"expect ≥10% improvement within 200 evals, else switch to X"`. The
controller logs it. At the next round the report says whether the expectation
held, so the agent's reasoning is checked against data.

### 3.4 Decision log (replaces roadmap + changelog)
- `decisions.jsonl`: one line per round with the StateReport hash, the Action,
  the rationale, the expectation, and whether it held.
- The pre-registered roadmap is kept, but shortened to the agent's initial
  hypotheses. The decision log becomes the main record.

---

## 4. Component 2 — Landscape probing

Goal: spend about 5–10% of the budget to learn *what kind of problem this is*,
and give the strategist evidence rather than guesses.

### 4.1 Probe set (v2)
| ID | Probe | Default cost (d=10) | Output feature(s) |
|---|---|---|---|
| P1 | Space-filling sample (Sobol/LHS) | 10·d = 100 | y distribution: skew, kurtosis, log-range, fraction of extreme values |
| P2 | Fitness–distance correlation (on P1) | 0 | FDC to the best sample; top-10% spread vs box diagonal |
| P3 | Global quadratic fit (on P1) | 0 | R² of linear, quadratic and quadratic+interaction models (smoothness, separability hint) |
| P4 | Multi-start short local runs | 3–5 starts × ≈30 evals | Do the endpoints coincide (same y and x within tolerance)? This estimates the number of basins |
| P5 | Finite-difference gradient consistency | ≈2·d per point × 3 points | Cosine similarity of FD gradients at step h vs h/10 (gradient reliability, roughness, noise) |
| P6 | Separability test | ≈2·d | Change in y from coordinate perturbations, additive vs joint |
| P7 | Re-evaluation noise check | 3–5 repeats | Std of repeated y. Skipped if intake says the problem is deterministic |

Notes:

- Probes run on the controller. Their raw evaluations go into the archive, so
  none of the budget is wasted (P4's best endpoint often becomes the incumbent).
- Total default cost for d=10 is about 250–300 evals. The strategist can ask
  for a reduced set when the budget is tight (e.g. P1+P4 only when the budget
  is under 20·d).

### 4.2 Interpretation guidance (given to the strategist as a prompt, not hard rules)
- P4 endpoints agree and P5 is consistent → smooth, likely unimodal. Go to a
  gradient or model-based local method (L-BFGS-B, BOBYQA) right away.
- P4 endpoints disagree, but the top-10% spread is small → a few basins. Use
  CMA-ES with restarts or race local methods from diverse starts.
- P4 disagrees, P2 FDC is low, and P1 is heavy-tailed → rugged or multimodal.
  Use population methods (CMA-ES IPOP/BIPOP, DE) or a race.
- P7 shows noise → avoid FD gradients. Prefer CMA-ES or noise-aware BO.
- Expensive evaluations (intake) → use a smaller probe set and prefer GP-BO or
  TuRBO.

Sanity target on Task 001: P4 should flag Rosenbrock-10D as single-basin
within about 200 evals, and the agent should then match or beat L-BFGS-B
alone (≈1277 evals to 1e-6).

---

## 5. Component 3 — Portfolio racing

Used when the probes are ambiguous. It is the agent's explicit hedge against
no free lunch.

- **Candidates:** 2–4 methods chosen by the strategist, all warm-started from
  the archive (§6).
- **Default rule: successive halving.** Each candidate gets `b0` evals
  (default `b0 = round(0.05·budget / n_candidates)`). Drop the worst half by
  normalized improvement (`(y_start − y_now) / evals`, with ties broken by the
  best y reached). Double the budget for the survivors and repeat until one is
  left or the race budget is spent.
- **Alternative rule:** a UCB bandit over improvement rate, for long budgets.
- **Fairness:** candidates are interleaved, so wall-clock order doesn't bias
  them. Each candidate's evals are tagged in the trace (`optimizer =
  race:<method>`).
- **Race cap:** at most 30% of the total budget by default. The StateReport
  shows the result of each round of the race.

---

## 6. Component 4 — Shared archive and warm-start hand-offs

Switching methods must never lose progress.

### 6.1 Archive
```python
class Archive:
    X: np.ndarray            # (n, d), every evaluated point
    y: np.ndarray            # (n,)
    tags: list[str]          # which method/probe produced each point
    def incumbent(self) -> tuple[np.ndarray, float]
    def top_k(self, k) -> tuple[np.ndarray, np.ndarray]
    def local_scale(self, k=2*d) -> np.ndarray   # per-dim spread of the top-k points
    def diverse_starts(self, m) -> np.ndarray    # good AND mutually distant points
```

### 6.2 Warm-start contract for every optimizer wrapper
Add an optional, standard constructor:

```python
Optimizer.from_archive(bounds, archive, *, start="incumbent", scale=None, seed=None)
```

| Optimizer | Warm-start behavior |
|---|---|
| CMA-ES | `x0` = incumbent, `sigma0` = median of `archive.local_scale()`; optionally seed the covariance from the top-k points |
| DE | Initial population = archive `top_k`, filled up with LHS around the incumbent |
| GP-BO / TuRBO | Fit the GP on archive points (all, or the nearest N to the incumbent); TuRBO's trust region is centered on the incumbent, with its length set from `local_scale` |
| BOBYQA | `x0` = incumbent, `rhobeg` from `local_scale`; actual bounds, not the re-centered box |
| L-BFGS-B / Nelder-Mead | `x0` = incumbent; NM initial simplex scaled by `local_scale` |

Requirements:

- Remove the "always start at the box midpoint" behavior from the current
  wrappers. Midpoint stays as an explicit `start="center"` option only.
- Expose internal state for the StateReport: CMA-ES `sigma`, TuRBO trust-region
  length, BOBYQA trust radius, and convergence flags.
- Fix the TuRBO sign bug: `MaxPosteriorSampling` maximizes but the model is fit
  on raw `y` for minimization. Fit on `−y`. This is a prerequisite for using
  TuRBO in races.
- Unit test for every optimizer: warm-started from the archive of a finished
  run, the first `ask()` must not return a point worse than the incumbent by
  more than a tolerance (for methods where that makes sense), and the incumbent
  must never get worse.

---

## 7. Component 6 — Problem intake

Before spending any budget, the agent gathers what the user (or the task
metadata) already knows. These facts change which method regime is right.

### 7.1 Intake schema
```yaml
dim: 10
bounds: [[-5, 5], ...]
variable_types: [continuous, ...]      # continuous | integer | categorical
budget_evals: 2000
eval_cost: cheap                        # cheap (<1s) | moderate | expensive (lab / DFT / hours)
noise: deterministic                    # deterministic | noisy | unknown
gradients_available: false
batch_size: 1                           # parallel evals per round
constraints: none                       # none | known_analytic | black_box
prior_knowledge: ""                     # free text: expected smoothness, known good points, ...
warm_start_points: []                   # optional user-provided x (and y)
```

### 7.2 Behavior
- Fields the task metadata leaves out are asked of the user, or set to
  `unknown`, in which case the probes (§4) infer them where they can.
- Intake sets the defaults: probe budget, round size, which methods may be
  chosen (e.g. no FD gradients when noisy, BO-family preferred when
  `eval_cost: expensive`, batch-capable methods when `batch_size > 1`).
- The intake record is saved with the results (`intake.yaml`), so every run
  can be explained afterwards.

---

## 8. Interface with the benchmark team

The benchmark and problem set are owned by other team members. To evaluate
the agent, I need the following:

1. **Task API:** `evaluate(x) -> y`, `dim`, `bounds`, `budget`, plus the
   intake metadata from §7.1 where it is known. The true optimum is kept only
   on the scoring side.
2. **Blind harness:** the problem's identity must not be readable by the
   agent. Suggestion: run the objective in a subprocess or RPC so no source
   file names the function. Today's `ioh.get_problem(8, ...)` in the sandbox
   harness leaks it.
3. **Trace format:** keep the current `trace.csv`/`meta.json` and add the
   `optimizer` tags used above (`probe:P4`, `race:cma_es`, …). The agent adds
   `decisions.jsonl` and `intake.yaml`.
4. **Reference results per task:** the best fixed single method (proposed:
   CMA-ES with restarts) and the per-task best method in hindsight, run with
   enough seeds.
5. **Proposed metrics (to agree together):**
   - Fraction of the gap closed between the best fixed method and the
     per-task best method
   - Area under the log-regret curve
   - Evals to reach target regrets (1e0, 1e-2, 1e-6)
   - Number of agent replicates (≥5), LLM tokens and wall-clock time

Until the suite lands, I will develop against Task 001 plus a few BBOB
functions run locally as smoke tests only. These are not official results.

---

## 9. Milestones (agent side)

| # | Deliverable | Done when |
|---|---|---|
| M1 | TuRBO fix + `Archive` + `from_archive()` on all 9 wrappers + unit tests | Warm-start tests pass; TuRBO beats random search on Task 001 |
| M2 | Problem intake + controller skeleton + `StateReport`/`Action` schema + `decisions.jsonl` | A scripted (non-LLM) policy runs end to end |
| M3 | Probe set P1–P7 | Probes classify Task 001 as single-basin and smooth within 300 evals |
| M4 | Racing (successive halving) | Race picks L-BFGS-B or BOBYQA on Task 001 |
| M5 | LLM strategist in the loop | On Task 001, ≤1277 evals to 1e-6 (matching or beating L-BFGS-B alone); then run on the benchmark team's suite |

Baselines for ablations, measured on the shared suite:

- Scripted policy without the LLM (fixed rules on probe features)
- Random action policy
- The v1 agent

These show what the LLM itself adds.

---

## 10. Open questions

- Round size: fixed fraction of the budget, or adaptive (shorter rounds early)?
- Should the strategist ever set hyperparameters freely, or only choose from
  presets?
- How should races and probes handle `batch_size > 1`?
- LLM choice and budget per decision: is a smaller model enough for routine
  `continue` rounds, with a larger one only on events?
- Metric agreement with the benchmark team (§8.5).
