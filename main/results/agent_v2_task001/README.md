# Agent v2 on Task 001: scripted vs LLM (GPT-5), 2026-10-21

**Dev results, not official.** Task 001 problem (d=10, budget 2,000), 5 seeds.
Official evaluation uses the benchmark team's suite.

## Contents
| Path | What |
|---|---|
| `v2_scripted_s{0..4}/` | scripted (rules) policy |
| `llm_openai_pb_s{0..4}/` | LLM policy, OpenAI `gpt-5`, playbook `d16bc3c09f` |
| `llm_openai_nopb_s{0..4}/` | LLM policy, OpenAI `gpt-5`, `--no-playbook` (ROLE only) |
| `baselines/rosenbrock_d10_i1/{bobyqa,turbo}/seed_0/` | BOBYQA and TuRBO rerun with the fixed wrappers (budget 2,000), replacing the invalid stored runs for plotting |
| `llm_vs_scripted.txt` | per-seed table: evals, regret, evals to 1e-2 / 1e-6, LLM calls, fallbacks, tokens |
| `eval_dev.txt`, `eval_dev.json` | `scripts/eval_dev.py --seeds 5` on all dev tasks (scripted vs fixed methods) |
| `plots/` | `regret_vs_eval.png`, `method_timeline.png` |

Each run folder has `trace.csv`, `meta.json`, `decisions.jsonl`, `intake.yaml`,
and for LLM runs `llm_transcript.jsonl`. All 10 LLM runs: 0 fallbacks.

## Summary
| Policy | Reached 1e-6 | Median evals to 1e-6 | Median final regret |
|---|---|---|---|
| scripted | 5/5 | 1,263 | 1.8e-8 |
| LLM + playbook | 5/5 | 1,086 | 1.2e-8 |
| LLM, no playbook | 4/5 | 1,321 | 8.0e-10 |

- LLM + playbook takes the same opening as scripted (probes, then L-BFGS-B)
  but stops once converged (median 1,357 evals used vs 2,000).
- Without the playbook the LLM switches and races more, polishes deeper, and
  failed one seed (seed 3, regret 0.12).
- Environment: Python 3.12, numpy 2.5.4, scipy 1.18.1 (not the reference machine).
  `eval_dev.txt` differs a little from the day-1 table in `design/v2_status.md`.

## Reproduce
```
python scripts/run_agent.py --task task001 --policy scripted --seed S --outdir results/agent_v2_task001/v2_scripted_sS
python scripts/run_agent.py --task task001 --policy llm --provider openai --seed S [--no-playbook] --outdir ...
python scripts/plot_agent_runs.py --baselines results/rosenbrock_d10_i1 \
    --override results/agent_v2_task001/baselines/rosenbrock_d10_i1 \
    --agent "v2 scripted=results/agent_v2_task001/v2_scripted_s{0..4}" \
    --agent "v2 LLM (GPT-5) + playbook=results/agent_v2_task001/llm_openai_pb_s{0..4}" \
    --agent "v2 LLM (GPT-5), no playbook=results/agent_v2_task001/llm_openai_nopb_s{0..4}" \
    --agent "v1 Claude agent=../task_001_sandbox_claud/results/agent_task_001" \
    --agent "v1 OpenAI agent=../task_001_sandbox_openai/results/agent_task_001" \
    --out results/agent_v2_task001/plots
```
