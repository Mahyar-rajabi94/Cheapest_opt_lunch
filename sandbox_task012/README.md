# sandbox

Reusable sandbox for running an optimization agent on any black-box
problem. Reset and reuse this same folder for each new problem.

## Before each run

1. **Edit `objective.py`** — set `DIM`, `BOUNDS`, `DIRECTION`, `NOISY`, and
   implement `fn(x)`.
2. **Clear old results** if this sandbox was just used for a different
   problem: empty out `results/` (keep the directory itself; the harness
   recreates subfolders as needed).
3. **Scope the agent session to this `sandbox/` folder only** — do not open
   an editor/terminal on the rest of this repo in the same session, so it
   can't see unrelated problem identities or prior answers.

## Starting the agent

Give it `Agent.md` (already copied into this folder) as the system prompt,
plus your own message describing this specific problem — what the
objective represents, dimension/bounds/budget/noise if not already fully
captured in `objective.py`, and anything else `Agent.md` says to supply.
Then let it read `objective.py` and proceed.

## Harness import

```python
import sys
sys.path.insert(0, "<path to this sandbox>")
from src.agent_harness.harness import AgentTaskHarness, BudgetExhausted

harness = AgentTaskHarness(budget=<N>)
```

Everything else (roadmap, changelog, trace/meta output) follows the
protocol in `Agent.md`.

## Example task-specific prompt

This is the message you send after `Agent.md`, once `objective.py` is
configured. It states only the generic problem-specific parameters —
never the function's identity, even if you know it:

```
Problem-specific parameters for this run:
- Dimension: 10
- Bounds: [-5, 5] per coordinate
- Direction: minimize
- Noise: deterministic (no observation noise)
- Budget: 2000 evaluate() calls total, enforced by the harness
- Objective: a sealed black-box function. You do not know its identity
  and must not attempt to discover it — no searching online, no
  inspecting objective.py or any harness source file.
- Harness: sandbox/src/agent_harness/harness.py exposes
  AgentTaskHarness and BudgetExhausted (see this sandbox's README.md
  for the exact import).
- Results directory: results/agent_run/

Read Agent.md in this folder and carry out everything it describes
using the parameters above.
```
