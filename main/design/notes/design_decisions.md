# Design decisions — agent v2

(Copied from the claude.ai project "cheaplunch_optimization".)

- Scope chosen by Mohamad: (1) round-based decision loop, (2) landscape probing, (3) portfolio racing, (4) shared archive + warm-start hand-offs, (6) problem intake.
- Out of scope for now: (5) cross-task memory / case library.
- Team split: benchmark suite and problem set owned by other team members; Mohamad owns agent improvement.
- Stack: no LangChain/LangGraph. Plain-Python deterministic controller; one structured-output LLM call per round (StateReport → Action via forced tool call `take_action`); strategy knowledge in a versioned markdown playbook with no benchmark identities, tested by ablation.
- LLM providers: Anthropic and OpenAI, switchable with `--provider`.
- Actions use presets only (start kind, popsize_factor), not free hyperparameters.
- `main/docs/` is gitignored, so design docs live in `main/design/`.
- Day 1 (2026-10-21): blocks 1–7 built on `agent-v2`. Bugs fixed: TuRBO sign; BOBYQA maxfun=10**9 (~8 GB allocation, the real cause of the "slow past 500 evals" limitation).
