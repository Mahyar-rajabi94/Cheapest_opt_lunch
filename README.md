# Agentoptimizer

Research project: an LLM agent that selects, switches, and hybridizes black-box
optimizers under a budget, instead of committing to a single fixed method.

## Structure

```text
Cheapest_opt_lunch/
├── main/                       project source, baseline runner and results
│   ├── AGENTS.md               reusable intake and campaign instructions
│   ├── CLAUDE.md               imports the same instructions for Claude
│   ├── task_definitions/       created by the agent during planning
│   │   └── <run_id>.md         synthesized problem, reviewed by the user
│   ├── src/tasks/registry.py   every baseline task, one line each
│   ├── scripts/make_sandbox.py builds a blind sandbox for task_<NNN>
│   └── agent_briefs/
│       ├── task_001.md         optional saved example; not required to run
│       ├── task_template.md    optional example of problem details
│       └── blind_brief_template.md  blind brief, rendered by make_sandbox.py
├── task_001_sandbox_claud/      self-contained Claude sandbox copy
├── task_001_sandbox_openai/     self-contained Codex sandbox copy
├── task_002_sandbox/            retained Task 002 benchmark sandbox
├── task_003_sandbox/            retained Task 003 benchmark sandbox
└── task_004_sandbox/            retained Task 004 benchmark sandbox
```

The two Task 001 sandboxes carry local copies of the instructions and example
prompts. They do not import instructions from `main/`: a blind run must not need
access outside its sandbox. Maintain these four Markdown files in `main/`, then
copy changes to those two sandboxes. Keep the copies identical.

The Task 002–004 sandboxes retain their existing benchmark briefs and artifacts.
For a fresh conversational run based on one of those evaluators, include copies
of `main/AGENTS.md` and `main/CLAUDE.md` in the clean standalone workspace.
The plotting scripts and existing benchmark results remain under `main/`.

## Define a task in your terminal

Start a fresh Claude or Codex session **from the chosen workspace's root**
(`main/` for development, or an isolated sandbox root for a blind campaign).
Codex discovers `AGENTS.md`; `CLAUDE.md` imports that same file for Claude.
These are project instructions, not a replacement for the terminal client's
system prompt. See the official [Codex instruction documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
and [Claude instruction documentation](https://code.claude.com/docs/en/memory#import-additional-files).

Describe the optimization problem as an ordinary user message. **The agent
writes the task definition; you do not need to write or select a task brief.**
The agent turns your description and clarification answers into
`task_definitions/<run_id>.md` for your review. If your first message supplies
everything needed, it goes directly to drafting that definition. Otherwise it
asks focused questions first. For example:

> I want to maximize measured yield by choosing experiments from my candidate
> table. I can afford 40 evaluations. I will provide the table location and the
> Python evaluator with its public calling example. Help me clarify the task.

This deliberately incomplete prompt should begin a clarification conversation.
The agent gathers the missing details, writes the definition, and waits for
confirmation before optimization. The current Task 001 harness does not become
a yield evaluator just because the user describes that problem.

For the existing Task 001 problem, you can also describe it directly:

> Minimize a deterministic black-box function with 10 continuous inputs, each
> bounded by [-5, 5], using at most 2000 evaluations. New points anywhere inside
> the box are allowed; there is no candidate table or additional constraint.
> The supplied Python evaluator is `src/agent_harness/task_001.py`; its
> implementation is sealed. I can provide its public calling contract. Clarify
> anything else you need and write the task definition for me to confirm.

This example still needs the public calling contract before it is ready to run.
The agent should ask for it, not inspect sealed source or silently load an old
brief. The files in `agent_briefs/` are retained only as optional examples. They
are not automatically read and can be omitted from a new run workspace.

## Planning and confirmation workflow

```text
User's natural-language problem description (complete or incomplete)
    -> extract known facts and identify missing/conflicting information
    -> clarify consequential uncertainty, if any
    -> agent writes task_definitions/<run_id>.md
    -> present that definition for review
    -> WAIT for the user's explicit confirmation
    -> mark the definition confirmed and preserve it
    -> write the initial roadmap with that confirmed definition
    -> run the existing optimization campaign and logging protocol
```

The agent uses the supplied objective, domain, candidate source, budget, and
Python interface. A sufficiently specified problem goes straight to the draft
definition. Questions should resolve uncertainty that could change the problem,
invalidate execution, or materially improve the plan. The agent can help you
notice missed assumptions, without requiring an exhaustive questionnaire.

Details such as pool eligibility, repetitions, or prior data are explored only
when relevant and unclear. Scientific properties such as smoothness may remain
unknown. The agent chooses routine operational defaults and shows them in the
summary. A real Python evaluator and public calling contract are required before
execution. Planning spends no evaluations, and even a complete definition still
waits for your confirmation.

The generated definition contains task-specific facts, assumptions, and the
public Python calling example; the reusable workflow stays in `AGENTS.md`.
The agent marks the draft as awaiting confirmation and updates it if you correct
anything. After you confirm, it marks that version confirmed and preserves it.
Later changes require a separate revision and renewed confirmation.

Before evaluating, the agent copies the confirmed definition into
`roadmap_initial.md`, references its file, and adds the method-selection plan.
Routine method switches remain agent decisions recorded in the append-only
changelog. The generated task definition lives outside the campaign result
directory, so that directory still contains exactly:

- `roadmap_initial.md` — confirmed problem and immutable initial method plan.
- `changelog.md` — append-only deviations; empty when there are none.
- `trace.csv` — real evaluations written by the evaluator.
- `meta.json` — run metadata written by the evaluator.

This is a prompt-level workflow. It does not toggle the CLI's built-in plan
mode or add a code-enforced approval gate. The existing harness still enforces
its evaluation count. Optimizer implementations, objective code, the baseline
runner, and trace/metadata formats are unchanged. The intake can describe a
wider range of tasks, but execution still needs a compatible evaluator and
optimization method; no discrete, pool, or multiobjective engine is added here.

## Task code → true identity

The blind-test sandboxes (and everything handed to an agent) refer to
tasks only by code — `task_001`, `task_002`, etc. This mapping is exactly
what must never leak into a sandbox or an agent's briefing; it exists only
here, for the human reading results from `main/`.

| Task code  | Function (BBOB id)     | dim | True optimum (`f_opt`) |
|------------|-------------------------|-----|-------------------------|
| `task_001` | Rosenbrock (f8)         | 10  | 149.15                  |
| `task_002` | Rastrigin (f15)         | 10  | 1000.0                  |
| `task_003` | Bent Cigar (f12)        | 10  | -621.11                 |
| `task_004` | Sphere (f1)             | 10  | 79.48                   |

All four use BBOB instance 1. `f_opt` values are specific to that instance
(a different instance would shift them).

These task codes identify the existing benchmark problems. Repeating a task is
another run of that same problem; use a fresh run identifier and output directory.
New user-defined problems need their own compatible Python evaluator and should
not reuse an existing task code for a different objective.

## Preserve blind runs and existing results

1. Start a brand-new conversation with no prior discussion of the objective's
   identity. This development conversation is not suitable for a blind run.
2. Give the agent access only to a standalone sandbox. `main/` includes source
   and baseline result paths that reveal the objective. Changing the terminal's
   working directory alone does not enforce filesystem isolation; do not expose
   the repository parent, sibling workspaces, baseline results, or Git history
   to the campaign agent.
3. The Task 001–003 sandbox copies already contain historical campaign results.
   Prepare a fresh standalone workspace without those results, old campaign
   scripts, or previous generated definitions before a blind rerun. Preserve
   the originals. Choosing a new output directory prevents overwriting, but
   does not hide old results from the agent.
4. Include `AGENTS.md`, `CLAUDE.md`, the required `src/` files, and the run
   environment in that clean sandbox. Supply the problem and public evaluator
   contract in the conversation or as explicitly provided documentation.
   `agent_briefs/` is optional. The agent creates `task_definitions/` itself;
   it does not need to read the sealed evaluator's implementation.

Score completed campaigns from `main/` afterward. The true optimum and other
methods' performance must never be exposed to the blind optimization agent.

## Check the conversational behavior

In a fresh isolated test session, check these cases without spending objective
budget. These are manual acceptance checks, not claims of automated enforcement:

| Input or action | Expected behavior |
| --- | --- |
| Complete problem and public Python contract in a user message, no brief files | Goes straight to the definition and confirmation without redundant questions; zero evaluations before confirmation. |
| “Optimize my experiment” | Asks focused questions; does not inherit Task 001's dimension, budget, or evaluator. |
| User supplies the missing details in follow-up messages | Synthesizes those answers into the definition itself; never asks the user to author a brief. |
| Problem with no Python evaluator | Requests the script/module and calling contract; does not invent one or start. |
| Fixed table with continuous feature columns | Uses supplied pool restrictions and asks only about consequential gaps; does not assume new points are allowed. |
| Operational details such as a run name are omitted | Chooses sensible defaults and shows them in the definition instead of asking extra questions. |
| User domain disagrees with evaluator documentation | Surfaces the conflict and waits for resolution. |
| Unknown smoothness or noise | Records it as unknown and proposes handling, while resolving actual execution blockers. |
| Correct the proposed definition instead of confirming | Updates the unconfirmed file and summary, then waits for confirmation of the revision. |
| Existing output directory or previous campaign | Arranges a fresh run without reading or overwriting historical results. |

For a subsequent approved smoke campaign, verify that the roadmap exists before
the first evaluation, every recorded value comes from the evaluator, optimizer
switches are labeled, and the same four output files are produced.
