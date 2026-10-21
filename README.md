# Agentoptimizer

Research project: an LLM agent that selects, switches and hybridizes
black-box optimizers under a budget, instead of committing to a single
fixed method.

## Structure

```
Agentoptimizer/
├── main/                 the actual project: optimizer library, tasks,
│                         docs, baseline campaign results, agent briefs
└── task_001_sandbox/     an isolated, minimal copy used to run an LLM
                          agent blindly on Task 001
```

**`main/`** is everything: the optimizer library (`src/optimizers/`),
task wrappers (`src/tasks/`), the campaign runner (`scripts/`), all
literature surveys and the optimizer/benchmark catalogues (`docs/`), the
baseline results already generated (`results/`), and the agent briefs
(`agent_briefs/`).

**`task_001_sandbox/`** exists for exactly one reason: `main/` has the
true identity of Task 001 all over it — `main/results/rosenbrock_d10_i1/`
as a directory name, `main/docs/` discussing it by name repeatedly. An
agent given full access to `main/` could trivially discover what the
black-box function actually is, which would invalidate the entire blind
test. The sandbox is a separate copy containing only the optimizer
library and the task harness — nothing that reveals what it's solving.

## Before running an agent on `task_001_sandbox/`, two things matter more than anything else

1. **Start a brand-new conversation, not a continuation of any chat that
   has discussed this project.** Any session with that context already
   knows Task 001 is Rosenbrock. The blind framing only works in a
   conversation that has never seen that discussion.

2. **Scope the agent to `task_001_sandbox/` only — never open an editor
   or terminal on `main/` or on this repo's parent folder in the same
   session.** If the agent's file tools can reach `main/` as a sibling
   directory, the same leak happens through the filesystem instead of
   through conversation history.

With both respected, start a fresh agent session with `task_001_sandbox/`
as its working directory, and give it exactly this as your first message:

> Read `agent_briefs/task_001.md` and carry out everything it describes.

Nothing more — no extra framing or context, for the same reason as above.
See that file for the full protocol (budget, required output files, the
pre-registered-roadmap + append-only-changelog rule, and the hard
constraints).

Results land in `task_001_sandbox/results/agent_task_001/`. Scoring them
against the true optimum has to happen from `main/` afterward, since that
number must never be anywhere the agent can reach.
