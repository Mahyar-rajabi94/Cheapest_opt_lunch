# Optional starting point for a user prompt

Describe your task directly in your Claude/Codex terminal. This file is optional
example material; neither you nor the agent needs it to start a task. You do not
need to fill every field: the agent extracts what you know, asks about missing
essentials, and writes `task_definitions/<run_id>.md` for your confirmation.
If your message is already complete, it can draft that definition immediately.
An existing Python evaluator and its public calling contract are required before
execution. This optional template is task input, not reusable agent instructions.

---

Help me define and run an optimization task.

- **Goal:** [quantity to minimize/maximize, and units if relevant]
- **Inputs:** [decision variables; continuous/integer/categorical/mixed;
  bounds/allowed values; constraints]
- **Candidates:** [select from a fixed pool / generate new feasible points /
  a defined combination]. For a pool: [data file, candidate IDs, permitted
  columns, and which rows have already been evaluated].
- **Observations:** [deterministic / noisy / unknown; repeats allowed?]
- **Budget:** [evaluation limit; known cost/time limits; sequential/batch]
- **Prior data:** [permitted observations, or none; any hidden columns/files]
- **Evaluator:** [path to the existing Python script/module]. Its public usage
  is [import/command and initialization; candidate input; returned value(s);
  budget counting/errors; method labels; save method and output files].
- **Run:** [task name and a fresh output directory]

[Include the Python usage example here, or reference supplied public interface
documentation. Do not replace the evaluator with an invented objective. If the
evaluator does not exist yet, say so: execution must wait until it is supplied.]
