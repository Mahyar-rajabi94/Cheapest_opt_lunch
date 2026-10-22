"""Build a blind sandbox for one agent task.

Creates <outdir> (default: task_<NNN>_sandbox at the repo root) containing only:
    agent_briefs/task_<NNN>.md     rendered from agent_briefs/blind_brief_template.md
    requirements.txt
    src/__init__.py
    src/optimizers/                copy of src/optimizers/, without pool-only optimizers
    src/agent_harness/             __init__.py, harness.py, task_<NNN>.py and the
                                   files in its SANDBOX_ASSETS
    results/                       empty

then leak-checks every file (binary files too) for registered task names and
`f_opt`. Only the
objective construction in task_<NNN>.py may identify the function (by id).
A sandbox that fails the check is deleted.

Usage
-----
    python scripts/make_sandbox.py --task-id 005 --budget 2000 --outdir ../task_005_sandbox_claud
"""

from __future__ import annotations

import argparse
import importlib
import re
import shutil
import sys
import tempfile
from pathlib import Path
from string import Template

MAIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAIN))

from src.tasks.registry import TASKS

# Pool-only optimizers need a pool task's candidates, which no blind harness
# exposes, and their docstrings describe pool tasks.
POOL_ONLY_FILES = ("pool_random.py",)


def render_brief(task_id: str, budget: int) -> str:
    """Fill the brief template from the harness's public dim and bounds."""
    module = importlib.import_module(f"src.agent_harness.task_{task_id}")
    with tempfile.TemporaryDirectory() as tmp:
        harness = module.AgentTaskHarness(budget=budget, outdir=tmp)
    (lo, hi), *rest = harness.bounds
    if any(b != [lo, hi] for b in rest):
        raise ValueError("The brief template only describes a uniform box.")
    template = Template((MAIN / "agent_briefs" / "blind_brief_template.md").read_text())
    return template.substitute(
        num=task_id,
        dim=harness.dim,
        box=f"[{lo:g}, {hi:g}]",
        bounds_repr=f"[[{lo!r}, {hi!r}], ...]",
        budget=budget,
    )


def leaks(sandbox: Path) -> list[str]:
    """Return "<file>: <term>" for every identifying term found."""
    names = {re.sub(r"\d+$", "", key) for key in TASKS}
    terms = names | {n.replace("_", " ") for n in names} | {"f_opt"}
    pattern = re.compile("|".join(re.escape(t) for t in sorted(terms)), re.IGNORECASE)
    found = []
    for path in sorted(sandbox.rglob("*")):
        if path.is_file():
            for match in set(pattern.findall(path.read_bytes().decode("latin-1"))):
                found.append(f"{path.relative_to(sandbox)}: {match}")
    return found


def make_sandbox(task_id: str, budget: int, outdir: Path) -> None:
    brief = render_brief(task_id, budget)
    outdir.mkdir(parents=True)

    (outdir / "agent_briefs").mkdir()
    (outdir / "agent_briefs" / f"task_{task_id}.md").write_text(brief)
    shutil.copy(MAIN / "requirements.txt", outdir)
    (outdir / "src").mkdir()
    shutil.copy(MAIN / "src" / "__init__.py", outdir / "src")
    shutil.copytree(
        MAIN / "src" / "optimizers",
        outdir / "src" / "optimizers",
        ignore=shutil.ignore_patterns("__pycache__", *POOL_ONLY_FILES),
    )
    (outdir / "src" / "agent_harness").mkdir()
    for name in ("__init__.py", "harness.py", f"task_{task_id}.py"):
        shutil.copy(MAIN / "src" / "agent_harness" / name, outdir / "src" / "agent_harness")
    module = importlib.import_module(f"src.agent_harness.task_{task_id}")
    for asset in getattr(module, "SANDBOX_ASSETS", ()):
        (outdir / "src" / "agent_harness" / asset).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MAIN / "src" / "agent_harness" / asset, outdir / "src" / "agent_harness" / asset)
    (outdir / "results").mkdir()

    found = leaks(outdir)
    if found:
        shutil.rmtree(outdir)
        raise SystemExit("Leak check failed; sandbox deleted:\n  " + "\n  ".join(found))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-id", required=True, help='e.g. "005" for src/agent_harness/task_005.py')
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--outdir", type=Path, help="default: task_<NNN>_sandbox at the repo root")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outdir = args.outdir or MAIN.parent / f"task_{args.task_id}_sandbox"
    make_sandbox(args.task_id, args.budget, outdir)
    print(f"Wrote {outdir}; start a fresh agent session there.")
