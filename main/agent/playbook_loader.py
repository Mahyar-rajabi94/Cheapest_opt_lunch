"""Assemble the strategist's system prompt from the markdown playbook, and
compute its version hash (stamped into every run's meta.json).

The playbook is small enough to send whole (ROLE + skills + references).
Once it grows, switch to a read_playbook(name) tool. The action-menu
section of ROLE.md is generated from agent/schema.py, so the docs can't
drift from the code.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from agent.schema import action_menu_markdown

PLAYBOOK_DIR = Path(__file__).resolve().parent / "playbook"

# Benchmark identities must never appear in the playbook (blind-test rule,
# design §8.2 / roadmap Phase 5). Case-insensitive.
FORBIDDEN = [
    r"rosenbrock", r"rastrigin", r"ackley", r"schwefel", r"griewank", r"weierstrass",
    r"katsuura", r"lunacek", r"gallagher", r"styblinski", r"\blevy\b", r"\bbbob\b",
    r"\bcoco\b", r"\bioh\b", r"\bf\d{1,2}\b", r"task[ _-]?0*1\b", r"149\.15",
    r"bent cigar", r"sharp ridge", r"different powers", r"attractive sector",
    r"buche", r"step ellipsoid",
]


def playbook_files() -> list[Path]:
    files = [PLAYBOOK_DIR / "ROLE.md"]
    files += sorted((PLAYBOOK_DIR / "skills").rglob("*.md"))
    return files


def find_leaks() -> list[tuple[str, str]]:
    leaks = []
    for f in playbook_files():
        text = f.read_text().lower()
        for pat in FORBIDDEN:
            for m in re.finditer(pat, text):
                leaks.append((str(f.relative_to(PLAYBOOK_DIR)), m.group(0)))
    return leaks


def version() -> str:
    h = hashlib.sha1()
    for f in playbook_files():
        h.update(str(f.relative_to(PLAYBOOK_DIR)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def system_prompt(allowed_methods: list[str] | None = None, with_playbook: bool = True) -> str:
    role = (PLAYBOOK_DIR / "ROLE.md").read_text()
    role = role.replace("{{ACTION_MENU}}", action_menu_markdown(allowed_methods))
    if not with_playbook:
        # Ablation: role + action menu only, no strategy skills.
        return role
    parts = [role, "\n\n# Skills\n"]
    for f in playbook_files()[1:]:
        rel = f.relative_to(PLAYBOOK_DIR)
        body = re.sub(r"^---\n.*?\n---\n", "", f.read_text(), flags=re.S)
        parts.append(f"\n\n<file path=\"{rel}\">\n{body.strip()}\n</file>")
    return "".join(parts)
