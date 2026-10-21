"""decisions.jsonl: one line per round (design §3.4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DecisionLog:
    def __init__(self):
        self.entries: list[dict] = []

    def record(self, report: dict, action: dict, source: str, note: str = "") -> None:
        h = hashlib.sha1(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.entries.append({
            "round": report["round"],
            "evals_used": report["evals_used"],
            "incumbent_y": (report.get("incumbent") or {}).get("y"),
            "report_hash": h,
            "action": action,
            "source": source,          # policy name, or "fallback"
            "note": note,
            "expectation_check": report.get("last_action_check"),
        })

    def save(self, path) -> None:
        with open(Path(path), "w") as f:
            for e in self.entries:
                f.write(json.dumps(e, default=str) + "\n")
