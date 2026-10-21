"""Fail (exit 1) if the playbook names a benchmark function or suite.
Run before committing playbook changes:  python scripts/check_playbook.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.playbook_loader import find_leaks, version  # noqa: E402

leaks = find_leaks()
for f, term in leaks:
    print(f"LEAK  {f}: '{term}'")
print(f"playbook version {version()}: {'FAIL' if leaks else 'ok'} ({len(leaks)} leaks)")
sys.exit(1 if leaks else 0)
