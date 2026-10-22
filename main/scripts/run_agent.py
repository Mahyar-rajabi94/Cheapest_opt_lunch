"""Entry point for the canonical, isolated native-CLI agent."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blind_runner.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
