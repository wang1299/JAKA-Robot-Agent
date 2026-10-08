"""Compatibility entry point for planner and robot diagnostics commands."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

if __name__ == '__main__':
    from jaka_agent.cli.entry import main
    main()
