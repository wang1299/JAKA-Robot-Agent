"""Compatibility entry point; implementation lives in src/jaka_agent/web."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from jaka_agent.web.app import main

if __name__ == '__main__':
    main()
