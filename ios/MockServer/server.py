"""Compatibility entry point for the shared v2 fixture server."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "tools/pilot_mock.py"), run_name="__main__")
