"""Locate KnowHyper's packaged Hydra configs for ieforge's search path."""

from __future__ import annotations

from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
"""Absolute path to the Hydra configs shipped inside the installed package."""
