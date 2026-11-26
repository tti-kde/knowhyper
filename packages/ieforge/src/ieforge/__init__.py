"""IEForge: research-first information extraction framework."""

from pathlib import Path

from ieforge.registry import get, list_components, load_entrypoints, register

CONFIGS_DIR: Path = Path(__file__).resolve().parent / "configs"
"""Absolute path to the Hydra configs shipped inside the installed package."""

__all__ = ["CONFIGS_DIR", "get", "list_components", "load_entrypoints", "register"]
