"""Shared stderr helpers for CLIs (Rich, no ANSI when not a TTY)."""

from __future__ import annotations

from rich.console import Console

_err = Console(stderr=True, highlight=False, color_system=None, soft_wrap=True)


def cli_error(message: str) -> None:
    _err.print(f"error: {message}")


def cli_warning(message: str) -> None:
    _err.print(f"warning: {message}")
