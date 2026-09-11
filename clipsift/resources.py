"""Resolve tracked application resources in source and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path


def resource_path(relative_path: str, bundle_root: Path | None = None) -> Path:
    """Return a resource path without assuming the current working directory."""
    root = bundle_root
    if root is None:
        frozen_root = getattr(sys, "_MEIPASS", None)
        root = Path(frozen_root) if frozen_root else Path(__file__).resolve().parent.parent
    return root / relative_path
