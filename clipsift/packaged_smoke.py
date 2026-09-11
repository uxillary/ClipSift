"""Offline module-discovery smoke check used by the packaged-build script."""

from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path

from clipsift import __version__
from clipsift.readiness import run_system_check


REQUIRED_MODULES = (
    "accelerate",
    "bitsandbytes",
    "cv2",
    "PIL",
    "torch",
    "transformers",
    "ttkbootstrap",
)


def run_packaged_smoke_check(output_path: Path) -> None:
    """Import packaged dependencies and write diagnostics without loading Gemma."""
    imports: dict[str, str] = {}
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            imports[module_name] = "ok"
        except Exception as exc:
            imports[module_name] = f"{type(exc).__name__}: {exc}"
    report = run_system_check("auto")
    payload = {
        "application": "ClipSift",
        "version": __version__,
        "platform": platform.platform(),
        "imports": imports,
        "diagnostic_status": report.status,
        "diagnostics": [item.__dict__ for item in report.items],
        "model_loaded": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failures = {name: detail for name, detail in imports.items() if detail != "ok"}
    if failures:
        raise RuntimeError(f"Packaged module imports failed: {failures}")
