"""Offline module-discovery smoke check used by the packaged-build script."""

from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path

from clipsift import __version__


REQUIRED_MODULES = (
    "accelerate",
    "bitsandbytes",
    "cv2",
    "PIL",
    "torch",
    "transformers",
    "ttkbootstrap",
)


def _safe_message(exc: BaseException) -> str:
    message = str(exc)
    for sensitive, replacement in (
        (str(Path.home()), "%USERPROFILE%"),
        (os.getcwd(), "%WORKING_DIRECTORY%"),
    ):
        if sensitive:
            message = message.replace(sensitive, replacement)
    return message[:1000]


def run_packaged_smoke_check(output_path: Path) -> int:
    """Import packaged dependencies and write diagnostics without loading Gemma."""
    imports: dict[str, str] = {}
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            imports[module_name] = "ok"
        except Exception as exc:
            imports[module_name] = f"{type(exc).__name__}: {_safe_message(exc)}"
    payload = {
        "application": "ClipSift",
        "version": __version__,
        "platform": platform.platform(),
        "imports": imports,
        "success": False,
        "diagnostic_status": "Unavailable",
        "diagnostics": [],
        "model_loaded": False,
    }
    try:
        failures = {name: detail for name, detail in imports.items() if detail != "ok"}
        if failures:
            raise RuntimeError(f"Required module imports failed: {failures}")
        from clipsift.readiness import run_system_check

        report = run_system_check("auto")
        payload["diagnostic_status"] = report.status
        payload["diagnostics"] = [item.__dict__ for item in report.items]
        payload["success"] = True
        return_code = 0
    except Exception as exc:
        payload["failure"] = {"type": type(exc).__name__, "message": _safe_message(exc)}
        return_code = 1
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return 2
    return return_code
