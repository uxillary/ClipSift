"""Display-independent helpers for the ClipSift Windows GUI."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from clipsift.scanner import VideoScanResult


DEVICE_CHOICES = {"Auto", "GPU", "CPU"}
STRATEGY_CHOICES = {"Hybrid", "Uniform", "Motion"}
FILTER_CHOICES = {"All", "Person Detected", "Needs Review", "No Person Detected", "Errors"}


@dataclass(frozen=True)
class GuiPreferences:
    input_folder: str = ""
    output_folder: str = ""
    device: str = "Auto"
    sampling_strategy: str = "Hybrid"
    max_frames: int = 12
    window_geometry: str = "1180x800"


def preferences_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "ClipSift" / "settings.json"
    return Path.home() / "AppData" / "Local" / "ClipSift" / "settings.json"


def validate_preferences(data: object) -> GuiPreferences:
    defaults = GuiPreferences()
    if not isinstance(data, dict):
        return defaults
    input_value = data.get("input_folder", "")
    input_folder = input_value if isinstance(input_value, str) and Path(input_value).is_dir() else ""
    output_value = data.get("output_folder", "")
    output_folder = output_value if isinstance(output_value, str) and (not output_value or Path(output_value).is_dir()) else ""
    device = data.get("device") if data.get("device") in DEVICE_CHOICES else defaults.device
    strategy = data.get("sampling_strategy") if data.get("sampling_strategy") in STRATEGY_CHOICES else defaults.sampling_strategy
    max_frames = data.get("max_frames")
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or not 1 <= max_frames <= 100:
        max_frames = defaults.max_frames
    geometry = data.get("window_geometry")
    if not isinstance(geometry, str) or not re.fullmatch(r"\d{3,4}x\d{3,4}(?:[+-]\d+[+-]\d+)?", geometry):
        geometry = defaults.window_geometry
    return GuiPreferences(input_folder, output_folder, device, strategy, max_frames, geometry)


def load_preferences(path: Path | None = None) -> GuiPreferences:
    target = path or preferences_path()
    try:
        return validate_preferences(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return GuiPreferences()


def save_preferences(preferences: GuiPreferences, path: Path | None = None) -> None:
    target = path or preferences_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(preferences), indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def filter_results(results: Iterable[VideoScanResult], selected_filter: str) -> list[VideoScanResult]:
    if selected_filter not in FILTER_CHOICES:
        selected_filter = "All"
    items = list(results)
    if selected_filter == "All":
        return items
    if selected_filter == "Errors":
        return [item for item in items if bool(item.row.error)]
    return [item for item in items if not item.row.error and item.row.status == selected_filter]


def control_states(scanning: bool) -> dict[str, str]:
    return {
        "configuration": "disabled" if scanning else "normal",
        "readonly_configuration": "disabled" if scanning else "readonly",
        "start": "disabled" if scanning else "normal",
        "cancel": "normal" if scanning else "disabled",
    }


def open_local_path(path: Path | str, opener: Callable[[str], object] | None = None) -> tuple[bool, str]:
    target = Path(path)
    if not target.exists():
        return False, f"Path does not exist: {target}"
    open_function = opener or getattr(os, "startfile", None)
    if open_function is None:
        return False, "Opening files is unavailable on this platform."
    try:
        open_function(str(target.resolve()))
        return True, ""
    except OSError as exc:
        return False, f"Could not open {target}: {exc}"
