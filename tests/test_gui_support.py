from pathlib import Path

from clipsift.gui_support import (
    GuiPreferences,
    control_states,
    filter_results,
    load_preferences,
    open_local_path,
    save_preferences,
    validate_preferences,
)
from clipsift.reporting import ClipReportRow
from clipsift.scanner import VideoScanResult


def _result(status: str, error: str = "") -> VideoScanResult:
    row = ClipReportRow("video.mp4", status, 1.0, 1, 0, 0, "", None, "", "", "", error)
    return VideoScanResult(row, None, "", 1.0)


def test_result_filtering_keeps_source_results() -> None:
    source = [_result("Person Detected"), _result("Needs Review"), _result("No Person Detected"), _result("Needs Review", "bad video")]
    assert len(filter_results(source, "All")) == 4
    assert len(filter_results(source, "Person Detected")) == 1
    assert len(filter_results(source, "Needs Review")) == 1
    assert len(filter_results(source, "Errors")) == 1
    assert len(source) == 4


def test_settings_round_trip_and_validation(tmp_path: Path) -> None:
    input_folder = tmp_path / "input"; input_folder.mkdir()
    output_folder = tmp_path / "output"; output_folder.mkdir()
    path = tmp_path / "settings.json"
    expected = GuiPreferences(str(input_folder), str(output_folder), "GPU", "Motion", 24, "1200x850+10+20")
    save_preferences(expected, path)
    assert load_preferences(path) == expected


def test_invalid_and_corrupt_settings_fall_back_safely(tmp_path: Path) -> None:
    invalid = validate_preferences({"input_folder": str(tmp_path / "missing"), "device": "TPU", "max_frames": -2, "window_geometry": "bad"})
    assert invalid == GuiPreferences()
    path = tmp_path / "settings.json"; path.write_text("{broken", encoding="utf-8")
    assert load_preferences(path) == GuiPreferences()


def test_control_state_transitions() -> None:
    assert control_states(True) == {"configuration": "disabled", "readonly_configuration": "disabled", "start": "disabled", "cancel": "normal"}
    assert control_states(False)["configuration"] == "normal"
    assert control_states(False)["cancel"] == "disabled"


def test_safe_path_opening_checks_existence_and_uses_direct_opener(tmp_path: Path) -> None:
    opened = []
    target = tmp_path / "a file ü.txt"; target.write_text("test", encoding="utf-8")
    assert open_local_path(tmp_path / "missing", opened.append)[0] is False
    assert open_local_path(target, opened.append) == (True, "")
    assert opened == [str(target.resolve())]
