from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np

import clipsift.scanner as scanner
from clipsift.classification import ClipStatus, parse_model_response
from clipsift.inference import InferenceResult
from clipsift.scanner import ScanConfig, resolve_trigger_metadata, scan_folder
from clipsift.video import SampledFrame, VideoMetadata


class FakeModel:
    loads = 0
    responses: list[str] = []

    def __init__(self, *args):
        type(self).loads += 1
        self.device_info = SimpleNamespace(device="cpu", device_name="CPU")

    def analyse_frame(self, frame):
        return InferenceResult(type(self).responses.pop(0), 0.1)

    def close(self):
        pass


def _files(tmp_path: Path, count: int = 2) -> Path:
    input_folder = tmp_path / "input"
    input_folder.mkdir()
    for index in range(count):
        (input_folder / f"clip-{index}.mp4").write_bytes(b"video")
    return input_folder


def _metadata(path: Path) -> VideoMetadata:
    return VideoMetadata(path, 10.0, 100, 10.0)


def _frame(path, *args):
    yield SampledFrame(np.zeros((2, 2, 3), dtype=np.uint8), 0.0, 0)


def test_scan_service_loads_once_writes_reports_and_emits_events(monkeypatch, tmp_path: Path) -> None:
    input_folder = _files(tmp_path)
    FakeModel.loads = 0
    FakeModel.responses = [
        '{"person_status":"present","assessment_confidence":"high","description":"Person"}',
        '{"person_status":"absent","assessment_confidence":"high","description":"Empty"}',
    ]
    monkeypatch.setattr(scanner, "read_metadata", _metadata)
    monkeypatch.setattr(scanner, "sample_frames", _frame)
    monkeypatch.setattr(scanner, "save_frame", lambda path, frame: path.write_bytes(b"image"))
    events = []

    summary = scan_folder(
        ScanConfig(input_folder, tmp_path / "output", sampling_strategy=None),
        on_event=events.append,
        model_factory=FakeModel,
    )

    assert FakeModel.loads == 1
    assert len(summary.rows) == 2
    assert (tmp_path / "output" / "report.csv").is_file()
    assert (tmp_path / "output" / "benchmark.json").is_file()
    assert [event.kind for event in events].count("video_completed") == 2
    assert events[-1].kind == "completed"


def test_unreadable_video_emits_error_and_scan_continues(monkeypatch, tmp_path: Path) -> None:
    input_folder = _files(tmp_path)
    FakeModel.loads = 0
    FakeModel.responses = ['{"person_status":"absent","assessment_confidence":"high","description":"Empty"}']

    def metadata(path: Path):
        if path.name == "clip-0.mp4":
            raise ValueError("Unreadable video")
        return _metadata(path)

    monkeypatch.setattr(scanner, "read_metadata", metadata)
    monkeypatch.setattr(scanner, "sample_frames", _frame)
    events = []
    summary = scan_folder(ScanConfig(input_folder, tmp_path / "output", sampling_strategy=None), events.append, model_factory=FakeModel)

    assert len(summary.rows) == 2
    assert summary.rows[0].error == "Unreadable video"
    assert summary.rows[1].error == ""
    assert any(event.kind == "video_error" for event in events)
    assert events[-1].kind == "completed"


def test_cancellation_preserves_completed_result(monkeypatch, tmp_path: Path) -> None:
    input_folder = _files(tmp_path)
    cancel = Event()
    FakeModel.loads = 0
    FakeModel.responses = ['{"person_status":"absent","assessment_confidence":"high","description":"Empty"}']
    monkeypatch.setattr(scanner, "read_metadata", _metadata)
    monkeypatch.setattr(scanner, "sample_frames", _frame)
    original = FakeModel.analyse_frame

    def analyse_and_cancel(self, frame):
        result = original(self, frame)
        cancel.set()
        return result

    monkeypatch.setattr(FakeModel, "analyse_frame", analyse_and_cancel)
    events = []
    summary = scan_folder(
        ScanConfig(input_folder, tmp_path / "output", sampling_strategy=None),
        events.append,
        cancel,
        FakeModel,
    )

    assert summary.cancelled is True
    assert len(summary.rows) == 1
    assert (tmp_path / "output" / "report.csv").is_file()
    assert events[-1].kind == "cancelled"


def test_person_trigger_uses_first_present_frame_not_earlier_review() -> None:
    assessments = [
        parse_model_response('{"person_status":"uncertain","assessment_confidence":"low","description":"Possible"}', 0.0),
        parse_model_response('{"person_status":"present","assessment_confidence":"medium","description":"Person"}', 35.0),
    ]
    assert resolve_trigger_metadata(assessments, ClipStatus.PERSON_DETECTED) == (35.0, "medium")


def test_needs_review_timestamp_and_confidence_share_uncertain_trigger() -> None:
    assessments = [
        parse_model_response('{"person_status":"absent","assessment_confidence":"high","description":"Empty"}', 0.0),
        parse_model_response('{"person_status":"uncertain","assessment_confidence":"low","description":"Possible"}', 12.5),
    ]
    assert resolve_trigger_metadata(assessments, ClipStatus.NEEDS_REVIEW) == (12.5, "low")


def test_no_person_has_no_trigger_and_strongest_absent_confidence() -> None:
    assessments = [
        parse_model_response('{"person_status":"absent","assessment_confidence":"medium","description":"Empty"}', 0.0),
        parse_model_response('{"person_status":"absent","assessment_confidence":"high","description":"Clear empty scene"}', 12.5),
    ]
    assert resolve_trigger_metadata(assessments, ClipStatus.NO_PERSON_DETECTED) == (None, "high")


def test_malformed_response_is_trigger_with_its_own_confidence() -> None:
    assessments = [
        parse_model_response('{"person_status":"absent","assessment_confidence":"high","description":"Empty"}', 0.0),
        parse_model_response("not valid JSON", 8.0),
    ]
    timestamp, confidence = resolve_trigger_metadata(assessments, ClipStatus.NEEDS_REVIEW)
    assert (timestamp, confidence) == (8.0, "low")
