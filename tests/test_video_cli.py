from pathlib import Path
from types import SimpleNamespace

import numpy as np

from clipsift import cli
from clipsift.inference import InferenceResult
from clipsift.video import SampledFrame, SelectedTimestamp, VideoMetadata


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "test.mp4"
    path.write_bytes(b"test")
    return path


def _metadata(path: Path) -> VideoMetadata:
    return VideoMetadata(path, frame_rate=10.0, frame_count=100, duration_seconds=10.0)


def _frames(count: int):
    for index in range(count):
        yield SampledFrame(np.zeros((2, 2, 3), dtype=np.uint8), index * 2.0, index * 20)


class _FakeModel:
    loads = 0
    calls = 0
    responses: list[str] = []

    def __init__(self, *args):
        type(self).loads += 1
        self.device_info = SimpleNamespace(device="cuda:0", device_name="Test GPU")

    def analyse_frame(self, frame):
        response = type(self).responses[type(self).calls]
        type(self).calls += 1
        return InferenceResult(response, 0.25, 1024)

    def close(self):
        pass


def _configure(monkeypatch, responses: list[str], frame_count: int = 5) -> None:
    _FakeModel.loads = _FakeModel.calls = 0
    _FakeModel.responses = responses
    monkeypatch.setattr(cli, "GemmaVisionModel", _FakeModel)
    monkeypatch.setattr(cli, "read_metadata", _metadata)
    selections = [SelectedTimestamp(index * 2.0, ("uniform",)) for index in range(frame_count)]
    monkeypatch.setattr(cli, "select_video_timestamps", lambda *args: selections)
    monkeypatch.setattr(cli, "frames_at_timestamps", lambda *args: _frames(frame_count))


def test_video_enforces_max_frames_and_loads_model_once(monkeypatch, tmp_path: Path) -> None:
    absent = '{"person_status":"absent","assessment_confidence":"high","description":"Empty"}'
    _configure(monkeypatch, [absent] * 3)
    assert cli.main(["test-video", str(_video(tmp_path)), "--max-frames", "3"]) == 0
    assert _FakeModel.loads == 1
    assert _FakeModel.calls == 3


def test_video_early_stops_and_selects_timestamp(monkeypatch, tmp_path: Path, capsys) -> None:
    responses = [
        '{"person_status":"absent","assessment_confidence":"high","description":"Empty"}',
        '{"person_status":"present","assessment_confidence":"medium","description":"Person"}',
        '{"person_status":"absent","assessment_confidence":"high","description":"Unused"}',
    ]
    _configure(monkeypatch, responses)
    assert cli.main(["test-video", str(_video(tmp_path)), "--max-frames", "3"]) == 0
    output = capsys.readouterr().out
    assert _FakeModel.calls == 2
    assert "First relevant timestamp: 2.000s" in output
    assert "Classification: Person Detected" in output
    assert "Early stop" in output


def test_unreadable_video_does_not_load_model(monkeypatch, tmp_path: Path, capsys) -> None:
    _FakeModel.loads = 0
    monkeypatch.setattr(cli, "GemmaVisionModel", _FakeModel)
    monkeypatch.setattr(cli, "read_metadata", lambda path: (_ for _ in ()).throw(ValueError("Could not open video")))
    assert cli.main(["test-video", str(_video(tmp_path))]) == 2
    assert _FakeModel.loads == 0
    assert "Could not open video" in capsys.readouterr().err


def test_dry_run_never_loads_model(monkeypatch, tmp_path: Path, capsys) -> None:
    absent = '{"person_status":"absent","assessment_confidence":"high","description":"Empty"}'
    _configure(monkeypatch, [absent])
    assert cli.main(["test-video", str(_video(tmp_path)), "--dry-run"]) == 0
    assert _FakeModel.loads == 0
    assert "Gemma was not loaded" in capsys.readouterr().out
