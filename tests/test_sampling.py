from pathlib import Path

import pytest

import clipsift.video as video
from clipsift.video import VideoMetadata, select_motion_peaks, select_video_timestamps, uniform_timestamps


def _metadata(duration: float) -> VideoMetadata:
    fps = 10.0
    return VideoMetadata(Path("test.mp4"), fps, int(duration * fps), duration)


@pytest.mark.parametrize("duration", [60.0, 120.0])
def test_uniform_sampling_covers_complete_duration(duration: float) -> None:
    timestamps = uniform_timestamps(_metadata(duration), 12)
    assert timestamps[0] == 0.0
    assert timestamps[-1] > duration * 0.95
    assert any(duration * 0.4 < value < duration * 0.6 for value in timestamps)
    assert len(timestamps) == 12


def test_uniform_timestamps_do_not_cluster_at_start() -> None:
    timestamps = uniform_timestamps(_metadata(60), 20)
    assert timestamps[10] > 25
    assert timestamps[-1] > 59


def test_motion_peaks_are_prioritised_and_nearby_duplicates_removed() -> None:
    scored = [
        (1.0, 1.0), (2.0, 1.1), (3.0, 0.9), (4.0, 1.2),
        (10.0, 20.0), (10.5, 19.0), (30.0, 15.0), (40.0, 1.0),
    ]
    selected = select_motion_peaks(scored, 3, minimum_spacing_seconds=2.0)
    assert (10.0, 20.0) in selected
    assert (30.0, 15.0) in selected
    assert not any(timestamp == 10.5 for timestamp, _ in selected)


def test_hybrid_includes_full_coverage_and_motion(monkeypatch) -> None:
    metadata = _metadata(60)
    monkeypatch.setattr(video, "motion_scores", lambda *args: [(15.0, 20.0), (45.0, 18.0), (46.0, 17.0)])
    selected = select_video_timestamps(Path("test.mp4"), metadata, "hybrid", 8)
    timestamps = [item.timestamp_seconds for item in selected]
    assert timestamps[0] == 0.0
    assert timestamps[-1] > 59
    assert any("motion" in item.sources for item in selected)
    assert len(selected) <= 8


def test_invalid_metadata_fails_safely() -> None:
    with pytest.raises(ValueError, match="duration, FPS"):
        uniform_timestamps(VideoMetadata(Path("bad.mp4"), 0.0, 0, 0.0), 12)
