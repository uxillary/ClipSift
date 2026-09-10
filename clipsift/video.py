"""Streaming video inspection and frame sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    frame_rate: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class SampledFrame:
    image: np.ndarray
    timestamp_seconds: float
    frame_index: int


@dataclass(frozen=True)
class SelectedTimestamp:
    timestamp_seconds: float
    sources: tuple[str, ...]
    motion_score: float | None = None


def find_videos(input_folder: Path) -> list[Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    videos = [
        path
        for path in sorted(input_folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return videos


def read_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = frame_count / frame_rate if frame_rate > 0 else 0.0
        return VideoMetadata(
            path=video_path,
            frame_rate=frame_rate,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
        )
    finally:
        capture.release()


def resize_preserving_aspect(frame: np.ndarray, max_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if max(height, width) <= max_size:
        return frame
    scale = max_size / float(max(height, width))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def sample_frames(video_path: Path, samples_per_second: float, max_size: int) -> Iterator[SampledFrame]:
    """Yield resized frames without loading the whole video into memory."""

    if samples_per_second <= 0:
        raise ValueError("samples_per_second must be greater than zero")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_rate <= 0:
            raise ValueError(f"Could not determine frame rate for video: {video_path}")

        step_frames = max(1, round(frame_rate / samples_per_second))
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step_frames == 0:
                yield SampledFrame(
                    image=resize_preserving_aspect(frame, max_size=max_size),
                    timestamp_seconds=frame_index / frame_rate,
                    frame_index=frame_index,
                )
            frame_index += 1
    finally:
        capture.release()


def uniform_timestamps(metadata: VideoMetadata, max_frames: int) -> list[float]:
    """Distribute timestamps over the complete readable frame range."""
    if metadata.frame_rate <= 0 or metadata.frame_count <= 0 or metadata.duration_seconds <= 0:
        raise ValueError("Video duration, FPS and frame count must all be positive.")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero.")
    readable_end = max(0.0, (metadata.frame_count - 2) / metadata.frame_rate)
    count = min(max_frames, max(1, metadata.frame_count - 1))
    if count == 1:
        return [0.0]
    return [readable_end * index / (count - 1) for index in range(count)]


def motion_scores(
    video_path: Path,
    metadata: VideoMetadata,
    interval_seconds: float = 0.5,
    analysis_width: int = 160,
) -> list[tuple[float, float]]:
    """Return lightweight grayscale frame-difference scores; motion is not detection."""
    if interval_seconds <= 0:
        raise ValueError("motion interval must be greater than zero.")
    capture = cv2.VideoCapture(str(video_path))
    scores: list[tuple[float, float]] = []
    previous: np.ndarray | None = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        for timestamp in np.arange(0.0, metadata.duration_seconds, interval_seconds):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            target_height = max(1, round(height * analysis_width / width))
            small = cv2.resize(frame, (analysis_width, target_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if previous is not None:
                score = float(np.mean(cv2.absdiff(gray, previous)))
                scores.append((float(timestamp), score))
            previous = gray
    finally:
        capture.release()
    return scores


def select_motion_peaks(
    scored: Sequence[tuple[float, float]],
    max_frames: int,
    minimum_spacing_seconds: float = 1.5,
) -> list[tuple[float, float]]:
    """Choose well-spaced peaks above a conservative noise threshold."""
    if max_frames <= 0 or not scored:
        return []
    values = np.asarray([score for _, score in scored], dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # Scores are means over blurred 160px grayscale frames, so genuine scene
    # changes may remain below one intensity level. Keep a small absolute floor
    # while requiring a robust relative departure from each camera's baseline.
    threshold = max(0.25, median + 2.0 * mad)
    ranked = sorted((item for item in scored if item[1] >= threshold), key=lambda item: item[1], reverse=True)
    selected: list[tuple[float, float]] = []
    for timestamp, score in ranked:
        if all(abs(timestamp - existing) >= minimum_spacing_seconds for existing, _ in selected):
            selected.append((timestamp, score))
            if len(selected) >= max_frames:
                break
    return sorted(selected)


def select_video_timestamps(
    video_path: Path,
    metadata: VideoMetadata,
    strategy: str,
    max_frames: int,
    motion_interval_seconds: float = 0.5,
) -> list[SelectedTimestamp]:
    """Plan uniform, motion-aware or hybrid timestamps in chronological order."""
    if strategy not in {"uniform", "motion", "hybrid"}:
        raise ValueError("sampling strategy must be uniform, motion, or hybrid")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    uniform_count = max_frames if strategy == "uniform" else min(max_frames, max(3, (max_frames + 1) // 2))
    uniform = uniform_timestamps(metadata, uniform_count) if strategy != "motion" else []
    scored = motion_scores(video_path, metadata, motion_interval_seconds) if strategy != "uniform" else []
    motion_limit = max_frames if strategy == "motion" else max_frames - len(uniform)
    spacing = max(1.5, metadata.duration_seconds / max(max_frames * 2, 1))
    motion = select_motion_peaks(scored, motion_limit, spacing)

    if strategy == "motion" and not motion:
        uniform = uniform_timestamps(metadata, max_frames)

    merged: list[SelectedTimestamp] = []
    tolerance = min(0.25, motion_interval_seconds / 2)
    for timestamp, source, score in [
        *((value, "uniform", None) for value in uniform),
        *((value, "motion", score) for value, score in motion),
    ]:
        duplicate = next((index for index, item in enumerate(merged) if abs(item.timestamp_seconds - timestamp) <= tolerance), None)
        if duplicate is None:
            merged.append(SelectedTimestamp(timestamp, (source,), score))
        else:
            item = merged[duplicate]
            sources = tuple(dict.fromkeys((*item.sources, source)))
            merged[duplicate] = SelectedTimestamp(item.timestamp_seconds, sources, score if score is not None else item.motion_score)
    return sorted(merged, key=lambda item: item.timestamp_seconds)[:max_frames]


def frames_at_timestamps(video_path: Path, selections: Sequence[SelectedTimestamp], max_size: int) -> Iterator[SampledFrame]:
    """Decode only selected timestamps from a video, in reporting order."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        for selection in selections:
            capture.set(cv2.CAP_PROP_POS_MSEC, selection.timestamp_seconds * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode selected frame at {selection.timestamp_seconds:.3f}s")
            actual_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or selection.timestamp_seconds * 1000.0)
            yield SampledFrame(
                resize_preserving_aspect(frame, max_size),
                selection.timestamp_seconds,
                max(0, round(actual_ms / 1000.0 * float(capture.get(cv2.CAP_PROP_FPS) or 0.0))),
            )
    finally:
        capture.release()


def save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise ValueError(f"Could not save evidence frame: {path}")
