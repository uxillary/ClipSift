"""Streaming video inspection and frame sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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


def save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise ValueError(f"Could not save evidence frame: {path}")
