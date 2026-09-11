"""Reusable folder-scan orchestration for the ClipSift CLI and GUI."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Iterator

from clipsift.classification import ClipStatus, FrameAssessment, decide_clip, parse_model_response
from clipsift.inference import CudaOutOfMemoryError, DEFAULT_MODEL_NAME, GemmaVisionModel
from clipsift.reporting import (
    Benchmark,
    ClipReportRow,
    create_output_dirs,
    row_from_decision,
    write_benchmark_json,
    write_report_csv,
)
from clipsift.video import (
    SampledFrame,
    find_videos,
    frames_at_timestamps,
    read_metadata,
    sample_frames,
    save_frame,
    select_video_timestamps,
)


@dataclass(frozen=True)
class ScanConfig:
    input_folder: Path
    output_folder: Path
    device: str = "auto"
    preset: str = "safe"
    sampling_strategy: str | None = "hybrid"
    max_frames: int = 12
    max_frame_size: int = 896
    motion_interval_seconds: float = 0.5
    samples_per_second: float = 1.0
    model_name: str = DEFAULT_MODEL_NAME


@dataclass(frozen=True)
class VideoScanResult:
    row: ClipReportRow
    trigger_timestamp: float | None
    trigger_confidence: str
    elapsed_seconds: float


@dataclass(frozen=True)
class ScanEvent:
    kind: str
    message: str = ""
    filename: str = ""
    current: int = 0
    total: int = 0
    progress: float = 0.0
    result: VideoScanResult | None = None


@dataclass(frozen=True)
class ScanSummary:
    rows: tuple[ClipReportRow, ...]
    cancelled: bool
    device: str
    device_name: str
    elapsed_seconds: float


EventHandler = Callable[[ScanEvent], None]


def _emit(handler: EventHandler | None, event: ScanEvent) -> None:
    if handler is not None:
        handler(event)


def _frames_for_video(config: ScanConfig, video_path: Path) -> Iterator[SampledFrame]:
    if config.sampling_strategy is None:
        yield from sample_frames(video_path, config.samples_per_second, config.max_frame_size)
        return
    metadata = read_metadata(video_path)
    selections = select_video_timestamps(
        video_path,
        metadata,
        config.sampling_strategy,
        config.max_frames,
        config.motion_interval_seconds,
    )
    yield from frames_at_timestamps(video_path, selections, config.max_frame_size)


def resolve_trigger_assessment(
    assessments: list[FrameAssessment], status: ClipStatus
) -> FrameAssessment | None:
    """Return the assessment responsible for evidence and GUI trigger details."""
    if status == ClipStatus.PERSON_DETECTED:
        return next((item for item in assessments if item.person_status == "present"), None)

    if status == ClipStatus.NEEDS_REVIEW:
        return next((item for item in assessments if item.review_required), None)

    return None


def resolve_trigger_metadata(
    assessments: list[FrameAssessment], status: ClipStatus
) -> tuple[float | None, str]:
    """Return GUI timestamp/confidence according to the clip classification."""
    trigger = resolve_trigger_assessment(assessments, status)
    if trigger is not None:
        return trigger.timestamp_seconds, trigger.confidence

    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    absent = [item for item in assessments if item.person_status == "absent" and item.parse_error is None]
    strongest_absent = max(absent, key=lambda item: confidence_rank.get(item.confidence, -1), default=None)
    return None, strongest_absent.confidence if strongest_absent else ""


def scan_folder(
    config: ScanConfig,
    on_event: EventHandler | None = None,
    cancel_event: Event | None = None,
    model_factory: Callable[..., GemmaVisionModel] = GemmaVisionModel,
) -> ScanSummary:
    """Scan a folder while reporting UI-neutral events.

    Model inference runs in the caller's thread. GUI callers must invoke this
    function from a worker thread and marshal events onto the Tk main thread.
    """
    cancel = cancel_event or Event()
    videos = find_videos(config.input_folder)
    if not videos:
        raise ValueError(f"No supported videos found in {config.input_folder}")
    if config.max_frames <= 0:
        raise ValueError("Maximum frames must be greater than zero.")

    output_paths = create_output_dirs(config.output_folder)
    rows: list[ClipReportRow] = []
    inference_times: list[float] = []
    source_seconds = 0.0
    started = time.perf_counter()
    model = None
    device = device_name = "unknown"
    _emit(on_event, ScanEvent("scan_started", total=len(videos), message=f"Found {len(videos)} video(s)."))

    try:
        if cancel.is_set():
            return ScanSummary(tuple(rows), True, device, device_name, time.perf_counter() - started)
        _emit(on_event, ScanEvent("operation", message="Loading Gemma model", total=len(videos)))
        model = model_factory(config.model_name, config.device, config.preset)
        device = model.device_info.device
        device_name = model.device_info.device_name
        _emit(on_event, ScanEvent("model_loaded", message=f"Using {device} ({device_name})", total=len(videos)))

        for index, video_path in enumerate(videos, start=1):
            if cancel.is_set():
                break
            video_started = time.perf_counter()
            _emit(on_event, ScanEvent("video_started", "Selecting frames", video_path.name, index, len(videos), (index - 1) / len(videos)))
            assessments: list[FrameAssessment] = []
            frame_lookup = {}
            duration = 0.0
            try:
                metadata = read_metadata(video_path)
                duration = metadata.duration_seconds
                if metadata.frame_rate <= 0 or metadata.frame_count <= 0 or duration <= 0:
                    raise ValueError(f"Invalid duration/FPS metadata for video: {video_path}")
                for sampled in _frames_for_video(config, video_path):
                    if cancel.is_set():
                        break
                    _emit(on_event, ScanEvent("operation", f"Analysing frame at {sampled.timestamp_seconds:.1f}s", video_path.name, index, len(videos), (index - 1) / len(videos)))
                    inference = model.analyse_frame(sampled.image)
                    assessment = parse_model_response(inference.raw_response, sampled.timestamp_seconds)
                    assessments.append(assessment)
                    frame_lookup[sampled.timestamp_seconds] = sampled.image
                    inference_times.append(inference.inference_seconds)
                    _emit(on_event, ScanEvent("frame", f"{assessment.person_status}, {assessment.assessment_confidence}", video_path.name, index, len(videos), (index - 1) / len(videos)))
                    if assessment.person_status == "present" and assessment.assessment_confidence in {"medium", "high"}:
                        break

                if cancel.is_set() and not assessments:
                    break
                decision = decide_clip(assessments)
                trigger_assessment = resolve_trigger_assessment(assessments, decision.status)
                evidence_path = None
                if trigger_assessment is not None:
                    evidence_path = output_paths["evidence"] / f"{video_path.stem}_{trigger_assessment.timestamp_seconds:.1f}s.jpg"
                    save_frame(evidence_path, frame_lookup[trigger_assessment.timestamp_seconds])
                if decision.status == ClipStatus.PERSON_DETECTED:
                    shutil.copy2(video_path, output_paths["flagged"] / video_path.name)
                elif decision.status == ClipStatus.NEEDS_REVIEW:
                    shutil.copy2(video_path, output_paths["needs_review"] / video_path.name)
                row = row_from_decision(video_path, duration, decision, evidence_path)
                trigger_timestamp, trigger_confidence = resolve_trigger_metadata(assessments, decision.status)
                result = VideoScanResult(row, trigger_timestamp, trigger_confidence, time.perf_counter() - video_started)
                rows.append(row)
                source_seconds += duration
                write_report_csv(config.output_folder / "report.csv", rows)
                _emit(on_event, ScanEvent("video_completed", decision.status.value, video_path.name, index, len(videos), index / len(videos), result))
            except CudaOutOfMemoryError:
                raise
            except Exception as exc:
                decision = decide_clip([])
                row = row_from_decision(video_path, duration, decision, None, str(exc))
                rows.append(row)
                write_report_csv(config.output_folder / "report.csv", rows)
                result = VideoScanResult(row, None, "", time.perf_counter() - video_started)
                _emit(on_event, ScanEvent("video_error", str(exc), video_path.name, index, len(videos), index / len(videos), result))

        elapsed = time.perf_counter() - started
        average = sum(inference_times) / len(inference_times) if inference_times else 0.0
        write_report_csv(config.output_folder / "report.csv", rows)
        write_benchmark_json(config.output_folder / "benchmark.json", Benchmark(
            device_name=device_name,
            device_type=device,
            model_name=config.model_name,
            videos_processed=sum(1 for row in rows if not row.error),
            video_minutes_processed=round(source_seconds / 60.0, 3),
            frames_analysed=sum(row.frames_analysed for row in rows),
            total_processing_time_seconds=round(elapsed, 3),
            average_inference_time_seconds=round(average, 4),
            approximate_frames_per_second=round(sum(row.frames_analysed for row in rows) / elapsed, 4) if elapsed else 0.0,
        ))
        cancelled = cancel.is_set()
        _emit(on_event, ScanEvent("cancelled" if cancelled else "completed", "Scan cancelled." if cancelled else "Scan complete.", total=len(videos), progress=1.0 if not cancelled else len(rows) / len(videos)))
        return ScanSummary(tuple(rows), cancelled, device, device_name, elapsed)
    finally:
        if model is not None:
            model.close()
