"""Report and benchmark writers for ClipSift."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from clipsift.classification import ClipDecision


@dataclass(frozen=True)
class ClipReportRow:
    video_path: str
    status: str
    duration_seconds: float
    frames_analysed: int
    person_votes: int
    review_votes: int
    evidence_frame: str
    strongest_timestamp_seconds: float | None
    strongest_confidence: str
    description: str
    rationale: str
    error: str


@dataclass(frozen=True)
class Benchmark:
    device_name: str
    device_type: str
    model_name: str
    videos_processed: int
    video_minutes_processed: float
    frames_analysed: int
    total_processing_time_seconds: float
    average_inference_time_seconds: float
    approximate_frames_per_second: float
    note: str = "Automated review aid benchmark. Values are recorded from this run only."


def create_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "flagged": output_dir / "flagged_clips",
        "needs_review": output_dir / "needs_review",
        "evidence": output_dir / "evidence_frames",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def row_from_decision(
    video_path: Path,
    duration_seconds: float,
    decision: ClipDecision,
    evidence_frame: Path | None,
    error: str = "",
) -> ClipReportRow:
    strongest = decision.strongest_frame
    return ClipReportRow(
        video_path=str(video_path),
        status=decision.status.value,
        duration_seconds=round(duration_seconds, 3),
        frames_analysed=decision.frames_analysed,
        person_votes=decision.person_votes,
        review_votes=decision.review_votes,
        evidence_frame=str(evidence_frame) if evidence_frame else "",
        strongest_timestamp_seconds=round(strongest.timestamp_seconds, 3) if strongest else None,
        strongest_confidence=strongest.confidence if strongest else "",
        description=strongest.description if strongest else "",
        rationale=decision.rationale,
        error=error,
    )


def write_report_csv(path: Path, rows: Iterable[ClipReportRow]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ClipReportRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_benchmark_json(path: Path, benchmark: Benchmark) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(benchmark), handle, indent=2)
        handle.write("\n")
