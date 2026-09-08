import csv
import json
from pathlib import Path

from clipsift.classification import decide_clip, parse_model_response
from clipsift.reporting import Benchmark, row_from_decision, write_benchmark_json, write_report_csv


def test_write_report_csv(tmp_path: Path) -> None:
    assessment = parse_model_response(
        '{"person_visible": true, "confidence": "high", "description": "Person visible.", "review_required": true}',
        timestamp_seconds=4.0,
    )
    decision = decide_clip([assessment])
    row = row_from_decision(Path("clip.mp4"), 12.5, decision, Path("evidence.jpg"))
    report_path = tmp_path / "report.csv"

    write_report_csv(report_path, [row])

    with report_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["status"] == "Person Detected"
    assert rows[0]["evidence_frame"] == "evidence.jpg"
    assert rows[0]["description"] == "Person visible."


def test_write_benchmark_json(tmp_path: Path) -> None:
    benchmark = Benchmark(
        device_name="NVIDIA test GPU",
        device_type="cuda",
        model_name="google/gemma-3-4b-it",
        videos_processed=2,
        video_minutes_processed=1.25,
        frames_analysed=10,
        total_processing_time_seconds=5.0,
        average_inference_time_seconds=0.4,
        approximate_frames_per_second=2.0,
    )
    path = tmp_path / "benchmark.json"

    write_benchmark_json(path, benchmark)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["device_type"] == "cuda"
    assert data["videos_processed"] == 2
    assert "Automated review aid" in data["note"]
