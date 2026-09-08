"""Command-line entry point for ClipSift."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from clipsift.classification import ClipStatus, decide_clip, parse_model_response
from clipsift.inference import DEFAULT_MODEL_NAME, GemmaVisionModel
from clipsift.reporting import (
    Benchmark,
    ClipReportRow,
    create_output_dirs,
    row_from_decision,
    write_benchmark_json,
    write_report_csv,
)
from clipsift.video import find_videos, read_metadata, sample_frames, save_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipsift",
        description="ClipSift: local automated CCTV review aid for likely person visibility.",
    )
    parser.add_argument("input_folder", type=Path, help="Folder containing .mp4, .avi, .mov or .mkv files.")
    parser.add_argument("-o", "--output", type=Path, default=Path("clipsift_output"), help="Output directory.")
    parser.add_argument("--samples-per-second", type=float, default=1.0, help="Frame sampling rate. Default: 1.")
    parser.add_argument("--max-frame-size", type=int, default=896, help="Resize longest frame edge to this size.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face Gemma vision model name.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_paths = create_output_dirs(args.output)
    report_rows: list[ClipReportRow] = []
    videos_processed = 0
    video_seconds_processed = 0.0
    frames_analysed = 0
    inference_times: list[float] = []
    started = time.perf_counter()

    try:
        videos = find_videos(args.input_folder)
        if not videos:
            print(f"No supported videos found in {args.input_folder}", file=sys.stderr)

        print(f"Loading model: {args.model_name}")
        model = GemmaVisionModel(args.model_name)
        if model.device_info.device_type == "cuda":
            print(f"Using CUDA device: {model.device_info.device_name}")
        else:
            print("CUDA was not detected. CPU inference is being used and may be slow.")

        for index, video_path in enumerate(videos, start=1):
            print(f"[{index}/{len(videos)}] Scanning {video_path.name}")
            metadata = read_metadata(video_path)
            assessments = []
            frame_lookup = {}

            try:
                for sampled in sample_frames(video_path, args.samples_per_second, args.max_frame_size):
                    result = model.analyse_frame(sampled.image)
                    inference_times.append(result.inference_seconds)
                    assessment = parse_model_response(result.raw_response, sampled.timestamp_seconds)
                    assessments.append(assessment)
                    frame_lookup[sampled.timestamp_seconds] = sampled.image
                    frames_analysed += 1
                    print(
                        f"  frame {sampled.timestamp_seconds:.1f}s: "
                        f"person_visible={assessment.person_visible} confidence={assessment.confidence}"
                    )

                decision = decide_clip(assessments)
                evidence_path = None
                if decision.strongest_frame:
                    evidence_path = output_paths["evidence"] / f"{video_path.stem}_{decision.strongest_frame.timestamp_seconds:.1f}s.jpg"
                    save_frame(evidence_path, frame_lookup[decision.strongest_frame.timestamp_seconds])

                if decision.status == ClipStatus.PERSON_DETECTED:
                    shutil.copy2(video_path, output_paths["flagged"] / video_path.name)
                elif decision.status == ClipStatus.NEEDS_REVIEW:
                    shutil.copy2(video_path, output_paths["needs_review"] / video_path.name)

                report_rows.append(row_from_decision(video_path, metadata.duration_seconds, decision, evidence_path))
                videos_processed += 1
                video_seconds_processed += metadata.duration_seconds
                print(f"  => {decision.status.value}")
            except Exception as exc:
                print(f"  Error scanning {video_path.name}: {exc}", file=sys.stderr)
                decision = decide_clip([])
                report_rows.append(row_from_decision(video_path, metadata.duration_seconds, decision, None, str(exc)))

    except KeyboardInterrupt:
        print("\nCancellation requested. Writing partial report...", file=sys.stderr)
    except Exception as exc:
        print(f"ClipSift failed: {exc}", file=sys.stderr)
        return 2
    finally:
        elapsed = time.perf_counter() - started
        report_path = args.output / "report.csv"
        benchmark_path = args.output / "benchmark.json"
        write_report_csv(report_path, report_rows)
        average_inference = sum(inference_times) / len(inference_times) if inference_times else 0.0
        benchmark = Benchmark(
            device_name=locals().get("model").device_info.device_name if "model" in locals() else "unknown",
            device_type=locals().get("model").device_info.device_type if "model" in locals() else "unknown",
            model_name=args.model_name,
            videos_processed=videos_processed,
            video_minutes_processed=round(video_seconds_processed / 60.0, 3),
            frames_analysed=frames_analysed,
            total_processing_time_seconds=round(elapsed, 3),
            average_inference_time_seconds=round(average_inference, 4),
            approximate_frames_per_second=round(frames_analysed / elapsed, 4) if elapsed > 0 else 0.0,
        )
        write_benchmark_json(benchmark_path, benchmark)
        print(f"Report written to {report_path}")
        print(f"Benchmark written to {benchmark_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
