"""Command-line entry point for ClipSift."""
from __future__ import annotations

import argparse, json, shutil, sys, time, traceback
from itertools import islice
from pathlib import Path
import cv2
from PIL import Image, UnidentifiedImageError
from clipsift.classification import ClipStatus, assessment_to_dict, decide_clip, parse_model_response
from clipsift.device import DeviceSelectionError
from clipsift.doctor import collect_doctor_info, format_doctor_info
from clipsift.inference import CudaOutOfMemoryError, DEFAULT_MODEL_NAME, GemmaVisionModel, ModelLoadError
from clipsift.reporting import Benchmark, ClipReportRow, create_output_dirs, row_from_decision, write_benchmark_json, write_report_csv
from clipsift.video import find_videos, read_metadata, sample_frames, save_frame

def _device_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--preset", choices=("safe", "balanced"), default="safe")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipsift", description="Local automated CCTV review aid.")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="Scan a folder of videos")
    scan.add_argument("input_folder", type=Path)
    scan.add_argument("-o", "--output", type=Path, default=Path("clipsift_output"))
    scan.add_argument("--samples-per-second", type=float, default=1.0)
    scan.add_argument("--max-frame-size", type=int, default=896)
    scan.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    _device_options(scan)
    doctor = commands.add_parser("doctor", help="Report hardware and dependency diagnostics")
    doctor.add_argument("--device", default="auto")
    image = commands.add_parser("test-image", help="Test one local JPEG or PNG")
    image.add_argument("image_path", type=Path)
    image.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    image.add_argument("--debug", action="store_true", help="Show a traceback for unexpected errors")
    _device_options(image)
    video = commands.add_parser("test-video", help="Test up to a few sampled frames from one local video")
    video.add_argument("video_path", type=Path)
    video.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    video.add_argument("--max-frames", type=int, default=3)
    video.add_argument("--interval-seconds", type=float, default=2.0)
    video.add_argument("--debug", action="store_true", help="Show a traceback for unexpected errors")
    _device_options(video)
    return parser

def _normalise_args(argv: list[str]) -> list[str]:
    if argv and argv[0] not in {"scan", "doctor", "test-image", "test-video", "-h", "--help"}:
        return ["scan", *argv]
    return argv

def _validate_image(path: Path) -> str | None:
    if not path.is_file(): return f"Image does not exist or is not a file: {path}"
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: return "Unsupported image format. ClipSift accepts JPEG and PNG files."
    try:
        with Image.open(path) as image: image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        return f"Image is not a valid JPEG or PNG: {exc}"
    return None

def run_doctor(args: argparse.Namespace) -> int:
    print("ClipSift hardware diagnostic (advisory only; this does not run inference)")
    print(format_doctor_info(collect_doctor_info(args.device)))
    return 0

def run_test_image(args: argparse.Namespace) -> int:
    error = _validate_image(args.image_path)
    if error:
        print(error, file=sys.stderr); return 2
    model = None
    try:
        print(f"Loading model: {args.model_name} (preset={args.preset}, requested_device={args.device})")
        model = GemmaVisionModel(args.model_name, args.device, args.preset)
        frame = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
        if frame is None: raise ValueError(f"OpenCV could not decode image: {args.image_path}")
        result = model.analyse_frame(frame)
        assessment = parse_model_response(result.raw_response, 0.0)
        print("\nRaw model response:\n" + result.raw_response)
        print("\nValidated ClipSift result (automated review aid; review decision calculated by ClipSift):")
        print(json.dumps(assessment_to_dict(assessment), indent=2, ensure_ascii=False))
        print(f"\nActual device: {model.device_info.device} ({model.device_info.device_name})")
        print(f"Inference time: {result.inference_seconds:.3f} seconds")
        if result.peak_gpu_memory_bytes is not None:
            print(f"Peak allocated GPU memory: {result.peak_gpu_memory_bytes / 1024**3:.2f} GiB")
        return 0
    except (DeviceSelectionError, ModelLoadError, CudaOutOfMemoryError, ValueError) as exc:
        print(f"ClipSift inference failed: {exc}", file=sys.stderr); return 2
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"ClipSift inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if model: model.close()

def _validate_video(path: Path, max_frames: int, interval_seconds: float) -> str | None:
    if not path.is_file():
        return f"Video does not exist or is not a file: {path}"
    if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        return "Unsupported video format. ClipSift accepts MP4, AVI, MOV and MKV files."
    if max_frames <= 0:
        return "--max-frames must be greater than zero."
    if interval_seconds <= 0:
        return "--interval-seconds must be greater than zero."
    try:
        metadata = read_metadata(path)
        if metadata.frame_rate <= 0 or metadata.frame_count <= 0:
            return f"Video has no readable frames or frame rate: {path}"
    except (OSError, ValueError) as exc:
        return str(exc)
    return None

def run_test_video(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    error = _validate_video(args.video_path, args.max_frames, args.interval_seconds)
    if error:
        print(error, file=sys.stderr)
        return 2
    model = None
    try:
        print(f"Loading model once: {args.model_name} (preset={args.preset}, requested_device={args.device})")
        load_started = time.perf_counter()
        model = GemmaVisionModel(args.model_name, args.device, args.preset)
        load_seconds = time.perf_counter() - load_started
        assessments = []
        inference_seconds = 0.0
        peak_bytes = 0
        samples_per_second = 1.0 / args.interval_seconds
        for sampled in islice(sample_frames(args.video_path, samples_per_second, 896), args.max_frames):
            result = model.analyse_frame(sampled.image)
            assessment = parse_model_response(result.raw_response, sampled.timestamp_seconds)
            assessments.append(assessment)
            inference_seconds += result.inference_seconds
            peak_bytes = max(peak_bytes, result.peak_gpu_memory_bytes or 0)
            print(f"\nFrame at {sampled.timestamp_seconds:.3f}s:")
            print(json.dumps(assessment_to_dict(assessment), indent=2, ensure_ascii=False))
            if assessment.person_status == "present":
                print("Early stop: a person-present observation guarantees human review.")
                break
        decision = decide_clip(assessments)
        relevant = next((item for item in assessments if item.review_required), None)
        elapsed = time.perf_counter() - started
        print("\nFinal clip result (automated review aid):")
        print(f"Classification: {decision.status.value}")
        print(f"First relevant timestamp: {relevant.timestamp_seconds:.3f}s" if relevant else "First relevant timestamp: none")
        print(f"Frames analysed: {len(assessments)}")
        print(f"Model load time: {load_seconds:.3f} seconds")
        print(f"Total inference time: {inference_seconds:.3f} seconds")
        print(f"Total elapsed time: {elapsed:.3f} seconds")
        print(f"Actual device: {model.device_info.device} ({model.device_info.device_name})")
        print(f"Peak allocated GPU memory: {peak_bytes / 1024**3:.2f} GiB" if peak_bytes else "Peak allocated GPU memory: n/a")
        return 0
    except (DeviceSelectionError, ModelLoadError, CudaOutOfMemoryError, ValueError) as exc:
        print(f"ClipSift video inference failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"ClipSift video inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if model:
            model.close()

def run_scan(args: argparse.Namespace) -> int:
    paths = create_output_dirs(args.output); rows: list[ClipReportRow] = []
    videos_processed = frames_analysed = 0; seconds = 0.0; timings: list[float] = []
    started = time.perf_counter(); model = None; code = 0
    try:
        videos = find_videos(args.input_folder)
        if not videos: print(f"No supported videos found in {args.input_folder}", file=sys.stderr)
        print(f"Loading model: {args.model_name}")
        model = GemmaVisionModel(args.model_name, args.device, args.preset)
        print(f"Actual device: {model.device_info.device} ({model.device_info.device_name})")
        for number, video in enumerate(videos, 1):
            print(f"[{number}/{len(videos)}] Scanning {video.name}"); metadata = read_metadata(video)
            assessments, frames = [], {}
            try:
                for sampled in sample_frames(video, args.samples_per_second, args.max_frame_size):
                    result = model.analyse_frame(sampled.image); timings.append(result.inference_seconds)
                    item = parse_model_response(result.raw_response, sampled.timestamp_seconds)
                    assessments.append(item); frames[item.timestamp_seconds] = sampled.image; frames_analysed += 1
                decision = decide_clip(assessments); evidence = None
                if decision.strongest_frame:
                    evidence = paths["evidence"] / f"{video.stem}_{decision.strongest_frame.timestamp_seconds:.1f}s.jpg"
                    save_frame(evidence, frames[decision.strongest_frame.timestamp_seconds])
                if decision.status == ClipStatus.PERSON_DETECTED: shutil.copy2(video, paths["flagged"] / video.name)
                elif decision.status == ClipStatus.NEEDS_REVIEW: shutil.copy2(video, paths["needs_review"] / video.name)
                rows.append(row_from_decision(video, metadata.duration_seconds, decision, evidence))
                videos_processed += 1; seconds += metadata.duration_seconds; print(f"  => {decision.status.value}")
            except CudaOutOfMemoryError: raise
            except Exception as exc:
                print(f"  Error scanning {video.name}: {exc}", file=sys.stderr)
                rows.append(row_from_decision(video, metadata.duration_seconds, decide_clip([]), None, str(exc)))
    except KeyboardInterrupt:
        print("Cancellation requested; writing partial report.", file=sys.stderr); code = 130
    except Exception as exc:
        print(f"ClipSift failed: {exc}", file=sys.stderr); code = 2
    finally:
        elapsed = time.perf_counter() - started; write_report_csv(args.output / "report.csv", rows)
        average = sum(timings) / len(timings) if timings else 0.0
        write_benchmark_json(args.output / "benchmark.json", Benchmark(
            device_name=model.device_info.device_name if model else "unknown", device_type=model.device_info.device if model else "unknown",
            model_name=args.model_name, videos_processed=videos_processed, video_minutes_processed=round(seconds / 60, 3),
            frames_analysed=frames_analysed, total_processing_time_seconds=round(elapsed, 3), average_inference_time_seconds=round(average, 4),
            approximate_frames_per_second=round(frames_analysed / elapsed, 4) if elapsed else 0.0))
        if model: model.close()
    return code

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_normalise_args(list(sys.argv[1:] if argv is None else argv)))
    return {"doctor": run_doctor, "test-image": run_test_image, "test-video": run_test_video, "scan": run_scan}[args.command](args)

if __name__ == "__main__": raise SystemExit(main())
