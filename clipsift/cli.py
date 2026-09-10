"""Command-line entry point for ClipSift."""
from __future__ import annotations

import argparse, json, sys, time, traceback
from itertools import islice
from pathlib import Path
import cv2
from PIL import Image, UnidentifiedImageError
from clipsift.classification import ClipStatus, assessment_to_dict, decide_clip, parse_model_response
from clipsift.device import DeviceSelectionError
from clipsift.doctor import collect_doctor_info, format_doctor_info
from clipsift.inference import CudaOutOfMemoryError, DEFAULT_MODEL_NAME, GemmaVisionModel, ModelLoadError
from clipsift.scanner import ScanConfig, ScanEvent, scan_folder
from clipsift.video import (
    frames_at_timestamps, read_metadata, select_video_timestamps,
)

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
    video.add_argument("--max-frames", type=int, default=12)
    video.add_argument("--sampling-strategy", choices=("uniform", "motion", "hybrid"), default="hybrid")
    video.add_argument("--motion-interval-seconds", "--interval-seconds", dest="motion_interval_seconds", type=float, default=0.5)
    video.add_argument("--dry-run", action="store_true", help="Display selected timestamps without loading Gemma")
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
    error = _validate_video(args.video_path, args.max_frames, args.motion_interval_seconds)
    if error:
        print(error, file=sys.stderr)
        return 2
    model = None
    try:
        metadata = read_metadata(args.video_path)
        selections = select_video_timestamps(
            args.video_path, metadata, args.sampling_strategy, args.max_frames, args.motion_interval_seconds
        )
        if not selections:
            raise ValueError("Sampling did not produce any readable timestamps.")
        print(f"Video duration: {metadata.duration_seconds:.3f} seconds")
        print(f"FPS: {metadata.frame_rate:.3f}")
        print(f"Total source frames: {metadata.frame_count}")
        print(f"Sampling strategy: {args.sampling_strategy}")
        print(f"Timestamps selected: {len(selections)}")
        for selection in selections:
            origins = "+".join(selection.sources)
            motion = f", motion_score={selection.motion_score:.3f}" if selection.motion_score is not None else ""
            print(f"  {selection.timestamp_seconds:.3f}s [{origins}{motion}]")
        if args.dry_run:
            print("Dry run complete; Gemma was not loaded.")
            return 0
        print(f"Loading model once: {args.model_name} (preset={args.preset}, requested_device={args.device})")
        load_started = time.perf_counter()
        model = GemmaVisionModel(args.model_name, args.device, args.preset)
        load_seconds = time.perf_counter() - load_started
        assessments = []
        inference_seconds = 0.0
        peak_bytes = 0
        for sampled in islice(frames_at_timestamps(args.video_path, selections, 896), args.max_frames):
            result = model.analyse_frame(sampled.image)
            assessment = parse_model_response(result.raw_response, sampled.timestamp_seconds)
            assessments.append(assessment)
            inference_seconds += result.inference_seconds
            peak_bytes = max(peak_bytes, result.peak_gpu_memory_bytes or 0)
            print(f"\nFrame at {sampled.timestamp_seconds:.3f}s:")
            print(json.dumps(assessment_to_dict(assessment), indent=2, ensure_ascii=False))
            if assessment.person_status == "present" and assessment.assessment_confidence in {"medium", "high"}:
                print("Early stop: a confident person-present observation guarantees human review.")
                break
        decision = decide_clip(assessments)
        relevant = next((item for item in assessments if item.review_required), None)
        first_person = next((item for item in assessments if item.person_status == "present"), None)
        elapsed = time.perf_counter() - started
        print("\nFinal clip result (automated review aid):")
        print(f"Classification: {decision.status.value}")
        print(f"First relevant timestamp: {relevant.timestamp_seconds:.3f}s" if relevant else "First relevant timestamp: none")
        print(f"First person timestamp: {first_person.timestamp_seconds:.3f}s" if first_person else "First person timestamp: none")
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
    def print_event(event: ScanEvent) -> None:
        if event.kind == "video_started":
            print(f"[{event.current}/{event.total}] Scanning {event.filename}")
        elif event.kind == "video_completed":
            print(f"  => {event.message}")
        elif event.kind == "video_error":
            print(f"  Error scanning {event.filename}: {event.message}", file=sys.stderr)
        elif event.kind in {"operation", "model_loaded"}:
            print(event.message)

    try:
        scan_folder(ScanConfig(
            input_folder=args.input_folder,
            output_folder=args.output,
            device=args.device,
            preset=args.preset,
            sampling_strategy=None,
            samples_per_second=args.samples_per_second,
            max_frame_size=args.max_frame_size,
            model_name=args.model_name,
        ), on_event=print_event)
        print(f"Report written to {args.output / 'report.csv'}")
        print(f"Benchmark written to {args.output / 'benchmark.json'}")
        return 0
    except KeyboardInterrupt:
        print("Cancellation requested.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ClipSift failed: {exc}", file=sys.stderr)
        return 2

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_normalise_args(list(sys.argv[1:] if argv is None else argv)))
    return {"doctor": run_doctor, "test-image": run_test_image, "test-video": run_test_video, "scan": run_scan}[args.command](args)

if __name__ == "__main__": raise SystemExit(main())
