"""Offline system diagnostics and scan preflight checks for the GUI."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from clipsift.doctor import collect_doctor_info
from clipsift.inference import DEFAULT_MODEL_NAME
from clipsift.video import SUPPORTED_EXTENSIONS


MODEL_URL = "https://huggingface.co/google/gemma-3-4b-it"
GITHUB_URL = "https://github.com/uxillary/ClipSift"
README_URL = "https://github.com/uxillary/ClipSift/blob/main/README.md"
ALLOWED_EXTERNAL_URLS = frozenset({MODEL_URL, GITHUB_URL, README_URL})


@dataclass(frozen=True)
class DiagnosticItem:
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class DiagnosticReport:
    status: str
    items: tuple[DiagnosticItem, ...]
    authenticated: bool
    model_cached: bool


@dataclass(frozen=True)
class PreflightResult:
    can_start: bool
    message: str
    report: DiagnosticReport


def huggingface_cache_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_files_cached(model_name: str = DEFAULT_MODEL_NAME, cache_root: Path | None = None) -> bool:
    """Inspect Hugging Face's local cache layout without network or model loading."""
    root = cache_root or huggingface_cache_root()
    repository = root / f"models--{model_name.replace('/', '--')}" / "snapshots"
    try:
        return any(
            (snapshot / "config.json").exists()
            and (any(snapshot.rglob("*.safetensors")) or any(snapshot.rglob("*.bin")))
            for snapshot in repository.iterdir()
            if snapshot.is_dir()
        )
    except OSError:
        return False


def map_diagnostic_report(info: dict[str, str], model_cached: bool) -> DiagnosticReport:
    imports_ready = info.get("Required imports") == "working"
    python_version = info.get("Python version", "unknown")
    try:
        python_ready = tuple(map(int, python_version.split(".")[:2])) >= (3, 11)
    except ValueError:
        python_ready = False
    cuda = info.get("CUDA available") == "True"
    selected_name = info.get("Selected device name", "unavailable")
    selected_failed = info.get("Selected GPU index") == "error"
    bitsandbytes = info.get("BitsAndBytes", "not installed")
    bnb_ready = bitsandbytes != "not installed"
    pytorch = info.get("PyTorch version", "not installed")
    pytorch_ready = pytorch not in {"", "not installed", "unavailable"}
    authenticated = info.get("Hugging Face authentication") == "available"

    items = (
        DiagnosticItem("Python/runtime", "Ready" if python_ready and imports_ready else "Unavailable", f"Python {python_version}; imports {info.get('Required imports', 'unknown')}"),
        DiagnosticItem("PyTorch", "Ready" if pytorch_ready else "Unavailable", pytorch),
        DiagnosticItem("CUDA", "Unavailable" if selected_failed else ("Ready" if cuda else "Attention required"), "Available" if cuda else "Not available; CPU can be selected explicitly"),
        DiagnosticItem("Selected device", "Unavailable" if selected_failed else "Ready", selected_name),
        DiagnosticItem("Available VRAM", "Ready" if cuda and info.get("Total VRAM") != "n/a" else "Attention required", info.get("Total VRAM", "n/a")),
        DiagnosticItem("BitsAndBytes", "Ready" if bnb_ready else ("Attention required" if selected_name == "CPU" else "Unavailable"), bitsandbytes),
        DiagnosticItem("Hugging Face authentication", "Ready" if authenticated else "Attention required", "Available; token is not displayed" if authenticated else "Not found; accept Gemma terms and run hf auth login"),
        DiagnosticItem("Gemma model", "Ready", info.get("Selected ClipSift model", DEFAULT_MODEL_NAME)),
        DiagnosticItem("Local model cache", "Ready" if model_cached else "Attention required", "Model files appear cached locally" if model_cached else "Model files were not found in the local Hugging Face cache"),
    )
    statuses = {item.status for item in items}
    overall = "Unavailable" if "Unavailable" in statuses else "Attention required" if "Attention required" in statuses else "Ready"
    return DiagnosticReport(overall, items, authenticated, model_cached)


def run_system_check(device: str = "auto", collector: Callable[[str], dict[str, str]] = collect_doctor_info, cache_checker: Callable[[str], bool] = model_files_cached) -> DiagnosticReport:
    return map_diagnostic_report(collector(device), cache_checker(DEFAULT_MODEL_NAME))


def output_folder_writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="clipsift-write-check-", dir=path, delete=True):
            pass
        return True, ""
    except OSError as exc:
        return False, f"The output folder cannot be created or written: {exc}"


def run_preflight(
    input_folder: Path,
    output_folder: Path,
    device: str,
    checker: Callable[[str], DiagnosticReport] = run_system_check,
) -> PreflightResult:
    report = checker(device)
    if not input_folder.is_dir():
        return PreflightResult(False, "Choose an existing input folder.", report)
    try:
        has_videos = any(path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS for path in input_folder.iterdir())
    except OSError as exc:
        return PreflightResult(False, f"The input folder cannot be read: {exc}", report)
    if not has_videos:
        return PreflightResult(False, "The input folder contains no supported videos.", report)
    writable, error = output_folder_writable(output_folder)
    if not writable:
        return PreflightResult(False, error, report)
    if report.status == "Unavailable":
        details = next((item.detail for item in report.items if item.status == "Unavailable"), "Required runtime component unavailable")
        return PreflightResult(False, details, report)
    if not report.authenticated and not report.model_cached:
        return PreflightResult(False, "Gemma is not cached and Hugging Face authentication was not found. Accept the model terms, then run 'hf auth login'.", report)
    return PreflightResult(True, "Setup is ready for inference.", report)


def open_external_url(url: str, opener: Callable[[str], object]) -> tuple[bool, str]:
    if url not in ALLOWED_EXTERNAL_URLS:
        return False, "URL is not an approved ClipSift link."
    try:
        opener(url)
        return True, ""
    except OSError as exc:
        return False, f"Could not open the link: {exc}"
