"""Hardware and dependency diagnostics."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys

from clipsift.device import DeviceSelectionError, select_device
from clipsift.inference import DEFAULT_MODEL_NAME


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def collect_doctor_info(device: str = "auto") -> dict[str, str]:
    required = ("torch", "transformers", "accelerate", "bitsandbytes", "PIL", "cv2")
    failures = []
    for module in required:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}")

    import torch

    try:
        selected = select_device(device)
        selected_index = str(selected.device_index) if selected.device_index is not None else "n/a"
        selected_name = selected.device_name
        vram = f"{selected.total_vram_bytes / 1024**3:.2f} GiB" if selected.total_vram_bytes else "n/a"
        recommendation = "safe" if selected.device_type == "cuda" and (selected.total_vram_bytes or 0) < 12 * 1024**3 else "balanced"
        if selected.device_type == "cpu":
            recommendation = "safe (CPU inference may be very slow)"
    except DeviceSelectionError as exc:
        selected_index, selected_name, vram, recommendation = "error", str(exc), "n/a", "unavailable"

    try:
        from huggingface_hub import get_token
        authenticated = "available" if get_token() else "not found"
    except Exception:
        authenticated = "unavailable (huggingface_hub import failed)"

    return {
        "Python version": platform.python_version(),
        "Operating system": platform.platform(),
        "PyTorch version": torch.__version__,
        "Transformers version": _version("transformers"),
        "Accelerate version": _version("accelerate"),
        "BitsAndBytes": _version("bitsandbytes"),
        "CUDA available": str(torch.cuda.is_available()),
        "PyTorch CUDA version": str(torch.version.cuda or "n/a"),
        "CUDA device count": str(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "Selected GPU index": selected_index,
        "Selected device name": selected_name,
        "Total VRAM": vram,
        "Selected ClipSift model": DEFAULT_MODEL_NAME,
        "Hugging Face authentication": authenticated,
        "Required imports": "working" if not failures else "failed (" + ", ".join(failures) + ")",
        "Recommended preset": recommendation + " (advisory; inference has not been tested by doctor)",
    }


def format_doctor_info(info: dict[str, str]) -> str:
    width = max(map(len, info))
    return "\n".join(f"{key:<{width}} : {value}" for key, value in info.items())
