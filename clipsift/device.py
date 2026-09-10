"""Runtime device selection shared by ClipSift commands."""

from __future__ import annotations

import re
from dataclasses import dataclass


class DeviceSelectionError(RuntimeError):
    """Raised when a requested compute device cannot be used."""


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    device: str
    device_type: str
    device_index: int | None
    device_name: str
    torch_dtype: str
    total_vram_bytes: int | None


def select_device(requested: str = "auto") -> DeviceInfo:
    import torch

    choice = requested.strip().lower()
    if choice not in {"auto", "cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", choice):
        raise DeviceSelectionError("Device must be one of: auto, cpu, cuda, cuda:0, cuda:1, ...")

    if choice == "cpu":
        return DeviceInfo(choice, "cpu", "cpu", None, "CPU", "float32", None)

    if choice == "auto" and not torch.cuda.is_available():
        return DeviceInfo(choice, "cpu", "cpu", None, "CPU", "float32", None)

    if not torch.cuda.is_available():
        raise DeviceSelectionError(f"CUDA device '{choice}' was requested, but PyTorch reports CUDA is unavailable.")

    index = 0 if choice in {"auto", "cuda"} else int(choice.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise DeviceSelectionError(f"CUDA device index {index} is invalid; PyTorch reports {count} CUDA device(s).")

    properties = torch.cuda.get_device_properties(index)
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return DeviceInfo(
        choice,
        f"cuda:{index}",
        "cuda",
        index,
        torch.cuda.get_device_name(index),
        dtype,
        int(properties.total_memory),
    )
