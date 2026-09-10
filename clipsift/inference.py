"""Gemma vision inference wrapper for ClipSift."""

from __future__ import annotations

import time
from importlib.metadata import version
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from clipsift.device import DeviceInfo, select_device


DEFAULT_MODEL_NAME = "google/gemma-3-4b-it"


PROMPT = """You are assisting with privacy-conscious CCTV triage.
Look only for whether a person is likely visible in this frame. Do not identify anyone.
Return only JSON with these keys:
{
  "person_status": "present" or "absent" or "uncertain",
  "assessment_confidence": "low" or "medium" or "high",
  "description": "brief neutral description"
}
Confidence means confidence in the complete visual assessment. Do not decide whether
human review is required; ClipSift applies that policy deterministically.
A clear scene with definitely no person can be absent with high confidence.
A small, obscured, or distant possible figure must be uncertain.
A clearly visible person must be present.
If frame quality prevents a reliable decision, use uncertain or low confidence.
"""


@dataclass(frozen=True)
class InferenceResult:
    raw_response: str
    inference_seconds: float
    peak_gpu_memory_bytes: int | None = None


class ModelLoadError(RuntimeError):
    """Raised when the selected model cannot be loaded."""


class CudaOutOfMemoryError(RuntimeError):
    """Raised with an actionable message after a CUDA allocation failure."""


class GemmaVisionModel:
    """Small adapter around Hugging Face Transformers vision-language inference."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "auto", preset: str = "safe") -> None:
        self.model_name = model_name
        self.device_info = select_device(device)
        self.preset = preset

        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        dtype = getattr(torch, self.device_info.torch_dtype)
        transformers_major = int(version("transformers").split(".", 1)[0])
        # Transformers 5 renamed the public loading argument; retain the older
        # spelling only for supported 4.x releases where it is still canonical.
        dtype_argument = "dtype" if transformers_major >= 5 else "torch_dtype"
        kwargs: dict[str, Any] = {dtype_argument: dtype}
        if self.device_info.device_type == "cuda":
            if preset == "safe":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif preset == "balanced":
                if (self.device_info.total_vram_bytes or 0) < 12 * 1024**3:
                    raise ModelLoadError("The balanced preset requires at least 12 GiB VRAM; use --preset safe on this GPU.")
            else:
                raise ValueError("Preset must be 'safe' or 'balanced'.")
            kwargs["device_map"] = {"": self.device_info.device}

        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
            if self.device_info.device_type == "cpu":
                self.model.to("cpu")
        except torch.cuda.OutOfMemoryError as exc:
            self.close()
            raise CudaOutOfMemoryError("CUDA ran out of memory while loading Gemma. Close GPU applications and retry with --preset safe; ClipSift did not fall back to CPU.") from exc
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"Could not load '{model_name}'. Confirm Gemma licence access, run 'hf auth login', and check the local cache/network. Details: {exc}") from exc

    def analyse_frame(self, frame_bgr: np.ndarray) -> InferenceResult:
        image = _frame_to_pil(frame_bgr)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device_info.device) for key, value in inputs.items()}

        try:
            if self.device_info.device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device_info.device_index)
                torch.cuda.synchronize(self.device_info.device_index)
            start = time.perf_counter()
            generated_ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
            if self.device_info.device_type == "cuda":
                torch.cuda.synchronize(self.device_info.device_index)
            elapsed = time.perf_counter() - start
        except torch.cuda.OutOfMemoryError as exc:
            raise CudaOutOfMemoryError("CUDA ran out of memory during inference. Close GPU applications or reduce image size, then retry with --preset safe; ClipSift did not fall back to CPU.") from exc

        input_token_count = inputs["input_ids"].shape[-1]
        output_ids = generated_ids[0][input_token_count:]
        raw_response = self.processor.decode(output_ids, skip_special_tokens=True).strip()
        peak = torch.cuda.max_memory_allocated(self.device_info.device_index) if self.device_info.device_type == "cuda" else None
        return InferenceResult(raw_response=raw_response, inference_seconds=elapsed, peak_gpu_memory_bytes=peak)

    def close(self) -> None:
        model = getattr(self, "model", None)
        processor = getattr(self, "processor", None)
        if model is not None:
            del self.model
        if processor is not None:
            del self.processor
        if getattr(self, "device_info", None) and self.device_info.device_type == "cuda":
            torch.cuda.empty_cache()


def _frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
