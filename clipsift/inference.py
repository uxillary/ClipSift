"""Gemma vision inference wrapper for ClipSift."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_MODEL_NAME = "google/gemma-3-4b-it"


PROMPT = """You are assisting with privacy-conscious CCTV triage.
Look only for whether a person is likely visible in this frame. Do not identify anyone.
Return only JSON with these keys:
{
  "person_visible": true or false,
  "confidence": "low" or "medium" or "high",
  "description": "brief neutral description",
  "review_required": true or false
}
"""


@dataclass(frozen=True)
class DeviceInfo:
    device_type: str
    device_name: str
    torch_dtype: str


@dataclass(frozen=True)
class InferenceResult:
    raw_response: str
    inference_seconds: float


def detect_device() -> DeviceInfo:
    import torch

    if torch.cuda.is_available():
        return DeviceInfo(
            device_type="cuda",
            device_name=torch.cuda.get_device_name(0),
            torch_dtype="bfloat16",
        )
    return DeviceInfo(device_type="cpu", device_name="CPU", torch_dtype="float32")


class GemmaVisionModel:
    """Small adapter around Hugging Face Transformers vision-language inference."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.device_info = detect_device()

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if self.device_info.device_type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device_info.device_type == "cuda" else None,
        )
        if self.device_info.device_type == "cpu":
            self.model.to("cpu")

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
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        start = time.perf_counter()
        generated_ids = self.model.generate(**inputs, max_new_tokens=180, do_sample=False)
        elapsed = time.perf_counter() - start

        input_token_count = inputs["input_ids"].shape[-1]
        output_ids = generated_ids[0][input_token_count:]
        raw_response = self.processor.decode(output_ids, skip_special_tokens=True).strip()
        return InferenceResult(raw_response=raw_response, inference_seconds=elapsed)


def _frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
