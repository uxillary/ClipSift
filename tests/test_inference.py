from types import SimpleNamespace

import numpy as np

from clipsift.device import DeviceInfo
from clipsift.inference import GemmaVisionModel


class _Tensor:
    shape = (1, 3)

    def to(self, device):
        return self


class _Processor:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        return {"input_ids": _Tensor()}

    def decode(self, output, skip_special_tokens):
        return '{"person_visible": false, "confidence": "high", "description": "Empty", "review_required": false}'


class _Model:
    def generate(self, **kwargs):
        return [[10, 11, 12, 13]]


def test_cuda_analyse_frame_has_runtime_torch_import(monkeypatch) -> None:
    """Exercise the post-load CUDA path without loading or downloading a model."""
    import clipsift.inference as inference

    monkeypatch.setattr(inference, "_frame_to_pil", lambda frame: object())
    monkeypatch.setattr(inference.torch.cuda, "reset_peak_memory_stats", lambda index: None)
    monkeypatch.setattr(inference.torch.cuda, "synchronize", lambda index: None)
    monkeypatch.setattr(inference.torch.cuda, "max_memory_allocated", lambda index: 1234)

    wrapper = GemmaVisionModel.__new__(GemmaVisionModel)
    wrapper.device_info = DeviceInfo("cuda", "cuda:0", "cuda", 0, "Test GPU", "float16", 6 * 1024**3)
    wrapper.processor = _Processor()
    wrapper.model = _Model()

    result = wrapper.analyse_frame(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.peak_gpu_memory_bytes == 1234
    assert result.raw_response.startswith("{")
