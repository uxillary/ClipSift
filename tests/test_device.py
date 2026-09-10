from types import SimpleNamespace
import pytest
from clipsift.device import DeviceSelectionError, select_device

def _mock_cuda(monkeypatch, *, available: bool, count: int = 0) -> None:
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Test GPU {index}")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=6 * 1024**3))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

def test_auto_selects_cuda(monkeypatch) -> None:
    _mock_cuda(monkeypatch, available=True, count=1)
    assert select_device("auto").device == "cuda:0"

def test_auto_selects_cpu_without_cuda(monkeypatch) -> None:
    _mock_cuda(monkeypatch, available=False)
    assert select_device("auto").device == "cpu"

def test_forced_cpu_ignores_cuda(monkeypatch) -> None:
    _mock_cuda(monkeypatch, available=True, count=1)
    assert select_device("cpu").device == "cpu"

def test_unavailable_cuda_fails(monkeypatch) -> None:
    _mock_cuda(monkeypatch, available=False)
    with pytest.raises(DeviceSelectionError, match="unavailable"):
        select_device("cuda")

def test_invalid_cuda_index_fails(monkeypatch) -> None:
    _mock_cuda(monkeypatch, available=True, count=1)
    with pytest.raises(DeviceSelectionError, match="invalid"):
        select_device("cuda:1")
