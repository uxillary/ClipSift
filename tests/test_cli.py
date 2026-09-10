from pathlib import Path
from PIL import Image
from clipsift import cli
from clipsift.inference import CudaOutOfMemoryError, ModelLoadError

def test_doctor_formatting() -> None:
    text = cli.format_doctor_info({"Python version": "3.11", "CUDA available": "True"})
    assert "Python version" in text and "CUDA available" in text

def test_missing_image_does_not_load_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "GemmaVisionModel", lambda *args: (_ for _ in ()).throw(AssertionError("loaded")))
    assert cli.main(["test-image", str(tmp_path / "missing.jpg")]) == 2

def test_unsupported_image_format(tmp_path: Path) -> None:
    path = tmp_path / "image.gif"; path.write_bytes(b"GIF89a")
    assert cli.main(["test-image", str(path)]) == 2

def _image(tmp_path: Path) -> Path:
    path = tmp_path / "person.jpg"; Image.new("RGB", (2, 2)).save(path); return path

def test_clear_model_access_error(monkeypatch, tmp_path: Path, capsys) -> None:
    def fail(*args): raise ModelLoadError("Confirm Gemma licence access and run 'hf auth login'.")
    monkeypatch.setattr(cli, "GemmaVisionModel", fail)
    assert cli.main(["test-image", str(_image(tmp_path))]) == 2
    assert "Gemma licence" in capsys.readouterr().err

def test_clear_cuda_oom_error(monkeypatch, tmp_path: Path, capsys) -> None:
    def fail(*args): raise CudaOutOfMemoryError("CUDA ran out of memory; did not fall back to CPU.")
    monkeypatch.setattr(cli, "GemmaVisionModel", fail)
    assert cli.main(["test-image", str(_image(tmp_path))]) == 2
    assert "did not fall back to CPU" in capsys.readouterr().err
