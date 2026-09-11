from pathlib import Path

from clipsift.readiness import (
    ALLOWED_EXTERNAL_URLS,
    GITHUB_URL,
    MODEL_URL,
    README_URL,
    DiagnosticReport,
    map_diagnostic_report,
    model_files_cached,
    open_external_url,
    run_preflight,
    run_system_check,
)


def _doctor(*, auth: bool = True, cuda: bool = True, failed: bool = False) -> dict[str, str]:
    return {
        "Python version": "3.11.5",
        "Required imports": "working",
        "PyTorch version": "2.7.1+cu118",
        "CUDA available": str(cuda),
        "Selected GPU index": "error" if failed else ("0" if cuda else "n/a"),
        "Selected device name": "CUDA unavailable" if failed else ("Test GPU" if cuda else "CPU"),
        "Total VRAM": "6.00 GiB" if cuda else "n/a",
        "BitsAndBytes": "0.50.2",
        "Hugging Face authentication": "available" if auth else "not found",
        "Selected ClipSift model": "google/gemma-3-4b-it",
    }


def test_diagnostic_mapping_ready_attention_and_unavailable() -> None:
    assert map_diagnostic_report(_doctor(), True).status == "Ready"
    assert map_diagnostic_report(_doctor(auth=False), False).status == "Attention required"
    assert map_diagnostic_report(_doctor(failed=True), True).status == "Unavailable"
    missing_torch = _doctor(); missing_torch["PyTorch version"] = "not installed"
    assert map_diagnostic_report(missing_torch, True).status == "Unavailable"


def test_hugging_face_auth_and_model_cache_reporting(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--google--gemma-3-4b-it" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001.safetensors").write_bytes(b"cached")
    assert model_files_cached(cache_root=tmp_path) is True
    report = map_diagnostic_report(_doctor(auth=False), True)
    assert report.authenticated is False
    assert report.model_cached is True
    assert next(item for item in report.items if item.label == "Local model cache").status == "Ready"


def test_system_check_never_loads_or_downloads_model() -> None:
    calls = []
    report = run_system_check(
        "auto",
        collector=lambda device: (calls.append(("doctor", device)) or _doctor()),
        cache_checker=lambda model: (calls.append(("cache", model)) or True),
    )
    assert report.status == "Ready"
    assert calls == [("doctor", "auto"), ("cache", "google/gemma-3-4b-it")]


def test_preflight_blocks_missing_auth_and_cache_but_allows_ready(tmp_path: Path) -> None:
    input_folder = tmp_path / "input videos"; input_folder.mkdir()
    (input_folder / "clip.mp4").write_bytes(b"video")
    attention = map_diagnostic_report(_doctor(auth=False), False)
    blocked = run_preflight(input_folder, tmp_path / "output", "auto", checker=lambda device: attention)
    assert blocked.can_start is False
    assert "hf auth login" in blocked.message
    ready = map_diagnostic_report(_doctor(), True)
    allowed = run_preflight(input_folder, tmp_path / "output ready", "auto", checker=lambda device: ready)
    assert allowed.can_start is True


def test_external_urls_are_fixed_https_constants() -> None:
    assert ALLOWED_EXTERNAL_URLS == {MODEL_URL, GITHUB_URL, README_URL}
    assert all(url.startswith("https://") for url in ALLOWED_EXTERNAL_URLS)
    opened = []
    assert open_external_url(MODEL_URL, opened.append) == (True, "")
    assert open_external_url("https://example.invalid", opened.append)[0] is False
    assert opened == [MODEL_URL]
