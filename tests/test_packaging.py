import json
from pathlib import Path

import clipsift.packaged_smoke as packaged_smoke
from clipsift.readiness import DiagnosticReport
from clipsift.resources import resource_path, set_windows_app_user_model_id


def test_resource_path_supports_explicit_bundle_root(tmp_path: Path) -> None:
    assert resource_path("assets/clipsift.ico", tmp_path) == tmp_path / "assets" / "clipsift.ico"


def test_windows_app_id_is_noop_off_windows(monkeypatch) -> None:
    monkeypatch.setattr("clipsift.resources.sys.platform", "linux")
    assert set_windows_app_user_model_id("ClipSift.test") is False


def test_packaged_smoke_imports_modules_without_loading_model(tmp_path: Path, monkeypatch) -> None:
    imported = []
    monkeypatch.setattr(packaged_smoke.importlib, "import_module", lambda name: imported.append(name))
    import clipsift.readiness
    monkeypatch.setattr(clipsift.readiness, "run_system_check", lambda device: DiagnosticReport("Ready", (), True, True))
    output = tmp_path / "smoke.json"
    assert packaged_smoke.run_packaged_smoke_check(output) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert imported == list(packaged_smoke.REQUIRED_MODULES)
    assert payload["model_loaded"] is False
    assert payload["diagnostic_status"] == "Ready"
    assert payload["success"] is True


def test_packaged_smoke_writes_structured_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        packaged_smoke.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(f"{Path.home()}\\secret module")) if name == "torch" else None,
    )
    output = tmp_path / "failure.json"
    assert packaged_smoke.run_packaged_smoke_check(output) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["failure"]["type"] == "RuntimeError"
    assert str(Path.home()) not in payload["failure"]["message"]
    assert str(Path.home()) not in payload["imports"]["torch"]
