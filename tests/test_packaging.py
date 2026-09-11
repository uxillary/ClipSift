import json
from pathlib import Path

import clipsift.packaged_smoke as packaged_smoke
from clipsift.readiness import DiagnosticReport
from clipsift.resources import resource_path


def test_resource_path_supports_explicit_bundle_root(tmp_path: Path) -> None:
    assert resource_path("assets/clipsift.ico", tmp_path) == tmp_path / "assets" / "clipsift.ico"


def test_packaged_smoke_imports_modules_without_loading_model(tmp_path: Path, monkeypatch) -> None:
    imported = []
    monkeypatch.setattr(packaged_smoke.importlib, "import_module", lambda name: imported.append(name))
    monkeypatch.setattr(
        packaged_smoke,
        "run_system_check",
        lambda device: DiagnosticReport("Ready", (), True, True),
    )
    output = tmp_path / "smoke.json"
    packaged_smoke.run_packaged_smoke_check(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert imported == list(packaged_smoke.REQUIRED_MODULES)
    assert payload["model_loaded"] is False
    assert payload["diagnostic_status"] == "Ready"
