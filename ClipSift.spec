# -*- mode: python ; coding: utf-8 -*-
"""Maintained PyInstaller onedir/windowed definition for ClipSift."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


root = Path(SPEC).resolve().parent
icon = root / "assets" / "clipsift.ico"
version_file = root / "build" / "ClipSift" / "version_info.txt"

datas = [(str(path), "assets") for path in (root / "assets").glob("*.png")]
datas.append((str(icon), "assets"))
datas += collect_data_files("ttkbootstrap")
datas += collect_data_files("transformers", includes=["**/*.json"])
for distribution in ("accelerate", "bitsandbytes", "opencv-python", "pillow", "torch", "transformers", "ttkbootstrap"):
    datas += copy_metadata(distribution)

binaries = []
for package in ("cv2", "torch"):
    binaries += collect_dynamic_libs(package)

hiddenimports = [
    "accelerate",
    "accelerate.big_modeling",
    "accelerate.hooks",
    "accelerate.utils",
]
for package in (
    "bitsandbytes",
    "transformers.generation",
    "transformers.models.gemma3",
):
    hiddenimports += collect_submodules(package)

a = Analysis(
    [str(root / "clipsift" / "gui_launcher.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(root / "packaging" / "pyinstaller-hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "jupyter", "matplotlib", "pandas", "pytest", "scipy", "tensorflow"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClipSift",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon),
    version=str(version_file),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=["*.dll", "*.pyd"],
    name="ClipSift",
)
