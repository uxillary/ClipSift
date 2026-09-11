"""Collect only the BitsAndBytes backends matching ClipSift's build runtime."""

from pathlib import Path

import torch
from PyInstaller.utils.hooks import collect_dynamic_libs


cuda_tag = (torch.version.cuda or "").replace(".", "")
binaries = []
for source, destination in collect_dynamic_libs("bitsandbytes"):
    name = Path(source).name.lower()
    if "_cpu." in name or (cuda_tag and f"_cuda{cuda_tag}." in name):
        binaries.append((source, destination))

# BitsAndBytes may JIT optional kernels and therefore needs importable source.
module_collection_mode = "pyz+py"
