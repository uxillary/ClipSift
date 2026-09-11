# Third-party notices

ClipSift source code is licensed under the MIT License in `LICENSE`. It depends on third-party software distributed under its own licences; those licences are not replaced by ClipSift's licence.

Primary runtime dependencies include:

| Component | Licence |
| --- | --- |
| PyTorch and torchvision | BSD 3-Clause |
| Transformers, Accelerate, Hugging Face Hub, Tokenizers and Safetensors | Apache License 2.0 |
| BitsAndBytes | MIT |
| OpenCV Python | Apache License 2.0 |
| Pillow | HPND/MIT-CMU-style licence |
| ttkbootstrap | MIT, with bundled assets under their stated Apache-2.0 or BSD terms |
| NumPy | BSD 3-Clause |

PyInstaller is a build-time tool licensed under GPL-2.0-or-later with its bootloader exception. It is not a ClipSift runtime dependency.

This summary is informational. Source releases install dependencies separately through Python package indexes, where their complete licence texts and metadata are provided. Anyone redistributing a packaged build must retain the complete licence and notice files supplied by every bundled direct and transitive dependency and review their current terms before publication.

Gemma model weights are not included in ClipSift source or packaged distributions. Users obtain `google/gemma-3-4b-it` separately from Hugging Face under Google's applicable Gemma terms; this notice does not grant or replace those terms.
