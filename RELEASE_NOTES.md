# ClipSift v0.1.0 Alpha

ClipSift's first public pre-release is a privacy-conscious Windows CCTV review assistant powered by the vision-capable `google/gemma-3-4b-it` model.

## Highlights

- **Desktop GUI:** Select input/output folders, device, sampling strategy and frame limit; follow scan progress; cancel safely; filter results; preview evidence; and open evidence or source video.
- **CLI:** Scan folders, inspect system readiness, and run controlled single-image or single-video tests using the same core workflow as the GUI.
- **Local folder scanning:** Read `.mp4`, `.avi`, `.mov`, and `.mkv` CCTV clips without modifying originals. Hybrid, uniform, and motion-prioritised sampling cover the readable timeline; motion selects candidate frames but is not treated as evidence of a person.
- **Clear classifications:** Deterministic application policy converts Gemma observations into **Person Detected**, **Needs Review**, or **No Person Detected**. Uncertain, malformed, and low-confidence assessments require review.
- **Evidence and reports:** Save the frame that triggered a person/review result, copy flagged/review clips, and produce `report.csv` and `benchmark.json` in the selected output folder.
- **System Check:** Inspect Python/runtime components, CUDA/GPU/VRAM, BitsAndBytes, Hugging Face authentication, and local model-cache signals without loading or downloading Gemma.

## Tested environments

- NVIDIA RTX 3060 Laptop GPU: controlled Gemma image inference was exercised with the 4-bit NF4 `safe` preset.
- Google Cloud NVIDIA T4 environment: the ClipSift workflow was exercised as part of pre-release development.

These are compatibility observations, not a broad hardware certification or an accuracy/performance benchmark. Real inference with the final packaged executable remains a release-checklist item.

## Packaging and model delivery

The Windows release is packaged as an unsigned, portable PyInstaller **onedir/windowed** application and distributed as a ZIP rather than an installer. The directory contains ClipSift and its Python/GUI/computer-vision/ML runtime, so it is large and must remain intact after extraction.

**Gemma is not bundled or redistributed.** Users must have a Hugging Face account, accept the terms for [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it), authenticate with `hf auth login`, and allow Hugging Face to download the model separately into its normal per-user cache. Credentials are not included.

## Known limitations

- This is Alpha software and may contain defects or change incompatibly.
- Automated classifications are review aids, not proof. Low light, obstruction, weather, reflections, motion blur and compression can cause errors.
- There is no facial recognition, identity/intent inference, live-camera support, real-time alerting, installer, or automatic update mechanism.
- Practical inference currently expects a compatible NVIDIA CUDA setup; CPU inference may be extremely slow and memory-heavy.
- The gated model requires internet access for initial download, sufficient disk space, accepted Gemma terms, and valid Hugging Face authentication.
- The Windows executable is not code-signed and may trigger Microsoft Defender SmartScreen. Verify the published SHA-256 and obtain downloads only from the documented release location.
- System Check is diagnostic and offline; passing it does not establish successful real inference.
