# ClipSift v0.1.0-alpha

ClipSift's first source release candidate provides a Windows desktop interface and CLI for locally reviewing folders of CCTV clips with `google/gemma-3-4b-it`.

## Requirements

- Windows 10 or 11, 64-bit, with Python 3.11.
- An NVIDIA GPU is recommended; CPU inference may be very slow and memory-heavy.
- A Hugging Face account, accepted Gemma model terms, and `hf auth login` are required before an uncached first run.
- Gemma is downloaded and cached separately in the user's normal Hugging Face cache. Approximately 20 GB of free disk space is recommended for the environment, model cache, and results.

## Included

- Source CLI and refined ttkbootstrap desktop GUI.
- Hybrid, uniform, and motion-prioritised frame sampling.
- Deterministic Person Detected, Needs Review, and No Person Detected policy.
- Trigger-matched evidence, CSV and benchmark reports, cancellation, preferences, and offline setup diagnostics.
- Tested Windows PyInstaller build configuration; packaged artifacts are not part of this source candidate.

## Safety and limitations

Video is processed locally and original footage is never modified. ClipSift is an automated review aid, not proof that a person is or is not present; users must review flagged footage and account for low light, obstruction, weather, reflections, and compression artifacts.

Any locally produced alpha executable is unsigned and may trigger Microsoft Defender SmartScreen. Review the source and checksum before running it. This candidate is not an installer and has not been published as a GitHub release.
