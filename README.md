# Clip Sift

ClipSift is a privacy-conscious local CCTV review assistant. It asks Gemma 3 whether an image is likely to contain a person and flags results for human review. It is not proof, does not identify people, and never uploads media.

## Windows setup

Use Python 3.11 and install the CUDA-enabled PyTorch build separately so it matches your NVIDIA driver. Get the current command from the [official PyTorch selector](https://pytorch.org/get-started/locally/); do not rely on the generic dependency install to replace a working CUDA build.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Run the CUDA command supplied by pytorch.org here.
python -m pip install -r requirements.txt
python -m pip install -e .
```

Accept the `google/gemma-3-4b-it` licence on Hugging Face, then authenticate locally. ClipSift checks only whether authentication is available and never displays the token.

```powershell
hf auth login
python -m clipsift.cli doctor
```

The first inference may download several gigabytes of model files. Hugging Face model access, network access, free disk space, and compatible Transformers/BitsAndBytes versions are required.

## Single-image smoke test

JPEG and PNG are supported. The image stays on the computer.

```powershell
python -m clipsift.cli test-image local_test_data/person.jpg --device auto --preset safe
# Equivalent installed command:
clipsift test-image local_test_data/person.jpg --device auto --preset safe
```

Gemma supplies only a visual observation (`present`, `absent`, or `uncertain`), confidence in that complete assessment, and a neutral description. ClipSift—not the model—calculates the review decision: present, uncertain, parse failures, and all low-confidence assessments require review; absent observations with medium or high confidence do not. A model-generated legacy `review_required` value is ignored.

Initial controlled image test on an RTX 3060 Laptop GPU using `google/gemma-3-4b-it` with the safe 4-bit NF4 preset:

- Person image: correctly observed a person near a fence; 8.439 seconds; 3.15 GiB peak allocated GPU memory.
- Empty image: correctly observed no person and described the car, fence and houses; 9.806 seconds; 3.15 GiB peak allocated GPU memory.

These two observations are a smoke-test milestone, not a general accuracy benchmark.

## Single-video smoke test

Current CCTV source clips are typically about two minutes long. The video smoke test selects at most 12 frames by default across the complete readable duration. Gemma is loaded once and reused for every selected frame. The source is read only and is never moved or uploaded.

```powershell
python -m clipsift.cli test-video local_test_data/person-short.mp4 --device auto --preset safe --sampling-strategy hybrid --max-frames 12
```

The default `--sampling-strategy hybrid` mixes full-timeline coverage with high-motion candidates found by a lightweight grayscale OpenCV pass. Motion only prioritises frames; it is never treated as evidence that a person is present. Use `uniform` for timeline-only selection or `motion` for spaced motion peaks. `--motion-interval-seconds`, `--max-frames`, and `--debug` adjust the controlled test. ClipSift stops early only after a confident `present` observation.

Inspect selection without loading Gemma:

```powershell
python -m clipsift.cli test-video local_test_data/person-short.mp4 --sampling-strategy hybrid --max-frames 12 --dry-run
```

Gemma makes the visual observation; deterministic ClipSift application policy makes the final review decision.

`auto` selects CUDA device 0 when PyTorch reports CUDA, otherwise CPU. Use `--device cuda`, `--device cuda:1`, or `--device cpu` to make an explicit choice. Device indices are validated before model loading; CUDA errors never cause a silent CPU fallback. This makes the same command portable between desktop and laptop GPUs without hard-coded GPU names.

The default `safe` CUDA preset loads Gemma directly in 4-bit NF4 with double quantisation and BF16 compute when supported (otherwise FP16). It is intended for the 6GB laptop GPU. `balanced` uses unquantised BF16/FP16 and refuses CUDA devices with less than 12 GiB VRAM. CPU uses FP32 and is likely to be very slow and memory-heavy.

Common failures are explicit: accept/request checkpoint access and run `hf auth login` for access errors; update compatible public packages for import/API errors; close GPU applications and use `safe` for CUDA-memory errors. `doctor` is diagnostic only and its preset recommendation does not claim inference was tested.

## Video scan

The original invocation remains accepted; `scan` is the explicit form:

```powershell
clipsift scan C:\path\to\clips --output "ClipSift Results" --device auto --preset safe
python -m clipsift.cli C:\path\to\clips --output "ClipSift Results"
```

Original recordings are not modified. Generated review folders, evidence, private media, credentials, model caches, and weights are excluded from Git.

## Windows desktop GUI — Phase 4A

The local ttkbootstrap interface wraps the same scan service used by the CLI. It keeps model loading and video analysis on a worker thread while all Tk updates are delivered to the main thread through a queue. Gemma is loaded once per scan and reused across videos; cancellation is cooperative between frames and files, and completed reports remain available.

Install the dependencies and editable entry points, then launch:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m clipsift.gui
# Or, after editable installation:
clipsift-gui
```

Choose input/output folders, Auto/GPU/CPU, Hybrid/Uniform/Motion sampling, and the maximum number of frames. The results table and activity log are automated-review aids; Gemma supplies observations while ClipSift's deterministic policy supplies classifications. Original videos are never modified or deleted.

Selecting a result shows its evidence image without stretching it and provides direct **Open Evidence** and **Open Video** actions. Person Detected evidence comes from the first person-present frame; Needs Review evidence comes from the exact uncertain, malformed, or low-confidence frame that triggered review. No Person Detected results do not require evidence. Compact filters show person, review, clear, and error results without changing stored results, while the activity log is collapsed by default.

GUI preferences are stored per user at `%LOCALAPPDATA%\ClipSift\settings.json`, outside the repository. ClipSift remembers the last valid input/output folders, device, sampling strategy, maximum frames, and window size. Missing or corrupt settings safely fall back to defaults.

Use **System Check** before a first scan to review Python and PyTorch readiness, CUDA/GPU and VRAM, BitsAndBytes, Hugging Face authentication, the selected Gemma checkpoint, and whether model files appear in the local Hugging Face cache. This check is offline: it does not load or download Gemma. A lightweight worker-thread preflight repeats the essential checks before scanning and verifies that the input contains supported videos and the output folder can be created and written.

Gemma requires a Hugging Face account, accepted terms on the [`google/gemma-3-4b-it` model page](https://huggingface.co/google/gemma-3-4b-it), and local authentication when the model is not already accessible from cache:

```powershell
hf auth login
```

ClipSift never asks for, displays, or stores the token. Model files downloaded by Hugging Face remain in its per-user cache; opening the GUI and running System Check do not initiate a download. The safe GPU preset requires CUDA and BitsAndBytes. If GPU is explicitly selected but CUDA is unavailable, the scan is blocked rather than silently falling back to CPU. If setup is reported unavailable, run `python -m clipsift.cli doctor`, confirm the CUDA-enabled PyTorch installation, accept the model terms, authenticate, and retry System Check.

**Privacy:** Video is processed locally and original footage is never modified. ClipSift writes only generated evidence, reports, and copies of flagged/review clips to the chosen output folder.

### Build the local Windows package

Phase 4A uses PyInstaller **onedir** and **windowed** mode. Onedir is used because the PyTorch/CUDA and BitsAndBytes native stack is large and is more predictable to inspect and troubleshoot without onefile extraction. The resulting unsigned build may trigger a Windows SmartScreen warning. It is a local verification artifact, not a published release or installer.

Activate the repository's `.venv`, install the build extra, and run the guarded PowerShell script:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[build,test]"
.\scripts\build-windows.ps1
```

The script runs the complete tests, builds from `ClipSift.spec`, performs non-interactive packaged module/offline-diagnostic checks, creates `dist\ClipSift-0.1.0-win64.zip`, and writes its adjacent `.sha256` file. It cleans only ClipSift's explicit packaging targets.

```text
dist\
├── ClipSift\
│   ├── ClipSift.exe
│   └── _internal\       # Python, UI, computer-vision and ML runtime
├── ClipSift-0.1.0-win64.zip
├── ClipSift-0.1.0-win64.zip.sha256
└── packaged-smoke-check.json
```

Expect several gigabytes because CUDA-enabled PyTorch, Transformers, OpenCV, torchvision, and BitsAndBytes native components are included. Gemma weights and credentials are deliberately not included: the packaged app uses the signed-in user's normal Hugging Face authentication and cache. Accept the Gemma terms and run `hf auth login` before first uncached use.

The verified Phase 4A build environment used Python 3.11.5, PyInstaller 6.16.0, PyInstaller hooks 2026.7, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 5.17.0, Accelerate 1.15.0, BitsAndBytes 0.50.2, OpenCV 5.0.0.93, Pillow 12.3.0, and ttkbootstrap 2.2.2. Rebuilds can vary when the runtime dependencies permitted by `pyproject.toml` resolve to newer versions.

Complete the manual checks in `scripts\smoke-check-windows.md`, including visible launch/no-console behavior, icon appearance, folder pickers, settings persistence, links, and—separately—real RTX inference. Automated diagnostics do not prove packaged inference.

Deleting `dist\ClipSift` removes only the packaged application folder. It does not delete Gemma files in the user's Hugging Face cache or preferences at `%LOCALAPPDATA%\ClipSift\settings.json`.

## Tests

Tests are offline and mock model/CUDA boundaries; they do not download Gemma or read private media.

```powershell
python -m pytest
pytest
python -m clipsift.cli --help
python -m clipsift.cli doctor
```

No general performance or accuracy benchmark is claimed until it has been measured on a larger controlled test set.
