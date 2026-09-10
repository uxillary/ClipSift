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

This command samples at most three frames approximately two seconds apart by default. Gemma is loaded once and reused for every sampled frame. The source is read only and is never moved or uploaded.

```powershell
python -m clipsift.cli test-video local_test_data/person-short.mp4 --device auto --preset safe --max-frames 3
```

Use `--interval-seconds`, `--max-frames`, and `--debug` to adjust the controlled test. ClipSift stops early when a `present` observation has already guaranteed human review.

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

## Tests

Tests are offline and mock model/CUDA boundaries; they do not download Gemma or read private media.

```powershell
python -m pytest
pytest
python -m clipsift.cli --help
python -m clipsift.cli doctor
```

No general performance or accuracy benchmark is claimed until it has been measured on a larger controlled test set.
