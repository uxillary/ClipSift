# ClipSift

ClipSift is a privacy-conscious CCTV review assistant built for the Google Cloud and NVIDIA GTC Berlin Golden Ticket competition. It scans a local folder of short CCTV clips, samples representative frames, and asks an open Gemma 3 vision model whether a person is likely visible.

This MVP is a local command-line prototype. It is an automated review aid, not proof of a person's presence, and it does not perform facial recognition or identity matching.

## How It Works

ClipSift uses OpenCV to stream each `.mp4`, `.avi`, `.mov` or `.mkv` file without loading the whole video into memory. By default it samples one frame per second, resizes each sampled frame while preserving aspect ratio, and sends the frame to a Gemma 3 vision-capable model through Hugging Face Transformers.

Google's Gemma model provides the visual assessment. NVIDIA GPUs accelerate inference locally when PyTorch detects CUDA. The planned cloud benchmark will compare CPU, a local RTX 3070 Ti, and a Google Cloud instance with an NVIDIA L4 GPU using the same controlled test set.

## Privacy And Safety

- Original videos are never moved, deleted or modified.
- Flagged clips are copied into the output folder.
- Evidence frames are generated only from sampled frames.
- Private footage, model caches, credentials and generated outputs are excluded from Git.
- Results may be wrong because of darkness, glare, rain, reflections, compression, partial obstruction or model error.
- Cloud testing should use only synthetic, public-domain or explicitly consented footage.

## Install On Windows With NVIDIA GPU

Use Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA acceleration, install a PyTorch build compatible with your NVIDIA driver and CUDA runtime. Check the current command from the official PyTorch install selector, then verify:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## CPU Fallback

If CUDA is unavailable, ClipSift reports that CPU inference is being used. CPU inference is expected to be much slower and may be impractical for larger Gemma checkpoints.

## Run ClipSift

```powershell
python -m clipsift.cli C:\path\to\clips --output "ClipSift Results"
```

Change the sampling rate:

```powershell
python -m clipsift.cli C:\path\to\clips --samples-per-second 0.5 --max-frame-size 768
```

Use a specific Hugging Face model:

```powershell
python -m clipsift.cli C:\path\to\clips --model-name google/gemma-3-4b-it
```

Output structure:

```text
ClipSift Results/
├── flagged_clips/
├── needs_review/
├── evidence_frames/
├── report.csv
└── benchmark.json
```

## Run Tests

```powershell
python -m pytest
```

The current tests cover model-response parsing, clip classification and report generation. They do not download or run a Gemma model.

## Benchmark Plan

No benchmark results are included until measured. The planned comparison is:

1. CPU baseline
2. Local NVIDIA RTX 3070 Ti
3. Google Cloud NVIDIA L4

Each run should record the device, model name, videos processed, video minutes processed, frames analysed, total time, average inference time and approximate frames per second.

## Current MVP Limitations

- Requires access to a compatible Gemma 3 vision checkpoint on Hugging Face.
- The exact model memory requirements must be validated on the target GPU.
- Video decoding and evidence saving are implemented, but need manual testing with safe sample footage.
- Classification thresholds are intentionally conservative and should be tuned against a labelled test set.
- No desktop GUI, EXE packaging, cloud deployment, live camera feed or facial recognition is included.

## TODO

- Google Cloud NVIDIA L4 deployment: document instance type, driver setup, model cache handling, test data handling and measured benchmark results.
- Future Windows GUI: add folder picker, progress view, result queue and evidence preview after the CLI workflow is validated.
