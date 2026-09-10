# ClipSift — Project Context

## Project status

- **Status:** In progress / early MVP
- **Created for:** Google Cloud × NVIDIA GTC Berlin 2026 Golden Ticket challenge
- **Submission deadline:** 10 September 2026
- **Owner:** Adam Johnston
- **Primary development platform:** Windows 11
- **Primary desktop GPU:** NVIDIA GeForce RTX 3070 Ti
- **Additional development hardware:** Windows laptops with differing NVIDIA GPUs and VRAM capacities

## One-sentence summary

ClipSift is a privacy-conscious CCTV review assistant that uses AI and GPU inference to find video clips likely to contain a person, reducing hours of footage to a short review queue.

## The problem

Home CCTV systems can generate many short clips, but manually watching every recording is slow and repetitive. Most clips may contain no useful activity.

The current CCTV camera typically creates clips approximately two minutes long. Frame selection must therefore cover the complete readable duration rather than only the beginning.

ClipSift is intended to perform the first review pass. It samples frames from a folder of recordings, flags likely human activity, and creates a clear report showing which clips should be checked manually.

It is a **triage and review tool**, not an autonomous security decision system.

## Core user workflow

1. The user selects a folder containing CCTV clips.
2. ClipSift reads each supported video without changing the original.
3. Representative frames are sampled at configurable intervals.
4. An open visual AI model assesses whether a person is likely visible.
5. Results from multiple frames are combined to reduce one-frame false positives.
6. Each clip is classified as:
   - **Person Detected**
   - **Needs Review**
   - **No Person Detected**
7. The app produces a review queue, evidence frames, timestamps and a CSV report.

## Competition relevance

The project is being developed alongside the Google Cloud and NVIDIA learning pathways, particularly **Intro to Inference: How to Run AI Models on a GPU**.

The competition requires entrants to build and document something new using Google Cloud and NVIDIA technology with an open model such as Gemma, Nemotron or Cosmos. Entries are judged equally on:

1. Technical innovation
2. Effective use of NVIDIA and Google Cloud technology
3. Potential impact or usefulness
4. Quality of documentation and presentation

ClipSift addresses these areas through a practical end-user problem, open-model visual inference, local and cloud GPU execution, measurable benchmarking, and public documentation.

## Technology direction

### Core application

- Python 3.11
- OpenCV for reading videos and extracting frames
- Pillow where image handling or thumbnails require it
- A simple command-line MVP first
- A Windows desktop GUI in a later stage
- PyInstaller for a later portable Windows release

### Open model

- A vision-capable **Gemma 3** model is the intended competition model.
- Gemma must perform a meaningful visual-analysis task rather than being mentioned only for branding.
- Model responses should be requested in a small structured format and validated before use.
- Gemma provides visual observations only. ClipSift deterministically derives whether human review is required and ignores any model-generated review decision.

Example internal response:

```json
{
  "person_status": "present",
  "assessment_confidence": "high",
  "description": "One person walking beside a parked vehicle"
}
```

The current preferred schema uses `person_status` (`present`, `absent`, or `uncertain`), `assessment_confidence`, and `description`. Present, uncertain, malformed, and low-confidence results require review; absent medium/high-confidence results do not.

### Verified local inference milestone

An initial controlled two-image smoke test completed on an RTX 3060 Laptop GPU using `google/gemma-3-4b-it` with BitsAndBytes 4-bit NF4:

- Person image: person near a fence observed, 8.439 seconds, 3.15 GiB peak allocated GPU memory.
- Empty image: no person observed, with car, fence and houses described, 9.806 seconds, 3.15 GiB peak allocated GPU memory.

These are initial controlled smoke-test results, not a general accuracy benchmark. The single-video smoke-test command loads Gemma once and reuses it across up to 12 selected frames by default. Hybrid selection combines full-timeline coverage with spaced high-motion candidates from a lightweight OpenCV pass. Motion only prioritises frames and is never considered person detection; Gemma makes the visual observation and ClipSift code makes the review decision.

The final implementation must use an exact Gemma checkpoint that genuinely supports image input and fits the available environment. The README must record the exact model name, model licence and runtime configuration actually tested. Do not invent these details in advance.

### NVIDIA

- Run local inference using Adam's primary desktop RTX 3070 Ti where practical.
- Detect devices at runtime across desktop and laptop development computers; no GPU model is hard-coded.
- Use the memory-conscious `safe` preset for lower-VRAM hardware, including 4-bit CUDA inference.
- Provide a CPU fallback with an honest warning that it may be slower.
- Record the actual GPU name and runtime details in benchmark output.
- A later optimisation experiment may use NVIDIA TensorRT, but this must not be presented as implemented until tested.

### Google Cloud

- Run the same or equivalent inference workload on a Google Cloud instance with an NVIDIA GPU.
- The intended comparison target is an NVIDIA L4 GPU, subject to quota and availability.
- Use only staged, synthetic, public-domain or explicitly consented test footage for cloud testing.
- If Cloud Storage is used, uploaded test data should be temporary and its handling documented.
- Record the exact Google Cloud service and GPU configuration actually used.

## Intended competition benchmark

Compare the same labelled test set across applicable environments:

1. CPU baseline
2. Local NVIDIA RTX 3070 Ti
3. Google Cloud NVIDIA GPU

Record only real measurements:

- Number of clips processed
- Total duration of source footage
- Number of frames analysed
- Total processing time
- Average inference time
- Approximate processed frames per second
- Correctly flagged clips
- False positives
- False negatives
- Reduction in footage requiring manual review

Never add estimated or fabricated benchmark results to the app, website, README or competition entry.

## MVP scope

The first working version should:

- Accept a folder of `.mp4`, `.avi`, `.mov` and `.mkv` files.
- Sample frames without loading whole videos into memory.
- Make the sampling interval configurable.
- Use the selected Gemma visual model to assess sampled frames.
- Combine multiple assessments into a clip-level result.
- Copy flagged clips into output folders while preserving originals.
- Save the strongest relevant evidence frame when available.
- Produce `report.csv` and `benchmark.json`.
- Display progress and handle cancellation safely.
- Use CUDA automatically when available.
- Fail clearly and safely when a video, model response or dependency is invalid.

Suggested output structure:

```text
ClipSift Results/
├── Person Detected/
├── Needs Review/
├── Evidence Frames/
├── report.csv
└── benchmark.json
```

## Privacy, safety and accuracy boundaries

ClipSift must:

- Never implement facial recognition.
- Never attempt to identify a person.
- Never infer identity, intent, criminality or personal characteristics.
- Never describe an automated result as proof.
- Never delete, move or modify original recordings.
- Process footage locally by default where possible.
- Require deliberate action before any cloud upload.
- Clearly explain that low light, reflections, weather, obstruction and compression may cause mistakes.
- Present every result as an aid for human review.

Private CCTV recordings, extracted evidence frames, model caches, credentials and local result folders must be excluded from Git.

## Explicit non-goals for the competition MVP

Do not add these before the basic workflow is working and documented:

- Facial recognition
- Live camera feeds
- Continuous surveillance
- Automatic police or security alerts
- Behaviour or threat prediction
- Number-plate recognition
- Cloud accounts or user authentication
- Mobile applications
- Remote footage storage
- Automatic deletion of empty clips
- A complex video editor
- A large polished desktop interface
- Multiple object-detection categories

## Recommended development stages

### Stage 1 — Local proof of concept

- Read one test clip.
- Extract representative frames.
- Load the selected Gemma vision model.
- Confirm whether a deliberately staged person is visible.
- Save the raw and parsed model result.

### Stage 2 — Batch scanner

- Scan a folder of clips.
- Aggregate frame results.
- Create output folders, evidence images and reports.
- Add safe cancellation and errors.

### Stage 3 — Benchmark

- Create a small labelled test set.
- Measure CPU and local RTX performance.
- Repeat the controlled test on a Google Cloud NVIDIA GPU.
- Document accuracy and limitations.

### Stage 4 — Presentation and competition entry

- Publish a clean GitHub repository.
- Add architecture, setup, usage, privacy and benchmark documentation.
- Record a short demonstration using safe footage.
- Publish the required LinkedIn or X post.
- Tag the required Google Cloud/NVIDIA accounts and include `#NVIDIAGTC`.
- Submit the public post through the official form before the deadline.

### Stage 5 — Post-competition product work

- Build a simple Windows GUI.
- Add Fast, Balanced and Thorough scan presets.
- Package and test a portable Windows release.
- Investigate an optional fast first-pass detector and TensorRT optimisation.

## Proposed repository structure

```text
clipsift/
├── clipsift/
│   ├── __init__.py
│   ├── cli.py
│   ├── video.py
│   ├── inference.py
│   ├── classification.py
│   └── reporting.py
├── tests/
├── sample_data/
├── docs/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

Only safe, intentionally created sample media may be committed under `sample_data/`.

## Portfolio presentation

### Project card

- **Eyebrow:** `PROJECT / IN PROGRESS`
- **Title:** `ClipSift`
- **Status:** `BUILDING`
- **Description:** `A privacy-focused CCTV review tool that uses AI and GPU inference to find clips likely to contain a person—reducing hours of footage to a short, manageable review queue.`

### Process line

```text
VIDEO CLIPS → FRAME SAMPLING → GEMMA VISION → REVIEW QUEUE
```

### Technical details

| Field | Value |
| --- | --- |
| Model | Gemma 3 |
| Local GPU | RTX 3070 Ti |
| Cloud | Google Cloud |
| Status | Building |

### Feature tags

```text
PERSON DETECTION
LOCAL INFERENCE
PRIVACY FIRST
GPU BENCHMARK
```

### Supporting copy

- **Callout:** `HOURS OF FOOTAGE → MINUTES TO REVIEW`
- **Footer:** `Built alongside Google Cloud and NVIDIA's GPU inference learning pathways.`
- **Challenge label:** `GTC BERLIN 2026 GOLDEN TICKET CHALLENGE · SEP 2026`

## Language rules

Use phrases such as:

- “likely to contain a person”
- “flags clips for human review”
- “automated review aid”
- “privacy-conscious”
- “measured on a controlled test set”

Avoid unsupported phrases such as:

- “perfect detection”
- “crime detection”
- “identifies intruders”
- “guaranteed accuracy”
- “real-time surveillance”
- “fully private” when cloud processing is enabled

## Definition of a successful first release

The first release is successful when a new user can select or provide a folder of safe test clips, run the documented command, receive sensible classifications and evidence, understand the limitations, and reproduce the published benchmark without risking the original footage.

## Instructions for future Codex work

When this file is supplied as project context:

1. Treat it as the current product and scope reference.
2. Preserve privacy and safety boundaries.
3. Do not claim planned technology has already been implemented.
4. Do not fabricate model compatibility, benchmark results or cloud deployment details.
5. Prefer completing and validating the smallest useful stage before expanding scope.
6. Update this file when a major technical choice or verified result changes.
