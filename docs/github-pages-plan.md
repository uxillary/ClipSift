# ClipSift GitHub Pages plan

> Planning document only. Do not build or publish the website as part of the v0.1.0 Alpha documentation work.

## Purpose and audience

The site should explain ClipSift quickly, set honest expectations, and guide visitors to the release or source repository. Its primary audience is Windows CCTV owners and technical evaluators who want a local, GPU-assisted first review pass while retaining human control. Secondary audiences include open-model developers and reviewers interested in the Gemma, NVIDIA GPU, and Google Cloud project workflow.

## Proposed page and draft messaging

1. **Hero** — “Turn a folder of CCTV clips into a manageable review queue.” Follow with: “ClipSift uses Gemma locally to flag clips likely to contain a person. Alpha software; always verify important footage.”
2. **How it works** — “Choose a folder → sample frames → assess with Gemma → review classifications and evidence.” Explain that motion prioritises frames but never counts as person evidence.
3. **What you receive** — Show **Person Detected**, **Needs Review**, and **No Person Detected**, evidence previews, copied review clips, CSV reporting, and benchmark metadata.
4. **Desktop and CLI** — Present the Windows GUI as the default journey and the CLI for repeatable/technical use.
5. **Setup** — Summarise Windows, GPU, disk, Hugging Face, and Gemma prerequisites before presenting the download.
6. **Privacy and safety** — “Your CCTV footage is processed locally and originals are never changed.” Clearly separate local footage processing from Hugging Face authentication/model download.
7. **Demo and technical notes** — Embed/link the safe-footage demo and link to full documentation/source.
8. **Alpha notice** — Repeat that classifications can be wrong, the binary is unsigned, and this is not a security decision system.

## Primary calls to action

- **Download ClipSift v0.1.0 Alpha** — `[DOWNLOAD_URL_TO_BE_ADDED]`
- **View the pre-release** — `[GITHUB_RELEASE_URL_TO_BE_ADDED]`
- **Watch the demo** — `[YOUTUBE_DEMO_URL_TO_BE_ADDED]`
- **Read setup documentation** — `[DOCUMENTATION_URL_TO_BE_ADDED]`

The download CTA must sit beside the version, file type, size, unsigned-build warning, and verified SHA-256—not ahead of prerequisites or safety language.

## Screenshots and assets required

- Branded GUI hero image with safe, non-private sample content.
- Input/output selection and scan-controls screenshot.
- System Check screenshot with tokens, usernames, paths, and machine identifiers removed.
- In-progress scan screenshot.
- Results table showing all three classifications.
- Evidence preview and generated report/output-folder example.
- Short architecture/workflow diagram.
- ClipSift icon, wordmark/header, social preview image, and favicon variants.
- Captioned YouTube demo thumbnail.

All captures need consistent Alpha branding, accessible alt text, legible scaling, and a privacy review before publication.

## Download and model setup explanation

The page should state that the download is a large portable ZIP, not an installer. Users extract the complete folder, retain `_internal` beside `ClipSift.exe`, verify the SHA-256, expect a possible SmartScreen warning, and run System Check before scanning.

Gemma must be described as a **separate gated download**, not part of ClipSift. Users need a Hugging Face account, accepted `google/gemma-3-4b-it` terms, `hf auth login`, internet access for the first download, and enough space for the per-user cache. Explain that ClipSift does not ship a token or model weights and does not upload CCTV footage to Hugging Face.

## Required reference sections

### Privacy

Explain local video processing, untouched originals, generated output, separate model-download traffic, no facial recognition, and no identity or intent inference.

### Requirements

List 64-bit Windows 10/11, recommended compatible NVIDIA CUDA GPU, lower-memory `safe` preset, 12 GiB minimum for `balanced`, approximately 20 GB free disk space, initial internet access, and Hugging Face/Gemma access.

### Limitations

Cover Alpha stability, possible classification errors, environmental/image-quality failure modes, slow CPU operation, unsigned packaging, lack of installer/updater/live feeds, and the distinction between System Check and real inference.

### FAQ

Answer: whether footage leaves the computer; why Gemma is not included; where the model is stored; why authentication is required; whether CPU works; whether ClipSift identifies people; what each classification means; and whether originals are changed.

### Troubleshooting

Provide symptom-led help for SmartScreen, incomplete ZIP extraction, failed Hugging Face access, missing cached model, CUDA/PyTorch mismatch, insufficient VRAM/disk, no supported videos, unwritable output folders, and where to find reports. Link to `[DOCUMENTATION_URL_TO_BE_ADDED]` for maintained detail.

## Publication prerequisites

Do not implement the site until the final package name, size, checksum, external download location, GitHub pre-release URL, demo URL, documentation URL, approved screenshots, and tested setup steps are available.
