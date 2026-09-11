# ClipSift

> [!WARNING]
> **ClipSift v0.1.0 Alpha is pre-release software.** Expect rough edges, incomplete testing, and breaking changes. It is an automated review aid—not proof that a person is or is not present. Always review important footage yourself.

ClipSift scans a local folder of CCTV video, samples representative frames, and uses the vision-capable `google/gemma-3-4b-it` model to help sort clips into **Person Detected**, **Needs Review**, and **No Person Detected**. The Windows GUI provides a review queue, evidence images, filters, progress and setup diagnostics; the CLI supports the same scanning workflow and controlled image/video tests.

## Requirements

- 64-bit Windows 10 or Windows 11.
- An NVIDIA CUDA-capable GPU is strongly recommended. The `safe` preset is intended for lower-memory GPUs (including tested 6 GB hardware); unquantised `balanced` mode requires at least 12 GiB VRAM.
- CPU mode is available but may be extremely slow and memory-heavy.
- Approximately 20 GB of free disk space for the application or Python environment, the separately cached model, and results.
- Internet access for initial setup/model download, a Hugging Face account, and access to the gated Gemma checkpoint.
- Source installation additionally requires Python 3.11 and a CUDA-enabled PyTorch build compatible with the installed NVIDIA driver.

## Download

- **Windows portable ZIP:** `[DOWNLOAD_URL_TO_BE_ADDED]`
- **Release page:** `[GITHUB_RELEASE_URL_TO_BE_ADDED]`
- **SHA-256:** `[SHA256_TO_BE_ADDED]`

These placeholders will be replaced after the final package is rebuilt, tested, uploaded, and hashed. The portable build is an unsigned PyInstaller `onedir` application, not an installer; Windows SmartScreen may display a warning.

## Install and run

### Windows portable package

1. Download the ZIP and compare its SHA-256 with the value published above.
2. Extract the **entire** ZIP to a local folder. Do not run the executable from inside the ZIP or move `ClipSift.exe` away from its `_internal` folder.
3. Complete the [Gemma access and authentication](#gemma-access-and-authentication) steps.
4. Open `ClipSift\ClipSift.exe`.
5. Run **System Check**, select input and output folders, then start a scan.

The first real scan can take time while the model is downloaded and cached. System Check itself is offline and does not download or load Gemma.

### Install from source

Use Python 3.11. Create a virtual environment, then install the CUDA-enabled PyTorch/torchvision command supplied by the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing ClipSift's dependencies.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Run the CUDA-enabled torch/torchvision install command from pytorch.org here.
python -m pip install -r requirements.txt
python -m pip install -e .
hf auth login
python -m clipsift.cli doctor
python -m clipsift.gui
```

To scan from the CLI:

```powershell
clipsift scan C:\path\to\clips --output "ClipSift Results" --device auto --preset safe
```

Supported video extensions are `.mp4`, `.avi`, `.mov`, and `.mkv`. Use `clipsift --help` for all CLI options.

## Gemma access and authentication

ClipSift uses the gated [`google/gemma-3-4b-it` checkpoint](https://huggingface.co/google/gemma-3-4b-it). Before an uncached first scan:

1. Create or sign in to a Hugging Face account.
2. Visit the model page and accept Google's Gemma terms/request access.
3. Install the Hugging Face CLI if needed and authenticate on the same Windows account that will run ClipSift:

   ```powershell
   hf auth login
   ```

ClipSift does not include or redistribute Gemma weights. Hugging Face downloads Gemma separately on first use and stores it in the signed-in user's normal model cache under its own terms. The release also contains no Hugging Face credentials. ClipSift checks whether authentication/cache access appears available but never displays or stores the token itself.

## Privacy and outputs

Video inference is performed locally on the user's computer. ClipSift does not upload footage, use facial recognition, or attempt to identify people. It reads but never moves, modifies, or deletes original recordings. It writes reports, evidence frames, and copies of flagged/review clips only to the selected output folder. Gemma authentication and model download involve Hugging Face, but CCTV media is not sent with those requests.

## Known Alpha limitations

- Results can be wrong because of low light, obstruction, weather, reflections, camera angle, motion blur, or compression; flagged and important footage needs human review.
- This release classifies person visibility only. It does not identify people, infer intent, monitor live feeds, or provide real-time alerts.
- The Windows build is large, portable rather than installed, and unsigned; SmartScreen may warn on launch.
- A compatible NVIDIA/CUDA/PyTorch/BitsAndBytes stack and separately downloaded gated model are required for practical GPU inference.
- CPU inference is likely to be impractically slow on many systems.
- System Check verifies setup signals only; it does not prove that real packaged inference will succeed.
- Performance observations to date are limited and are not a general accuracy benchmark.

See [release notes](RELEASE_NOTES.md) for this pre-release and [third-party notices](THIRD_PARTY_NOTICES.md) for bundled-software licensing guidance.
