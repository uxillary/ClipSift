# ClipSift packaged-build smoke checklist

Use only safe test footage. The automated build check imports packaged modules and runs the offline System Check; it never constructs `GemmaVisionModel` and therefore does not load or download Gemma.

1. Open `dist\ClipSift\ClipSift.exe` and confirm no console window appears.
2. Confirm the ClipSift icon is visible in the window/taskbar and About shows the expected version.
3. Open **System Check**. Confirm it reports the expected GPU and VRAM, Hugging Face authentication, and cached Gemma files without network activity or model loading.
4. Use both folder pickers with paths containing spaces and Unicode characters.
5. Change a preference, close ClipSift, reopen it, and confirm the preference persists in `%LOCALAPPDATA%\ClipSift\settings.json`.
6. Confirm the GitHub, README / Help, and Gemma Model Page buttons open their fixed HTTPS destinations.
7. On the intended RTX test machine only, run a scan with safe footage and verify real packaged GPU inference. Record the genuine result; do not infer success from System Check.
