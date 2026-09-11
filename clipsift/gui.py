"""Phase 2 ttkbootstrap desktop interface for ClipSift."""

from __future__ import annotations

import queue
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

import ttkbootstrap as ttk
from PIL import Image, ImageTk

from clipsift import __version__
from clipsift.gui_support import (
    GuiPreferences,
    control_states,
    filter_results,
    load_preferences,
    open_local_path,
    save_preferences,
    toggle_log_state,
)
from clipsift.readiness import (
    GITHUB_URL,
    MODEL_URL,
    README_URL,
    DiagnosticReport,
    PreflightResult,
    open_external_url,
    run_preflight,
    run_system_check,
)
from clipsift.resources import resource_path
from clipsift.scanner import ScanConfig, ScanEvent, VideoScanResult, scan_folder


class ClipSiftApp:
    """Tk main-thread event consumer backed by one scan worker."""

    DEVICE_VALUES = {"Auto": "auto", "GPU": "cuda", "CPU": "cpu"}
    STRATEGY_VALUES = {"Hybrid": "hybrid", "Uniform": "uniform", "Motion": "motion"}
    FILTERS = ("All", "Person Detected", "Needs Review", "No Person Detected", "Errors")

    def __init__(self, root: ttk.Window) -> None:
        self.root = root
        icon_path = resource_path("assets/clipsift.ico")
        if icon_path.is_file():
            try:
                self.root.iconbitmap(default=str(icon_path))
            except Exception:
                pass
        self.preferences = load_preferences()
        self.events: queue.Queue[ScanEvent | tuple[object, ...]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.check_worker: threading.Thread | None = None
        self.all_results: list[VideoScanResult] = []
        self.visible_results: dict[str, VideoScanResult] = {}
        self.configuration_controls: list[tuple[object, str]] = []
        self.preview_photo = None
        self.selected_result: VideoScanResult | None = None
        self.log_expanded = False

        default_output = self.preferences.output_folder or str(Path.cwd() / "ClipSift Results")
        self.input_var = ttk.StringVar(value=self.preferences.input_folder)
        self.output_var = ttk.StringVar(value=default_output)
        self.device_var = ttk.StringVar(value=self.preferences.device)
        self.strategy_var = ttk.StringVar(value=self.preferences.sampling_strategy)
        self.max_frames_var = ttk.IntVar(value=self.preferences.max_frames)
        self.filename_var = ttk.StringVar(value="No scan running")
        self.operation_var = ttk.StringVar(value="Choose an input folder to begin")
        self.status_var = ttk.StringVar(value="● Idle")
        self.progress_var = ttk.DoubleVar(value=0.0)
        self.filter_var = ttk.StringVar(value="All")
        self.filter_labels = {name: ttk.StringVar() for name in self.FILTERS}
        self.summary_vars = {name: ttk.StringVar(value="0") for name in self.FILTERS[1:4]}

        self._build()
        self._update_filter_counts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        self.root.title("ClipSift")
        self.root.geometry(self.preferences.window_geometry)
        self.root.minsize(1000, 700)
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 14))
        title = ttk.Frame(header)
        title.pack(side="left")
        ttk.Label(title, text="ClipSift", font=("Segoe UI", 24, "bold"), bootstyle="success").pack(anchor="w")
        ttk.Label(title, text="Local CCTV review aid · observations by Gemma, decisions by ClipSift", font=("Segoe UI", 10)).pack(anchor="w")
        self.status_label = ttk.Label(header, textvariable=self.status_var, padding=(12, 6), bootstyle="secondary-inverse")
        self.status_label.pack(side="right", anchor="n")
        ttk.Button(header, text="About", command=self._show_about, bootstyle="link").pack(side="right", anchor="n", padx=4)
        self.check_button = ttk.Button(header, text="System Check", command=self._check_setup, bootstyle="secondary")
        self.check_button.pack(side="right", anchor="n", padx=4)

        controls = ttk.Labelframe(outer, text="Scan configuration", padding=12)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        self._folder_row(controls, 0, "Input folder", self.input_var, self._choose_input)
        self._folder_row(controls, 1, "Output folder", self.output_var, self._choose_output)
        options = ttk.Frame(controls)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(options, text="Device").pack(side="left")
        device = ttk.Combobox(options, textvariable=self.device_var, values=list(self.DEVICE_VALUES), state="readonly", width=12)
        device.pack(side="left", padx=(8, 24)); self.configuration_controls.append((device, "readonly"))
        ttk.ToolTip(device, text="Auto prefers CUDA. GPU requires CUDA. CPU forces CPU inference.")
        ttk.Label(options, text="Sampling strategy").pack(side="left")
        strategy = ttk.Combobox(options, textvariable=self.strategy_var, values=list(self.STRATEGY_VALUES), state="readonly", width=12)
        strategy.pack(side="left", padx=(8, 24)); self.configuration_controls.append((strategy, "readonly"))
        ttk.ToolTip(strategy, text="Hybrid mixes timeline and motion frames; Uniform covers time; Motion prioritises change.")
        ttk.Label(options, text="Maximum frames").pack(side="left")
        maximum = ttk.Spinbox(options, from_=1, to=100, textvariable=self.max_frames_var, width=8)
        maximum.pack(side="left", padx=(8, 0)); self.configuration_controls.append((maximum, "normal"))
        ttk.ToolTip(maximum, text="Maximum number of frames Gemma may inspect per video.")

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=10)
        self.start_button = ttk.Button(actions, text="Start Scan", command=self._start_scan, bootstyle="success")
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel_scan, state="disabled", bootstyle="secondary")
        self.cancel_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Open Results Folder", command=self._open_results, bootstyle="secondary").pack(side="right")

        progress = ttk.Frame(outer)
        progress.pack(fill="x", pady=(0, 10))
        ttk.Label(progress, textvariable=self.filename_var, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(progress, textvariable=self.operation_var).pack(anchor="w", pady=(1, 5))
        ttk.Progressbar(progress, variable=self.progress_var, maximum=100, bootstyle="success-striped").pack(fill="x")

        summaries = ttk.Frame(outer)
        summaries.pack(fill="x", pady=(0, 10))
        for name, style in (("Person Detected", "success"), ("Needs Review", "warning"), ("No Person Detected", "secondary")):
            box = ttk.Labelframe(summaries, padding=(8, 7))
            box.pack(side="left", fill="x", expand=True, padx=(0, 8))
            accent = ttk.Frame(box, width=4, bootstyle=style)
            accent.pack(side="left", fill="y", padx=(0, 9))
            ttk.Label(box, text=name).pack(side="left")
            ttk.Label(box, textvariable=self.summary_vars[name], font=("Segoe UI", 13, "bold"), bootstyle=style).pack(side="right")

        filters = ttk.Frame(outer)
        filters.pack(fill="x", pady=(0, 8))
        ttk.Label(filters, text="Show:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))
        self.filter_buttons = {}
        for name in self.FILTERS:
            button = ttk.Radiobutton(filters, textvariable=self.filter_labels[name], variable=self.filter_var, value=name, command=self._apply_filter, bootstyle="toolbutton")
            button.pack(side="left", padx=2)
            self.filter_buttons[name] = button

        content = ttk.Panedwindow(outer, orient="horizontal")
        content.pack(fill="both", expand=True)
        results_box = ttk.Labelframe(content, text="Scan results", padding=8)
        preview_box = ttk.Labelframe(content, text="Evidence preview", padding=10)
        content.add(results_box, weight=7)
        content.add(preview_box, weight=3)

        columns = ("filename", "classification", "timestamp", "confidence", "frames", "elapsed")
        self.results = ttk.Treeview(results_box, columns=columns, show="headings", height=10, selectmode="browse")
        headings = ("Filename", "Classification", "Trigger timestamp", "Confidence", "Frames", "Elapsed")
        widths = (190, 135, 120, 85, 65, 75)
        for column, heading, width in zip(columns, headings, widths):
            self.results.heading(column, text=heading)
            self.results.column(column, width=width, minwidth=55, anchor="w")
        self.results.tag_configure("person", foreground="#64d8bd")
        self.results.tag_configure("review", foreground="#f0bd62")
        self.results.tag_configure("clear", foreground="#a7adb3")
        self.results.tag_configure("error", foreground="#ff7979")
        self.results.bind("<<TreeviewSelect>>", self._result_selected)
        scrollbar = ttk.Scrollbar(results_box, orient="vertical", command=self.results.yview)
        self.results.configure(yscrollcommand=scrollbar.set)
        self.results.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.preview_label = ttk.Label(preview_box, text="Select a result to view evidence", anchor="center", justify="center")
        self.preview_label.pack(fill="both", expand=True, pady=(8, 12))
        preview_actions = ttk.Frame(preview_box)
        preview_actions.pack(fill="x")
        self.open_evidence_button = ttk.Button(preview_actions, text="Open Evidence", command=self._open_evidence, state="disabled", bootstyle="outline-success")
        self.open_evidence_button.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.open_video_button = ttk.Button(preview_actions, text="Open Video", command=self._open_video, state="disabled", bootstyle="outline-secondary")
        self.open_video_button.pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.log_toggle = ttk.Button(outer, text="Show activity log", command=self._toggle_log, bootstyle="link")
        self.log_toggle.pack(anchor="w", pady=(8, 0))
        self.log_box = ttk.Frame(outer)
        self.log = scrolledtext.ScrolledText(self.log_box, height=6, wrap="word", state="disabled", bg="#171a1d", fg="#d7e0d9", insertbackground="white", relief="flat")
        self.log.pack(fill="x")
        ttk.Label(outer, text="Video is processed locally and original footage is never modified.", font=("Segoe UI", 9), bootstyle="secondary").pack(anchor="w", pady=(7, 0))

    def _folder_row(self, parent, row: int, label: str, variable, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        browse = ttk.Button(parent, text="Browse…", command=command, bootstyle="secondary")
        browse.grid(row=row, column=2, padx=(10, 0), pady=4)
        self.configuration_controls.extend(((entry, "normal"), (browse, "normal")))

    def _choose_input(self) -> None:
        selected = filedialog.askdirectory(title="Choose CCTV input folder")
        if selected:
            self.input_var.set(selected)

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Choose results folder")
        if selected:
            self.output_var.set(selected)

    def _start_scan(self) -> None:
        input_folder = Path(self.input_var.get().strip())
        output_text = self.output_var.get().strip()
        if not input_folder.is_dir():
            messagebox.showerror("ClipSift", "Choose an existing input folder."); return
        if not output_text:
            messagebox.showerror("ClipSift", "Choose an output folder."); return
        try:
            max_frames = int(self.max_frames_var.get())
            if not 1 <= max_frames <= 100:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror("ClipSift", "Maximum frames must be between 1 and 100."); return

        self.cancel_event = threading.Event()
        self.all_results.clear()
        self._apply_filter()
        self.progress_var.set(0)
        self._set_scanning_controls(True)
        self.filename_var.set("Preparing scan")
        self.operation_var.set("Validating input")
        self._set_status("Loading Model", "warning")
        self._save_preferences()
        config = ScanConfig(
            input_folder=input_folder,
            output_folder=Path(output_text),
            device=self.DEVICE_VALUES[self.device_var.get()],
            preset="safe",
            sampling_strategy=self.STRATEGY_VALUES[self.strategy_var.get()],
            max_frames=max_frames,
        )
        self.worker = threading.Thread(target=self._preflight_and_scan, args=(config,), daemon=True, name="ClipSiftScan")
        self.worker.start()

    def _preflight_and_scan(self, config: ScanConfig) -> None:
        try:
            preflight = run_preflight(config.input_folder, config.output_folder, config.device)
            self.events.put(("preflight", preflight))
            if not preflight.can_start:
                return
            scan_folder(config, on_event=self.events.put, cancel_event=self.cancel_event)
        except Exception as exc:
            self.events.put(("fatal", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))

    def _check_setup(self) -> None:
        if self.check_worker is not None and self.check_worker.is_alive():
            return
        self.check_button.configure(state="disabled")
        self._set_status("Checking Setup", "warning")
        device = self.DEVICE_VALUES.get(self.device_var.get(), "auto")
        self.check_worker = threading.Thread(target=self._system_check_worker, args=(device,), daemon=True, name="ClipSiftCheck")
        self.check_worker.start()

    def _system_check_worker(self, device: str) -> None:
        try:
            self.events.put(("system_check", run_system_check(device)))
        except Exception as exc:
            self.events.put(("check_error", f"System check failed: {type(exc).__name__}: {exc}"))

    def _cancel_scan(self) -> None:
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.operation_var.set("Cancellation requested; finishing the current frame")
        self._set_status("Cancelling", "warning")
        self._append_log("Cancellation requested.")

    def _poll_events(self) -> None:
        try:
            while True:
                self._handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: ScanEvent | tuple[object, ...]) -> None:
        if isinstance(event, tuple):
            kind = str(event[0])
            if kind == "system_check":
                self.check_worker = None
                self.check_button.configure(state="normal")
                report = event[1]
                self._set_status("Idle", "secondary")
                self._show_system_check(report)  # type: ignore[arg-type]
            elif kind == "check_error":
                self.check_worker = None
                self.check_button.configure(state="normal")
                self._set_status("Error", "danger")
                messagebox.showerror("ClipSift System Check", str(event[1]))
            elif kind == "preflight":
                preflight = event[1]
                if isinstance(preflight, PreflightResult):
                    self._append_log(f"Preflight: {preflight.message}")
                    if not preflight.can_start:
                        messagebox.showerror("ClipSift cannot start", preflight.message)
                        self._finish_scan("Error", "danger", "Preflight check failed")
                        self._show_system_check(preflight.report)
            else:
                detail = str(event[1])
                self._append_log(detail)
                messagebox.showerror("ClipSift scan failed", detail.splitlines()[0])
                self._finish_scan("Error", "danger", "Scan failed")
            return
        self._append_log(f"{event.filename + ': ' if event.filename else ''}{event.message}")
        if event.filename:
            self.filename_var.set(event.filename)
        if event.kind == "operation":
            self.operation_var.set(event.message)
            if "Loading" in event.message:
                self._set_status("Loading Model", "warning")
        elif event.kind in {"video_started", "frame", "model_loaded"}:
            self.operation_var.set(event.message)
            if event.kind != "model_loaded":
                self._set_status("Scanning", "success")
        if event.total:
            self.progress_var.set(event.progress * 100)
        if event.kind in {"video_completed", "video_error"} and event.result is not None:
            self.all_results.append(event.result)
            self._apply_filter()
        if event.kind == "completed":
            self.progress_var.set(100)
            self._finish_scan("Complete", "success", "Scan complete")
        elif event.kind == "cancelled":
            self._finish_scan("Idle", "secondary", "Scan cancelled; completed results were saved")

    def _apply_filter(self) -> None:
        selected = self.filter_var.get()
        shown = filter_results(self.all_results, selected)
        for item in self.results.get_children():
            self.results.delete(item)
        self.visible_results.clear()
        for index, result in enumerate(shown):
            row = result.row
            timestamp = "—" if result.trigger_timestamp is None else f"{result.trigger_timestamp:.1f}s"
            confidence = result.trigger_confidence or "—"
            tag = "error" if row.error else {"Person Detected": "person", "Needs Review": "review", "No Person Detected": "clear"}.get(row.status, "clear")
            iid = f"result-{index}"
            self.results.insert("", "end", iid=iid, values=(Path(row.video_path).name, "Error" if row.error else row.status, timestamp, confidence, row.frames_analysed, f"{result.elapsed_seconds:.1f}s"), tags=(tag,))
            self.visible_results[iid] = result
        self._update_filter_counts()
        self._clear_preview("Select a result to view evidence")

    def _update_filter_counts(self) -> None:
        errors = sum(bool(item.row.error) for item in self.all_results)
        counts = {name: sum(not item.row.error and item.row.status == name for item in self.all_results) for name in self.FILTERS[1:4]}
        for name, variable in self.summary_vars.items():
            variable.set(str(counts[name]))
        totals = {"All": len(self.all_results), **counts, "Errors": errors}
        for name, variable in self.filter_labels.items():
            variable.set(f"{name} ({totals[name]})")

    def _result_selected(self, _event=None) -> None:
        selection = self.results.selection()
        if not selection:
            return
        result = self.visible_results.get(selection[0])
        if result is None:
            return
        self.selected_result = result
        self.open_video_button.configure(state="normal" if Path(result.row.video_path).is_file() else "disabled")
        evidence = Path(result.row.evidence_frame) if result.row.evidence_frame else None
        if evidence is None or not evidence.is_file():
            self._clear_preview("No evidence image available for this result")
            self.selected_result = result
            self.open_video_button.configure(state="normal" if Path(result.row.video_path).is_file() else "disabled")
            return
        try:
            with Image.open(evidence) as source:
                preview = source.convert("RGB")
                preview.thumbnail((360, 260), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self.preview_photo, text="")
            self.open_evidence_button.configure(state="normal")
        except (OSError, ValueError) as exc:
            self._clear_preview(f"Evidence preview unavailable\n{exc}")
            self.selected_result = result
            self.open_video_button.configure(state="normal" if Path(result.row.video_path).is_file() else "disabled")

    def _clear_preview(self, message: str) -> None:
        self.selected_result = None
        self.preview_photo = None
        self.preview_label.configure(image="", text=message)
        self.open_evidence_button.configure(state="disabled")
        self.open_video_button.configure(state="disabled")

    def _open_selected_path(self, path: str, label: str) -> None:
        ok, error = open_local_path(path)
        if not ok:
            messagebox.showerror("ClipSift", f"Could not open {label}.\n{error}")

    def _open_evidence(self) -> None:
        if self.selected_result and self.selected_result.row.evidence_frame:
            self._open_selected_path(self.selected_result.row.evidence_frame, "evidence")

    def _open_video(self) -> None:
        if self.selected_result:
            self._open_selected_path(self.selected_result.row.video_path, "video")

    def _open_results(self) -> None:
        self._open_selected_path(self.output_var.get().strip(), "results folder")

    def _set_scanning_controls(self, scanning: bool) -> None:
        states = control_states(scanning)
        for widget, idle_state in self.configuration_controls:
            state = states["readonly_configuration"] if idle_state == "readonly" else states["configuration"]
            widget.configure(state=state)
        self.start_button.configure(state=states["start"])
        self.cancel_button.configure(state=states["cancel"])
        self.check_button.configure(state=states["configuration"])

    def _set_status(self, text: str, style: str) -> None:
        self.status_var.set(f"● {text}")
        self.status_label.configure(bootstyle=f"{style}-inverse")

    def _finish_scan(self, status: str, style: str, operation: str) -> None:
        self.operation_var.set(operation)
        self._set_status(status, style)
        self._set_scanning_controls(False)
        self.worker = None
        self._save_preferences()

    def _toggle_log(self) -> None:
        self.log_expanded, label = toggle_log_state(self.log_expanded)
        if self.log_expanded:
            self.log_box.pack(fill="x", pady=(4, 0))
        else:
            self.log_box.pack_forget()
        self.log_toggle.configure(text=label)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _save_preferences(self) -> None:
        try:
            save_preferences(GuiPreferences(
                self.input_var.get().strip(), self.output_var.get().strip(), self.device_var.get(),
                self.strategy_var.get(), int(self.max_frames_var.get()), self.root.geometry(),
            ))
        except (OSError, ValueError, TypeError):
            pass

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("ClipSift", "A scan is still active. Cancel it and close ClipSift?"):
                return
            self.cancel_event.set()
        self._save_preferences()
        self.root.destroy()

    def _show_system_check(self, report: DiagnosticReport) -> None:
        dialog = ttk.Toplevel(self.root)
        dialog.title("ClipSift System Check")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=18)
        body.pack(fill="both", expand=True)
        style = {"Ready": "success", "Attention required": "warning", "Unavailable": "danger"}[report.status]
        ttk.Label(body, text=f"Setup status: {report.status}", font=("Segoe UI", 15, "bold"), bootstyle=style).pack(anchor="w", pady=(0, 10))
        for item in report.items:
            row = ttk.Frame(body)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=item.label, width=28, font=("Segoe UI", 9, "bold")).pack(side="left", anchor="n")
            ttk.Label(row, text=item.status, width=18, bootstyle={"Ready": "success", "Attention required": "warning", "Unavailable": "danger"}[item.status]).pack(side="left", anchor="n")
            ttk.Label(row, text=item.detail, width=54, wraplength=380, justify="left").pack(side="left", anchor="n")
        if not report.authenticated or not report.model_cached:
            ttk.Separator(body).pack(fill="x", pady=10)
            ttk.Label(body, text="Gemma setup", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            ttk.Label(body, text="Gemma requires a Hugging Face account and accepted model terms. ClipSift never requests, displays, or stores your token. After accepting access, run:  hf auth login", wraplength=650, justify="left").pack(anchor="w", pady=(3, 8))
            setup_actions = ttk.Frame(body)
            setup_actions.pack(fill="x")
            ttk.Button(setup_actions, text="Open Gemma Model Page", command=lambda: self._open_url(MODEL_URL), bootstyle="secondary").pack(side="left")
            ttk.Button(setup_actions, text="Retry Check", command=lambda: (dialog.destroy(), self._check_setup()), bootstyle="success").pack(side="left", padx=8)
        ttk.Button(body, text="Close", command=dialog.destroy, bootstyle="secondary").pack(anchor="e", pady=(14, 0))

    def _show_about(self) -> None:
        dialog = ttk.Toplevel(self.root)
        dialog.title("About ClipSift")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=20)
        body.pack()
        ttk.Label(body, text="ClipSift", font=("Segoe UI", 18, "bold"), bootstyle="success").pack(anchor="w")
        ttk.Label(body, text=f"Version {__version__}").pack(anchor="w", pady=(0, 10))
        ttk.Label(body, text="Video is processed locally and original footage is never modified.", wraplength=440, justify="left").pack(anchor="w", pady=(0, 12))
        links = ttk.Frame(body)
        links.pack(fill="x")
        ttk.Button(links, text="GitHub", command=lambda: self._open_url(GITHUB_URL), bootstyle="link").pack(side="left")
        ttk.Button(links, text="README / Help", command=lambda: self._open_url(README_URL), bootstyle="link").pack(side="left", padx=8)
        ttk.Button(body, text="Close", command=dialog.destroy, bootstyle="secondary").pack(anchor="e", pady=(12, 0))

    def _open_url(self, url: str) -> None:
        ok, error = open_external_url(url, webbrowser.open)
        if not ok:
            messagebox.showerror("ClipSift", error)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--packaged-smoke-check":
        from clipsift.packaged_smoke import run_packaged_smoke_check

        run_packaged_smoke_check(Path(sys.argv[2]))
        return
    root = ttk.Window(themename="darkly")
    ClipSiftApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
