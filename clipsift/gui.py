"""Phase 1 ttkbootstrap desktop interface for ClipSift."""

from __future__ import annotations

import os
import queue
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

import ttkbootstrap as ttk

from clipsift.scanner import ScanConfig, ScanEvent, scan_folder


class ClipSiftApp:
    """Tk main-thread event consumer backed by one scan worker."""

    DEVICE_VALUES = {"Auto": "auto", "GPU": "cuda", "CPU": "cpu"}
    STRATEGY_VALUES = {"Hybrid": "hybrid", "Uniform": "uniform", "Motion": "motion"}

    def __init__(self, root: ttk.Window) -> None:
        self.root = root
        self.events: queue.Queue[ScanEvent | tuple[str, str]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.counts = {"Person Detected": 0, "Needs Review": 0, "No Person Detected": 0}

        self.input_var = ttk.StringVar()
        self.output_var = ttk.StringVar(value=str(Path.cwd() / "ClipSift Results"))
        self.device_var = ttk.StringVar(value="Auto")
        self.strategy_var = ttk.StringVar(value="Hybrid")
        self.max_frames_var = ttk.IntVar(value=12)
        self.filename_var = ttk.StringVar(value="No scan running")
        self.operation_var = ttk.StringVar(value="Choose an input folder to begin")
        self.progress_var = ttk.DoubleVar(value=0.0)
        self.summary_vars = {name: ttk.StringVar(value="0") for name in self.counts}

        self._build()
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        self.root.title("ClipSift")
        self.root.geometry("1120x780")
        self.root.minsize(900, 650)

        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="ClipSift", font=("Segoe UI", 24, "bold"), bootstyle="success").pack(anchor="w")
        ttk.Label(outer, text="Local CCTV review aid", font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 16))

        controls = ttk.Labelframe(outer, text="Scan configuration", padding=12)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        self._folder_row(controls, 0, "Input folder", self.input_var, self._choose_input)
        self._folder_row(controls, 1, "Output folder", self.output_var, self._choose_output)

        options = ttk.Frame(controls)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        for column in range(6):
            options.columnconfigure(column, weight=1 if column in {1, 3, 5} else 0)
        ttk.Label(options, text="Device").grid(row=0, column=0, padx=(0, 8))
        ttk.Combobox(options, textvariable=self.device_var, values=list(self.DEVICE_VALUES), state="readonly", width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(options, text="Sampling").grid(row=0, column=2, padx=(20, 8))
        ttk.Combobox(options, textvariable=self.strategy_var, values=list(self.STRATEGY_VALUES), state="readonly", width=12).grid(row=0, column=3, sticky="w")
        ttk.Label(options, text="Maximum frames").grid(row=0, column=4, padx=(20, 8))
        ttk.Spinbox(options, from_=1, to=100, textvariable=self.max_frames_var, width=8).grid(row=0, column=5, sticky="w")

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=12)
        self.start_button = ttk.Button(actions, text="Start Scan", command=self._start_scan, bootstyle="success")
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel_scan, state="disabled", bootstyle="secondary")
        self.cancel_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(actions, text="Open Results Folder", command=self._open_results, bootstyle="outline-success")
        self.open_button.pack(side="right")

        status = ttk.Labelframe(outer, text="Progress", padding=12)
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.filename_var, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(status, textvariable=self.operation_var).pack(anchor="w", pady=(2, 8))
        ttk.Progressbar(status, variable=self.progress_var, maximum=100, bootstyle="success-striped").pack(fill="x")

        summaries = ttk.Frame(outer)
        summaries.pack(fill="x", pady=12)
        for name, style in (("Person Detected", "success"), ("Needs Review", "warning"), ("No Person Detected", "secondary")):
            card = ttk.Labelframe(summaries, text=name, padding=10)
            card.pack(side="left", fill="x", expand=True, padx=(0, 8))
            ttk.Label(card, textvariable=self.summary_vars[name], font=("Segoe UI", 18, "bold"), bootstyle=style).pack()

        results_box = ttk.Labelframe(outer, text="Results", padding=8)
        results_box.pack(fill="both", expand=True)
        columns = ("filename", "classification", "timestamp", "confidence", "frames", "elapsed")
        self.results = ttk.Treeview(results_box, columns=columns, show="headings", height=9)
        headings = ("Filename", "Classification", "Trigger timestamp", "Confidence", "Frames analysed", "Elapsed time")
        widths = (210, 145, 155, 100, 110, 100)
        for column, heading, width in zip(columns, headings, widths):
            self.results.heading(column, text=heading)
            self.results.column(column, width=width, anchor="w")
        scrollbar = ttk.Scrollbar(results_box, orient="vertical", command=self.results.yview)
        self.results.configure(yscrollcommand=scrollbar.set)
        self.results.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        log_box = ttk.Labelframe(outer, text="Activity log", padding=8)
        log_box.pack(fill="x", pady=(12, 0))
        self.log = scrolledtext.ScrolledText(log_box, height=6, wrap="word", state="disabled", bg="#171a1d", fg="#d7e0d9", insertbackground="white")
        self.log.pack(fill="x")

    def _folder_row(self, parent, row: int, label: str, variable, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse…", command=command, bootstyle="outline-secondary").grid(row=row, column=2, padx=(10, 0), pady=4)

    def _choose_input(self) -> None:
        selected = filedialog.askdirectory(title="Choose CCTV input folder")
        if selected:
            self.input_var.set(selected)
            if not self.output_var.get():
                self.output_var.set(str(Path(selected).parent / "ClipSift Results"))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Choose results folder")
        if selected:
            self.output_var.set(selected)

    def _start_scan(self) -> None:
        input_folder = Path(self.input_var.get().strip())
        output_text = self.output_var.get().strip()
        if not input_folder.is_dir():
            messagebox.showerror("ClipSift", "Choose an existing input folder.")
            return
        if not output_text:
            messagebox.showerror("ClipSift", "Choose an output folder.")
            return
        try:
            max_frames = int(self.max_frames_var.get())
            if max_frames <= 0:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror("ClipSift", "Maximum frames must be greater than zero.")
            return

        self.cancel_event = threading.Event()
        self.counts = {name: 0 for name in self.counts}
        for name, variable in self.summary_vars.items():
            variable.set("0")
        for item in self.results.get_children():
            self.results.delete(item)
        self.progress_var.set(0)
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.filename_var.set("Preparing scan")
        self.operation_var.set("Validating input")

        config = ScanConfig(
            input_folder=input_folder,
            output_folder=Path(output_text),
            device=self.DEVICE_VALUES[self.device_var.get()],
            preset="safe",
            sampling_strategy=self.STRATEGY_VALUES[self.strategy_var.get()],
            max_frames=max_frames,
        )
        self.worker = threading.Thread(target=self._scan_worker, args=(config,), daemon=True, name="ClipSiftScan")
        self.worker.start()

    def _scan_worker(self, config: ScanConfig) -> None:
        try:
            scan_folder(config, on_event=self.events.put, cancel_event=self.cancel_event)
        except Exception as exc:
            self.events.put(("fatal", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))

    def _cancel_scan(self) -> None:
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.operation_var.set("Cancellation requested; finishing the current frame")
        self._append_log("Cancellation requested.")

    def _poll_events(self) -> None:
        try:
            while True:
                self._handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: ScanEvent | tuple[str, str]) -> None:
        if isinstance(event, tuple):
            self._append_log(event[1])
            messagebox.showerror("ClipSift scan failed", event[1].splitlines()[0])
            self._set_idle("Scan failed")
            return
        self._append_log(f"{event.filename + ': ' if event.filename else ''}{event.message}")
        if event.filename:
            self.filename_var.set(event.filename)
        if event.kind in {"operation", "video_started", "model_loaded"}:
            self.operation_var.set(event.message)
        if event.total:
            self.progress_var.set(event.progress * 100)
        if event.kind in {"video_completed", "video_error"} and event.result is not None:
            result = event.result
            row = result.row
            timestamp = "—" if result.trigger_timestamp is None else f"{result.trigger_timestamp:.1f}s"
            self.results.insert("", "end", values=(Path(row.video_path).name, row.status, timestamp, result.trigger_confidence or "—", row.frames_analysed, f"{result.elapsed_seconds:.1f}s"))
            self.counts[row.status] += 1
            self.summary_vars[row.status].set(str(self.counts[row.status]))
        if event.kind == "completed":
            self.progress_var.set(100)
            self._set_idle("Scan complete")
        elif event.kind == "cancelled":
            self._set_idle("Scan cancelled; completed results were saved")

    def _set_idle(self, operation: str) -> None:
        self.operation_var.set(operation)
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.worker = None

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open_results(self) -> None:
        path = Path(self.output_var.get().strip())
        if not path.is_dir():
            messagebox.showinfo("ClipSift", "The results folder does not exist yet.")
            return
        os.startfile(path)  # type: ignore[attr-defined]


def main() -> None:
    root = ttk.Window(themename="darkly")
    ClipSiftApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
