"""Minimal packaged entry point that keeps smoke failures non-interactive."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--packaged-smoke-check":
        from clipsift.packaged_smoke import run_packaged_smoke_check

        raise SystemExit(run_packaged_smoke_check(Path(sys.argv[2])))

    from clipsift.gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
