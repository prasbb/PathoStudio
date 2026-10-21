"""Entrypoint: ``python -m ashby_studio``.

Launches the full annotation workspace (napari/PySide6) with a synthetic H&E
demo; use ``-i/--image`` to open a local H&E image at launch. With a display the
Qt event loop runs; in ``--offscreen`` mode the window is built and the process
returns (used by headless tests, which are not guaranteed to run on macOS).
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ashby_studio", description="Virchow's Suite (research only, not for diagnosis).")
    parser.add_argument("--offscreen", action="store_true", help="Build window with offscreen Qt + software OpenGL.")
    parser.add_argument("-i", "--image", default=None, help="Open this H&E image (png/tif/tiff) at launch.")
    args = parser.parse_args(argv)

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    try:
        from qtpy.QtWidgets import QApplication
        from .app import build_app
    except Exception as exc:  # noqa: BLE001
        print("No Qt backend available. Install PySide6 (see README).", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    viewer, win = build_app(offscreen=args.offscreen)
    if args.image:
        win.open_path(os.fspath(args.image))
    win.show()
    if args.offscreen:
        return 0

    app = QApplication.instance() or QApplication(sys.argv)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())