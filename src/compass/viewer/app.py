# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Standalone viewer entry point:

    python -m compass.viewer slide.zarr

One Qt process, one ``QMainWindow``; the :class:`SlideCanvas` is the
central widget, and analytical panels will attach as dock widgets around
it (docs/architecture.md section 6). Logging is configured here, per
entry point — never at import time.
"""

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compass.viewer",
        description="COMPASS whole-slide viewer")
    parser.add_argument("slide", type=Path,
                        help="path to a pyramidal Zarr store "
                             "(see compass.connector.wsi2zarr)")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="tile edge in pixels (default: 512, "
                             "matching the ingestion chunk size)")
    parser.add_argument("--cache-mb", type=int, default=768,
                        help="decoded-tile RAM budget in MB (default: 768)")
    parser.add_argument("--workers", type=int, default=2,
                        help="tile reader threads (default: 2)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="log tile traffic to stderr")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="{asctime} {name} {levelname}: {message}", style="{")

    from vispy import app as vispy_app
    vispy_app.use_app("pyside6")

    from PySide6 import QtWidgets

    from ..core import ZarrRaster
    from .canvas import SlideCanvas

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    raster = ZarrRaster(args.slide)
    canvas = SlideCanvas(raster,
                         tile_size=args.tile_size,
                         cache_bytes=args.cache_mb * 2 ** 20,
                         max_workers=args.workers)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"COMPASS — {args.slide.name}")
    win.setCentralWidget(canvas.native)
    win.resize(1280, 900)
    win.show()

    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
