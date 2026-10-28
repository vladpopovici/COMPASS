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
import html
import logging
import sys
from pathlib import Path

#: main-window shortcuts (the canvas' own keys live in navigation.CANVAS_KEYS)
APP_KEYS: list[tuple[str, str]] = [
    ("q, Ctrl+Q", "quit"),
    ("F11", "toggle fullscreen"),
    ("i", "show/hide the info bar and overview thumbnail"),
    ("a", "show/hide the annotations and their panel"),
    ("?, F1", "show this help"),
]


def format_value(values, channel_names=None, max_channels: int = 8) -> str:
    """Pixel value text: "RGB (r, g, b)" for RGB, else "name=v ..."."""
    fmt = lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.4g}"
    names = list(channel_names or [])
    if len(values) == 3 and names in ([], ["R", "G", "B"]):
        return "RGB (" + ", ".join(fmt(v) for v in values) + ")"
    if len(names) != len(values):
        names = [f"ch{i}" for i in range(len(values))]
    items = [f"{n}={fmt(v)}" for n, v in zip(names, values)]
    if len(items) > max_channels:
        items = items[:max_channels] + ["…"]
    return " ".join(items)


def format_info(state, raster) -> str:
    """Info-bar text: level, zoom, resolution on screen, cursor, value."""
    scale = state.scale  # screen px per level-0 px
    mpp0 = raster.native_resolution
    parts = [f"level {state.current_level}",
             f"zoom {scale:.3g}x (≈{raster.native_magnification * scale:.3g}x objective)",
             f"{mpp0 / scale:.3g} µm/screen px"]
    if state.cursor is not None:
        x, y = state.cursor
        parts.append(f"cursor ({x:.0f}, {y:.0f}) px = ({x * mpp0:.1f}, {y * mpp0:.1f}) µm")
    if state.cursor_value is not None:
        parts.append(format_value(state.cursor_value,
                                  getattr(raster, "channel_names", None))
                     + f" @ level {state.cursor_value_level}")
    return "   |   ".join(parts)


def help_html() -> str:
    from .navigation import MOUSE_HELP, help_rows
    sections = [("Keyboard", [*help_rows(), *APP_KEYS]), ("Mouse", MOUSE_HELP)]
    body = ""
    for title, rows in sections:
        body += f"<tr><td colspan=2><br><u>{title}</u></td></tr>"
        body += "".join(f"<tr><td><b>{html.escape(k)}</b></td>"
                        f"<td>&nbsp;&nbsp;{html.escape(d)}</td></tr>" for k, d in rows)
    return f"<table>{body}</table>"


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
    parser.add_argument("--annot", type=Path, action="append", default=[], metavar="FILE",
                        help="annotation store (.ann.sqlite) to draw over the slide; "
                             "repeat for several")
    parser.add_argument("--colors", type=Path, metavar="TOML",
                        help="group colours: a [colors] table or [[group]] entries "
                             "with name and color (a vocabulary file works)")
    parser.add_argument("--annot-width", type=float, default=2.0, metavar="PX",
                        help="annotation outline width in screen pixels (default: 2)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="log tile traffic to stderr")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="{asctime} {name} {levelname}: {message}", style="{")

    from vispy import app as vispy_app
    vispy_app.use_app("pyside6")

    from PySide6 import QtCore, QtGui, QtWidgets

    from ..core import ZarrRaster
    from .canvas import SlideCanvas

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    overlay = None
    if args.annot:
        from .overlay import load_overlay, read_colors
        missing = [str(p) for p in args.annot if not p.is_file()]
        if missing:
            parser.error(f"annotation store not found: {', '.join(missing)}")
        try:
            colors = read_colors(args.colors) if args.colors else None
            overlay = load_overlay(args.annot, colors)
        except (OSError, ValueError) as e:
            parser.error(str(e))
        for w in overlay.warnings:
            print(f"warning: {w}", file=sys.stderr)
        if not overlay.groups:
            print("warning: the annotation stores hold no objects", file=sys.stderr)
    elif args.colors:
        parser.error("--colors needs at least one --annot")

    raster = ZarrRaster(args.slide)
    canvas = SlideCanvas(raster,
                         tile_size=args.tile_size,
                         cache_bytes=args.cache_mb * 2 ** 20,
                         max_workers=args.workers)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"COMPASS — {args.slide.name}")
    win.setCentralWidget(canvas.native)

    # annotation overlay and its panel
    annot = panel = None
    if overlay is not None and overlay.groups:
        from .overlay_panel import AnnotationPanel
        from .overlay_visuals import AnnotationOverlay
        annot = AnnotationOverlay(canvas, overlay, line_width=args.annot_width)
        panel = AnnotationPanel(overlay, canvas.state, win)
        win.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, panel)

    # info bar
    info = QtWidgets.QLabel()
    win.statusBar().addWidget(info, 1)

    def refresh_info(*_):
        if win.statusBar().isVisible():
            info.setText(format_info(canvas.state, raster))

    for sig in (canvas.state.events.viewport, canvas.state.events.current_level,
                canvas.state.events.cursor, canvas.state.events.cursor_value):
        sig.connect(refresh_info)

    # app-level shortcuts (window-wide, so they work whatever has focus)
    def add_action(keys, slot):
        act = QtGui.QAction(win)
        act.setShortcuts([QtGui.QKeySequence(k.strip()) for k in keys.split(",")])
        act.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        act.triggered.connect(slot)
        win.addAction(act)

    def toggle_fullscreen():
        win.showNormal() if win.isFullScreen() else win.showFullScreen()

    def toggle_info():
        show = not win.statusBar().isVisible()
        win.statusBar().setVisible(show)
        canvas.overview_visible = show  # thumbnail follows the info bar
        refresh_info()

    def toggle_annotations():
        if annot is None:
            return
        show = not annot.visible
        annot.visible = show
        panel.setVisible(show)

    def show_help():
        QtWidgets.QMessageBox.information(win, "COMPASS — keys", help_html())

    add_action("q, Ctrl+Q", win.close)
    add_action("F11", toggle_fullscreen)
    add_action("i", toggle_info)
    add_action("a", toggle_annotations)
    add_action("?, F1", show_help)

    win.resize(1280, 900)
    win.show()
    canvas.native.setFocus()
    refresh_info()

    rc = qapp.exec()
    if annot is not None:
        annot.close()
    canvas.close()  # stop tile reads before the interpreter shuts down
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
