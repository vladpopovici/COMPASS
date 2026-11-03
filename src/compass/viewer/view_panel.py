# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Qt dock with the display settings: annotation width, image opacity and
light/dark background. It only writes :class:`ViewerState`; the canvas and
the overlay follow. Also hosts the small persistence helpers (``QSettings``).
"""

from PySide6 import QtCore, QtWidgets

from .state import (ANNOT_WIDTH_MAX, ANNOT_WIDTH_MIN, ANNOT_WIDTH_STEP,
                    IMAGE_OPACITY_STEP, ViewerState)

_WIDTH_SCALE = int(round(1 / ANNOT_WIDTH_STEP))  # slider ticks per pixel

DEFAULTS = {"annot_width": 2.0, "image_opacity": 1.0, "light_background": False}


class ViewPanel(QtWidgets.QDockWidget):
    """Sliders for annotation width / image opacity and a background switch."""

    def __init__(self, state: ViewerState, parent=None):
        super().__init__("View", parent)
        self.setObjectName("view-panel")
        self._state = state
        self._busy = False

        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        nofocus = QtCore.Qt.FocusPolicy.NoFocus  # keys stay on the canvas

        self._width = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._width.setRange(int(round(ANNOT_WIDTH_MIN * _WIDTH_SCALE)),
                             int(round(ANNOT_WIDTH_MAX * _WIDTH_SCALE)))
        self._width_lbl = QtWidgets.QLabel()
        self._opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity_lbl = QtWidgets.QLabel()
        self._light = QtWidgets.QCheckBox("Light background")
        self._reset = QtWidgets.QPushButton("Reset")
        for w in (self._width, self._opacity, self._light, self._reset):
            w.setFocusPolicy(nofocus)

        form.addRow("Annotation width", self._row(self._width, self._width_lbl))
        form.addRow("Image opacity", self._row(self._opacity, self._opacity_lbl))
        form.addRow(self._light)
        form.addRow(self._reset)
        self.setWidget(box)

        self._width.valueChanged.connect(self._on_width)
        self._opacity.valueChanged.connect(self._on_opacity)
        self._light.toggled.connect(self._on_light)
        self._reset.clicked.connect(self.reset)
        for sig in (state.events.annot_width, state.events.image_opacity,
                    state.events.light_background):
            sig.connect(self._sync)
        self._sync()

    @staticmethod
    def _row(slider, label) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        label.setMinimumWidth(42)
        lay.addWidget(slider, 1)
        lay.addWidget(label)
        return w

    # -- widgets -> state ---------------------------------------------

    def _write(self, **kw) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            for k, v in kw.items():
                setattr(self._state, k, v)
        finally:
            self._busy = False
        self._sync(force=True)

    def _on_width(self, v: int) -> None:
        self._write(annot_width=v / _WIDTH_SCALE)

    def _on_opacity(self, v: int) -> None:
        self._write(image_opacity=v / 100.0)

    def _on_light(self, on: bool) -> None:
        self._write(light_background=bool(on))

    def reset(self) -> None:
        """Back to the defaults (width 2 px, opaque image, dark background)."""
        for k, v in DEFAULTS.items():
            setattr(self._state, k, v)

    # -- state -> widgets ---------------------------------------------

    def _sync(self, *_, force: bool = False) -> None:
        if self._busy and not force:
            return
        st = self._state
        self._busy = True
        try:
            self._width.setValue(int(round(st.annot_width * _WIDTH_SCALE)))
            self._opacity.setValue(int(round(st.image_opacity * 100)))
            self._light.setChecked(st.light_background)
        finally:
            self._busy = False
        self._width_lbl.setText(f"{st.annot_width:g} px")
        self._opacity_lbl.setText(f"{st.image_opacity * 100:.0f} %")


# -- key actions (also used by the tests) ------------------------------------

def nudge_width(state: ViewerState, direction: int) -> None:
    """Annotation width one step wider (+1) or thinner (-1), within limits."""
    w = state.annot_width + direction * ANNOT_WIDTH_STEP
    state.annot_width = round(min(max(w, ANNOT_WIDTH_MIN), ANNOT_WIDTH_MAX), 3)


def nudge_opacity(state: ViewerState, direction: int) -> None:
    """Image opacity one step more opaque (+1) or more faded (-1)."""
    o = state.image_opacity + direction * IMAGE_OPACITY_STEP
    state.image_opacity = round(min(max(o, 0.0), 1.0), 3)


def toggle_background(state: ViewerState) -> None:
    state.light_background = not state.light_background


# -- persistence -------------------------------------------------------------

def load_settings(state: ViewerState, settings=None) -> dict:
    """Read the saved display settings (``QSettings``); returns what was found.

    Invalid or missing values are ignored. ``settings`` is injectable for tests.
    """
    s = settings if settings is not None else QtCore.QSettings("COMPASS", "viewer")
    out = {}
    for key, kind in (("annot_width", float), ("image_opacity", float),
                      ("light_background", bool)):
        v = s.value(f"view/{key}")
        if v is None:
            continue
        try:
            out[key] = (str(v).lower() == "true") if kind is bool else float(v)
        except (TypeError, ValueError):
            continue
    applied = {}
    for k, v in out.items():
        try:
            setattr(state, k, v)
            applied[k] = v
        except ValueError:  # out of range: keep the default
            pass
    return applied


def save_settings(state: ViewerState, settings=None) -> None:
    s = settings if settings is not None else QtCore.QSettings("COMPASS", "viewer")
    for key in DEFAULTS:
        s.setValue(f"view/{key}", getattr(state, key))
    s.sync()
