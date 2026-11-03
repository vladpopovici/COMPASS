# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtWidgets  # noqa: E402

from compass.viewer.state import ViewerState  # noqa: E402
from compass.viewer.view_panel import (ViewPanel, load_settings, nudge_opacity,  # noqa: E402
                                       nudge_width, save_settings, toggle_background, toggle_fill)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_state_defaults_and_limits():
    st = ViewerState()
    assert (st.annot_width, st.image_opacity, st.light_background) == (2.0, 1.0, False)
    with pytest.raises(ValueError):
        st.annot_width = 0.0
    with pytest.raises(ValueError):
        st.image_opacity = 1.5


def test_state_signals():
    st = ViewerState()
    seen = []
    st.events.annot_width.connect(lambda v: seen.append(("w", v)))
    st.events.image_opacity.connect(lambda v: seen.append(("o", v)))
    st.annot_width = 3.5
    st.image_opacity = 0.4
    assert seen == [("w", 3.5), ("o", 0.4)]


def test_nudges_clamp():
    st = ViewerState()
    nudge_width(st, +1)
    assert st.annot_width == 2.5
    for _ in range(40):
        nudge_width(st, -1)
    assert st.annot_width == 0.5
    for _ in range(40):
        nudge_width(st, +1)
    assert st.annot_width == 12.0
    for _ in range(3):
        nudge_opacity(st, -1)
    assert st.image_opacity == pytest.approx(0.7)
    for _ in range(20):
        nudge_opacity(st, -1)
    assert st.image_opacity == 0.0
    for _ in range(20):
        nudge_opacity(st, +1)
    assert st.image_opacity == 1.0
    toggle_background(st)
    assert st.light_background is True


def test_fill_state_and_toggle():
    st = ViewerState()
    assert st.fill_annotations is False and st.fill_opacity == 0.3
    toggle_fill(st)
    assert st.fill_annotations is True
    with pytest.raises(ValueError):
        st.fill_opacity = -0.1


def test_panel_follows_state_and_writes_state(qapp):
    st = ViewerState()
    panel = ViewPanel(st)
    st.annot_width = 4.0
    st.image_opacity = 0.25
    st.light_background = True
    assert panel._width.value() == 8
    assert panel._opacity.value() == 25
    assert panel._light.isChecked()
    assert panel._opacity_lbl.text() == "25 %"
    panel._opacity.setValue(60)
    panel._width.setValue(6)
    panel._light.setChecked(False)
    assert st.image_opacity == pytest.approx(0.6)
    assert st.annot_width == 3.0
    assert st.light_background is False
    assert not panel._fill_op.isEnabled()
    panel._fill.setChecked(True)
    assert st.fill_annotations and panel._fill_op.isEnabled()
    panel._fill_op.setValue(45)
    assert st.fill_opacity == pytest.approx(0.45)
    panel.reset()
    assert (st.annot_width, st.image_opacity, st.light_background) == (2.0, 1.0, False)
    assert st.fill_annotations is False and st.fill_opacity == 0.3
    assert panel._width.value() == 4


def test_settings_roundtrip_and_bad_values(qapp, tmp_path):
    ini = str(tmp_path / "s.ini")
    cfg = QtCore.QSettings(ini, QtCore.QSettings.Format.IniFormat)
    st = ViewerState(annot_width=5.0, image_opacity=0.3, light_background=True,
                     fill_annotations=True, fill_opacity=0.6)
    save_settings(st, cfg)
    st2 = ViewerState()
    got = load_settings(st2, QtCore.QSettings(ini, QtCore.QSettings.Format.IniFormat))
    assert got == {"annot_width": 5.0, "image_opacity": 0.3, "light_background": True,
                   "fill_annotations": True, "fill_opacity": 0.6}
    assert (st2.annot_width, st2.image_opacity, st2.light_background) == (5.0, 0.3, True)

    bad = QtCore.QSettings(str(tmp_path / "b.ini"), QtCore.QSettings.Format.IniFormat)
    bad.setValue("view/annot_width", "abc")
    bad.setValue("view/image_opacity", 7)
    st3 = ViewerState()
    assert load_settings(st3, bad) == {}
    assert st3.annot_width == 2.0 and st3.image_opacity == 1.0
