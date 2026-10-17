# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

import pytest

from compass.viewer.navigation import (CANVAS_KEYS, MAX_ZOOM, Navigator, View,
                                       ViewHistory, action_for_key, help_rows)

SW, SH = 1000.0, 500.0  # screen, physical px


@pytest.fixture
def nav():
    # 8 levels, step 2, slide 64000 x 16000
    return Navigator(64000, 16000, lambda lv: 2.0 ** lv, nlevels=8)


def test_view_rect_roundtrip():
    v = View(500.0, 300.0, 2.0)
    r = v.rect(SW, SH)
    assert r == (-500.0, -200.0, 2000.0, 1000.0)
    assert View.from_rect(r, SW) == v


def test_fit_covers_slide_centered(nav):
    v = nav.fit(SW, SH)
    x0, y0, w, h = v.rect(SW, SH)
    assert (v.cx, v.cy) == (32000, 8000)
    assert x0 < 0 and y0 < 0 and x0 + w > 64000 and y0 + h > 16000
    # limited by width here (64:16 slide on a 2:1 screen)
    assert v.upp == pytest.approx(64 * 1.04)


def test_zoom_about_center_and_limits(nav):
    v = View(1000, 1000, 4.0)
    z = nav.zoom(v, 2.0, SW, SH)
    assert (z.cx, z.cy, z.upp) == (1000, 1000, 2.0)
    assert nav.zoom(v, 0.5, SW, SH).upp == 8.0
    # zoom-in limit
    assert nav.zoom(View(1000, 1000, 0.2), 2.0, SW, SH).upp == 1 / MAX_ZOOM
    # zoom-out limit is fit-to-window
    assert nav.zoom(View(1000, 1000, 60.0), 0.5, SW, SH).upp == nav.fit(SW, SH).upp


def test_at_level(nav):
    v = View(1000, 1000, 4.0)
    assert nav.at_level(v, 0, SW, SH) == View(1000, 1000, 1.0)
    assert nav.at_level(v, 5, SW, SH).upp == 32.0
    assert nav.at_level(v, 8, SW, SH) is None  # slide has levels 0..7
    assert nav.at_level(v, -1, SW, SH) is None


def test_pan_and_center_clamp(nav):
    v = View(10000, 8000, 2.0)  # view is 2000 x 1000 level-0 px
    assert nav.pan(v, 0.25, 0, SW, SH) == View(10500, 8000, 2.0)
    assert nav.pan(v, 0, -1.0, SW, SH) == View(10000, 7000, 2.0)
    # the center cannot leave the slide
    assert nav.pan(View(100, 100, 2.0), -1.0, -1.0, SW, SH) == View(0, 0, 2.0)


def test_view_larger_than_slide_is_centered(nav):
    # at upp 40 the view is 40000 x 20000: taller than the slide (16000),
    # narrower than it (64000) -> y centered, x kept
    assert nav.clamp(View(100, 100, 40.0), SW, SH) == View(100, 8000, 40.0)
    # zooming out to the limit from a corner ends up fully centered
    v = nav.zoom(View(0, 0, 0.5), 1e-6, SW, SH)
    assert v == nav.fit(SW, SH)


def test_history_back_forward():
    h = ViewHistory()
    a, b, c = View(0, 0, 1), View(1, 1, 1), View(2, 2, 1)
    assert h.back(a) is None
    h.record(a, "fit")
    h.record(b, "level0")
    # now at c
    assert h.back(c) == b
    assert h.back(b) == a
    assert h.back(a) is None
    assert h.forward(a) == b
    assert h.forward(b) == c
    assert not h.can_forward


def test_history_collapses_repeats_and_new_nav_clears_forward():
    h = ViewHistory()
    v = [View(i, 0, 1) for i in range(5)]
    h.record(v[0], "pan")
    h.record(v[1], "pan")  # same kind: collapsed
    h.record(v[2], "pan")
    assert h.back(v[3]) == v[0]
    assert h.can_forward
    h.record(v[0], "zoom")  # new navigation clears forward
    assert not h.can_forward


@pytest.mark.parametrize("name,text,mods,expected", [
    ("Plus", "+", {"Shift"}, ("zoom", 2.0)),   # '+' typed with Shift (US)
    ("Plus", "+", set(), ("zoom", 2.0)),       # keypad '+'
    ("Equal", "=", set(), ("zoom", 2.0)),
    ("Minus", "-", set(), ("zoom", 0.5)),
    ("0", "0", set(), ("level", 0)),
    ("7", "7", set(), ("level", 7)),
    ("Home", "", set(), ("fit",)),
    ("F", "f", set(), ("fit",)),
    ("Left", "", set(), ("pan", -0.25, 0.0)),
    ("Left", "", {"Shift"}, ("pan", -1.0, 0.0)),
    ("Left", "", {"Alt"}, ("back",)),
    ("Right", "", {"Alt"}, ("forward",)),
    ("PageDown", "", set(), ("pan", 0.0, 1.0)),
    ("Backspace", "\b", set(), ("back",)),
    ("Q", "q", set(), None),                   # app-level, not the canvas
    ("Left", "", {"Control"}, None),
    (None, "", set(), None),
])
def test_action_for_key(name, text, mods, expected):
    assert action_for_key(name, text, mods) == expected


def test_help_rows_cover_all_actions():
    rows = help_rows()
    text = " ".join(k + " " + d for k, d in rows)
    for spec, action, _ in CANVAS_KEYS:
        if action[0] != "level":
            assert spec in text
    assert ("0", "pyramid level 0 at 1:1 (full resolution)") in rows
    assert ("Shift+Up, PageUp", "pan up one screen") in rows
    assert len({d for _, d in rows}) == len(rows)


# --- mouse helpers ---------------------------------------------------------

from compass.viewer.navigation import inset_rect, inset_to_slide  # noqa: E402


def test_zoom_at_keeps_point_fixed(nav):
    v = View(10000, 8000, 4.0)
    px, py = 10800.0, 7600.0
    z = nav.zoom_at(v, 2.0, px, py, SW, SH)
    assert z.upp == 2.0
    # the point keeps its screen offset from the view center
    assert (px - z.cx) / z.upp == pytest.approx((px - v.cx) / v.upp)
    assert (py - z.cy) / z.upp == pytest.approx((py - v.cy) / v.upp)
    # clamped at the zoom-in limit, still anchored
    z = nav.zoom_at(View(10000, 8000, 0.2), 4.0, px, py, SW, SH)
    assert z.upp == 1 / MAX_ZOOM
    assert (px - z.cx) / z.upp == pytest.approx((px - 10000) / 0.2)


def test_to_box(nav):
    # 2000 x 400 box on a 1000 x 500 screen: width-limited, upp 2
    assert nav.to_box(1000, 1000, 3000, 1400, SW, SH) == View(2000, 1200, 2.0)
    # corners in any order
    assert nav.to_box(3000, 1400, 1000, 1000, SW, SH) == View(2000, 1200, 2.0)
    assert nav.to_box(5, 5, 5, 50, SW, SH) is None
    # tiny box -> zoom-in limit
    assert nav.to_box(100, 100, 101, 101, SW, SH).upp == 1 / MAX_ZOOM


def test_center_on(nav):
    assert nav.center_on(View(0, 0, 2.0), 5000, 6000, SW, SH) == View(5000, 6000, 2.0)


def test_history_kind_none_never_collapses():
    h = ViewHistory()
    a, b, c = View(0, 0, 1), View(1, 0, 1), View(2, 0, 1)
    h.record(a)
    h.record(b)
    assert h.back(c) == b
    assert h.back(b) == a


def test_inset_rect_and_mapping():
    # 4:1 slide on a large canvas: 220 x 55 thumbnail, top-right, 10 px margin
    r = inset_rect(1000, 800, 64000, 16000)
    assert r == pytest.approx((1000 - 10 - 220, 10, 220, 55))
    x, y, w, h = r
    assert inset_to_slide(x, y, r, 64000, 16000) == (0, 0)
    assert inset_to_slide(x + w, y + h, r, 64000, 16000) == pytest.approx((64000, 16000))
    assert inset_to_slide(x + w / 2, y + h / 2, r, 64000, 16000) == pytest.approx((32000, 8000))
    assert inset_to_slide(x - 1, y, r, 64000, 16000) is None
    # small canvas: thumbnail shrinks to at most half of it
    assert inset_rect(200, 100, 1000, 1000)[2] == pytest.approx(40)


def test_format_value():
    from compass.viewer.app import format_value
    assert format_value((220.0, 180.0, 200.0)) == "RGB (220, 180, 200)"
    assert format_value((220.0, 180.0, 200.0), ["R", "G", "B"]) == "RGB (220, 180, 200)"
    assert format_value((1.0, 2.5), ["DAPI", "CD3"]) == "DAPI=1 CD3=2.5"
    assert format_value((7.0,)) == "ch0=7"
    assert format_value(tuple(float(i) for i in range(10))).endswith("ch7=7 …")
