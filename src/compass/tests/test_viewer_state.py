# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

import pytest

pytest.importorskip("psygnal")

from compass.viewer import ViewerState


def test_defaults():
    s = ViewerState()
    assert s.current_level == 0
    assert s.visible_groups == frozenset()
    assert s.hovered_id is None


def test_field_signals_fire_on_change():
    s = ViewerState()
    seen = []
    s.events.viewport.connect(lambda *a: seen.append(("viewport", a)))
    s.events.selected_ids.connect(lambda *a: seen.append(("selected", a)))

    s.viewport = (10.0, 20.0, 500.0, 400.0)
    s.selected_ids = frozenset({1, 2, 3})
    assert [name for name, _ in seen] == ["viewport", "selected"]


def test_no_signal_on_equal_assignment():
    s = ViewerState()
    s.visible_groups = frozenset({1})
    fired = []
    s.events.visible_groups.connect(lambda *a: fired.append(a))
    s.visible_groups = frozenset({1})  # unchanged
    assert fired == []
    s.visible_groups = frozenset({1, 2})
    assert len(fired) == 1


def test_scale_property():
    s = ViewerState()
    s.canvas_size = (2000, 1000)
    s.viewport = (0.0, 0.0, 4000.0, 2000.0)
    assert s.scale == 0.5


def test_cross_widget_wiring_pattern():
    """The intended usage: one widget writes, another reacts."""
    s = ViewerState()
    canvas_redraws = []
    s.events.visible_groups.connect(lambda *a: canvas_redraws.append(s.visible_groups))
    # a panel toggles two group checkboxes
    s.visible_groups = s.visible_groups | {5}
    s.visible_groups = s.visible_groups | {7}
    assert canvas_redraws == [frozenset({5}), frozenset({5, 7})]
