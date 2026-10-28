# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""vispy visuals for the annotation overlay.

GUI-only (needs vispy and, to instantiate, OpenGL). The data and colours come
from :mod:`compass.viewer.overlay`. Every group becomes at most two visuals
under the canvas' scene graph, in level-0 pixel coordinates: one ``Line`` of
independent segments (polygon outlines, polylines) and one ``Markers`` (points).
Line width and marker size are in screen pixels, so outlines keep their
thickness at every zoom. Visibility follows ``ViewerState.visible_groups``.
"""

import numpy as np
from vispy import scene

from .overlay import Overlay, OverlayGroup
from .state import ViewerState

#: draw order of the overlay: above every tile (tiles use -level, the
#: overview -1000), below the zoom-box band (1000)
OVERLAY_ORDER = 500


class AnnotationOverlay:
    """Draws an :class:`Overlay` on a :class:`SlideCanvas`.

    Args:
        canvas: the slide canvas (its ``view.scene`` is the parent).
        overlay: loaded groups.
        state: viewer state; ``visible_groups`` selects what is shown. It is
            set to all groups on construction.
        line_width: outline width, screen pixels.
        marker_size: point marker diameter, screen pixels.
    """

    def __init__(self, canvas, overlay: Overlay, state: ViewerState | None = None,
                 line_width: float = 2.0, marker_size: float = 9.0):
        self._canvas = canvas
        self._overlay = overlay
        self._state = state if state is not None else canvas.state
        self._visuals: dict[int, list] = {}
        self._enabled = True
        for g in overlay.groups:
            self._visuals[g.uid] = self._make_visuals(g, line_width, marker_size)
        self._state.visible_groups = frozenset(g.uid for g in overlay.groups)
        self._state.events.visible_groups.connect(self._apply)
        self._apply()

    # -- building -----------------------------------------------------

    def _make_visuals(self, g: OverlayGroup, width: float, size: float) -> list:
        parent = self._canvas.view.scene
        out = []
        if len(g.segments):
            line = scene.visuals.Line(pos=np.ascontiguousarray(g.segments, np.float32),
                                      color=g.color, width=width,
                                      connect="segments", method="gl", parent=parent)
            self._stack(line)
            out.append(line)
        if len(g.points):
            mk = scene.visuals.Markers(parent=parent)
            mk.set_data(np.ascontiguousarray(g.points, np.float32),
                        face_color=g.color, edge_color=(0, 0, 0, 1),
                        edge_width=1.0, size=size)
            self._stack(mk)
            out.append(mk)
        return out

    @staticmethod
    def _stack(visual) -> None:
        # same reason as the tiles: all layers sit at z=0, order alone decides
        visual.set_gl_state("translucent", depth_test=False, cull_face=False)
        visual.order = OVERLAY_ORDER

    # -- visibility ---------------------------------------------------

    @property
    def visible(self) -> bool:
        """Master switch for the whole overlay (groups keep their own state)."""
        return self._enabled

    @visible.setter
    def visible(self, value: bool) -> None:
        self._enabled = bool(value)
        self._apply()

    def _apply(self, *_) -> None:
        shown = self._state.visible_groups
        for uid, vis in self._visuals.items():
            for v in vis:
                v.visible = self._enabled and uid in shown
        self._canvas.update()

    def set_group_color(self, uid: int, rgba) -> None:
        for v in self._visuals.get(uid, []):
            if isinstance(v, scene.visuals.Line):
                v.set_data(color=rgba)
            else:
                g = self._overlay.by_uid()[uid]
                v.set_data(np.ascontiguousarray(g.points, np.float32), face_color=rgba,
                           edge_color=(0, 0, 0, 1), edge_width=1.0)
        self._canvas.update()

    def close(self) -> None:
        self._state.events.visible_groups.disconnect(self._apply)
        for vis in self._visuals.values():
            for v in vis:
                v.parent = None
        self._visuals.clear()
