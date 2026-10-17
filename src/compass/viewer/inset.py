# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Overview thumbnail drawn in a corner of the slide canvas: the whole
slide (coarsest level) with the current viewport outlined. GUI-only
(vispy); geometry lives in :mod:`compass.viewer.navigation`. Mouse input is
routed here by :class:`compass.viewer.canvas.SlideCanvas`.
"""

import numpy as np
from vispy import scene
from vispy.visuals.transforms import STTransform

from .navigation import inset_rect, inset_to_slide

#: smallest drawn viewport outline, in thumbnail pixels
_MIN_BOX_PX = 6.0


#####
class _StaticCamera(scene.PanZoomCamera):
    """Camera ignoring all user input (vispy cameras listen to the whole
    canvas: Backspace would reset it, drags would pan it)."""

    def viewbox_key_event(self, event):
        pass

    def viewbox_mouse_event(self, event):
        pass
##


#####
class OverviewInset:
    """Thumbnail of the whole slide in the canvas' top-right corner.

    Args:
        canvas: the SceneCanvas to draw on.
        plane: coarsest-level image (H, W[, C]).
        downsample: level-0 pixels per ``plane`` pixel.
        slide_w, slide_h: level-0 slide extent.
    """

    def __init__(self, canvas: scene.SceneCanvas, plane: np.ndarray,
                 downsample: float, slide_w: float, slide_h: float):
        self._slide = (float(slide_w), float(slide_h))
        self._rect = (0.0, 0.0, 1.0, 1.0)
        self._canvas_size = None

        self._vb = scene.widgets.ViewBox(parent=canvas.scene,
                                         border_color=(1, 1, 1, 0.7),
                                         bgcolor=(0.08, 0.08, 0.08, 0.9))
        self._vb.order = 100  # above the slide view
        cam = _StaticCamera(aspect=1)
        cam.flip = (False, True, False)
        self._vb.camera = cam

        img = scene.visuals.Image(plane, parent=self._vb.scene,
                                  interpolation="linear")
        img.transform = STTransform(scale=(downsample, downsample, 1))
        img.set_gl_state("translucent", depth_test=False, cull_face=False)

        self._box = scene.visuals.Line(pos=np.zeros((5, 2), np.float32),
                                       color=(1.0, 0.85, 0.0, 1.0), width=2,
                                       parent=self._vb.scene, method="gl")
        self._box.set_gl_state("translucent", depth_test=False)
        self._box.order = 1

    @property
    def visible(self) -> bool:
        return self._vb.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._vb.visible = bool(value)

    def layout(self, canvas_w: float, canvas_h: float) -> None:
        """Place the thumbnail for a canvas size (logical pixels)."""
        if self._canvas_size == (canvas_w, canvas_h):
            return
        self._canvas_size = (canvas_w, canvas_h)
        self._rect = inset_rect(canvas_w, canvas_h, *self._slide)
        x, y, w, h = self._rect
        self._vb.pos = (x, y)
        self._vb.size = (w, h)
        self._vb.camera.set_range(x=(0, self._slide[0]), y=(0, self._slide[1]),
                                  margin=0)

    def set_viewport(self, viewport: tuple[float, float, float, float]) -> None:
        """Outline the visible part of the slide (level-0 x0, y0, w, h)."""
        x0, y0, w, h = viewport
        # keep tiny (deep-zoom) outlines visible
        min_w = _MIN_BOX_PX * self._slide[0] / max(self._rect[2], 1.0)
        if w < min_w:
            x0, w = x0 + w / 2 - min_w / 2, min_w
        if h < min_w:
            y0, h = y0 + h / 2 - min_w / 2, min_w
        x1, y1 = x0 + w, y0 + h
        self._box.set_data(pos=np.array([[x0, y0], [x1, y0], [x1, y1],
                                         [x0, y1], [x0, y0]], np.float32))

    def slide_point(self, px: float, py: float) -> tuple[float, float] | None:
        """Level-0 point under canvas pixel (px, py), or None if the
        thumbnail is hidden or not under it."""
        if not self.visible:
            return None
        return inset_to_slide(px, py, self._rect, *self._slide)

    def slide_point_clamped(self, px: float, py: float) -> tuple[float, float] | None:
        """Like :meth:`slide_point`, but a point outside the thumbnail is
        clamped to its edge (for drags that leave it)."""
        if not self.visible:
            return None
        x, y, w, h = self._rect
        return inset_to_slide(min(max(px, x), x + w), min(max(py, y), y + h),
                              self._rect, *self._slide)
##
