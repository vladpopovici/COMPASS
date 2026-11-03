# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Shared reactive viewer state (psygnal ``EventedModel``).

One instance of :class:`ViewerState` is the single source of truth every
panel subscribes to (docs/architecture.md section 6): the canvas writes
``viewport``/``current_level`` as the user navigates, panels write
``visible_groups``/``selected_ids`` and everyone reacts through the
per-field signals (``state.events.<field>.connect(...)``).

All field types are immutable (tuples/frozensets) so change detection and
signal emission work by simple equality — mutate by assignment, never in
place.
"""

from psygnal import EventedModel
from pydantic import ConfigDict, Field

#: annotation line width limits (screen pixels) and the step of the keys
ANNOT_WIDTH_MIN, ANNOT_WIDTH_MAX, ANNOT_WIDTH_STEP = 0.5, 12.0, 0.5
#: image opacity step of the keys (range is 0..1)
IMAGE_OPACITY_STEP = 0.1


class ViewerState(EventedModel):
    """Cross-widget viewer state.

    Coordinates are (x, y) in level-0 pixel units, matching the annotation
    store and the raster protocol. Assignments are validated (the display
    settings have ranges); an out-of-range value raises ``ValueError``.
    """

    model_config = ConfigDict(validate_assignment=True)

    #: current viewport as (x0, y0, width, height), level-0 pixels
    viewport: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    #: canvas size in device pixels (width, height)
    canvas_size: tuple[int, int] = (1, 1)
    #: pyramid level the canvas is currently rendering from
    current_level: int = 0
    #: annotation groups toggled visible (group_ids)
    visible_groups: frozenset[int] = frozenset()
    #: currently selected annotation object_ids
    selected_ids: frozenset[int] = frozenset()
    #: mouse position over the canvas, level-0 pixels (None: not yet seen)
    cursor: tuple[float, float] | None = None
    #: pixel value under the cursor (one entry per channel), read from the
    #: finest loaded tile there; None when nothing is loaded at the cursor
    cursor_value: tuple[float, ...] | None = None
    #: pyramid level ``cursor_value`` was read from
    cursor_value_level: int | None = None
    #: annotation object under the cursor, if any
    hovered_id: int | None = None
    #: annotation outline width, screen pixels (point markers scale with it)
    annot_width: float = Field(2.0, ge=ANNOT_WIDTH_MIN, le=ANNOT_WIDTH_MAX)
    #: fill polygon annotations with their colour (translucent)
    fill_annotations: bool = False
    #: opacity of that fill (0..1)
    fill_opacity: float = Field(0.3, ge=0.0, le=1.0)
    #: image opacity, 1 = opaque, 0 = fully faded into the background
    image_opacity: float = Field(1.0, ge=0.0, le=1.0)
    #: light (white) instead of dark background; the faded image blends to it
    light_background: bool = False

    @property
    def scale(self) -> float:
        """Displayed device pixels per level-0 pixel (zoom factor)."""
        return self.canvas_size[0] / self.viewport[2] if self.viewport[2] > 0 else 1.0
