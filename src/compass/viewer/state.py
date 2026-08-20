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


class ViewerState(EventedModel):
    """Cross-widget viewer state.

    Coordinates are (x, y) in level-0 pixel units, matching the annotation
    store and the raster protocol.
    """

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
    #: annotation object under the cursor, if any
    hovered_id: int | None = None

    @property
    def scale(self) -> float:
        """Displayed device pixels per level-0 pixel (zoom factor)."""
        return self.canvas_size[0] / self.viewport[2] if self.viewport[2] > 0 else 1.0
