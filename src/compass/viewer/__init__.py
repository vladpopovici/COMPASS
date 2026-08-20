# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Qt viewer: vispy gigapixel canvas, psygnal reactive state, pyqtgraph and
matplotlib/seaborn analytical panels.

Depends on ``compass.core`` (data access) and optionally
``compass.processing`` (on-demand analysis). Never on ``compass.connector``.

The package import stays GUI-free: :class:`ViewerState` and the tile
machinery (grid, cache, fetcher) need no OpenGL/Qt and are unit-tested
headless. The vispy canvas and the Qt entry point are only reachable as
submodules (``compass.viewer.canvas``, ``compass.viewer.app``) or via
``python -m compass.viewer <slide.zarr>``.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from .fetch import TileFetcher
from .state import ViewerState
from .tiling import TileCache, TileGrid, TileKey

__all__ = [
    "TileCache",
    "TileFetcher",
    "TileGrid",
    "TileKey",
    "ViewerState",
]
