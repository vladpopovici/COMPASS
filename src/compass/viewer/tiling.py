# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Tile geometry and caching for the gigapixel canvas — pure logic, no GUI
dependencies, fully testable headless.

The canvas works in level-0 pixel coordinates ("world" coordinates, shared
with the annotation store). Each pyramid level is cut into a grid of
``tile_size`` x ``tile_size`` tiles; :class:`TileGrid` answers which level
to render at a given zoom and which tiles a viewport needs;
:class:`TileCache` is a byte-budgeted, thread-safe LRU of decoded tile
arrays.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from math import ceil, floor, log

import numpy as np


@dataclass(frozen=True, slots=True)
class TileKey:
    """Identity of one tile: pyramid level + grid column/row."""
    level: int
    ix: int
    iy: int


#####
class TileGrid:
    """Tile arithmetic for one pyramidal raster.

    Args:
        widths, heights: per-level extents in pixels, level 0 first
            (e.g. ``raster.widths`` / ``raster.heights``).
        magnif_step: scale factor between consecutive levels.
        tile_size: tile edge in pixels; match the Zarr chunk size so one
            tile read maps to whole chunks.
    """

    def __init__(self, widths, heights,
                 magnif_step: float = 2.0,
                 tile_size: int = 512):
        self._widths = np.asarray(widths, dtype=np.int64)
        self._heights = np.asarray(heights, dtype=np.int64)
        if self._widths.size != self._heights.size or self._widths.size == 0:
            raise ValueError("widths/heights must be non-empty and equal length")
        self._step = float(magnif_step)
        self._tile_size = int(tile_size)

    @classmethod
    def from_raster(cls, raster, tile_size: int = 512) -> "TileGrid":
        """Build a grid for a :class:`compass.core.PyramidalRaster`."""
        step = (raster.between_level_scaling_factor(1, 0)
                if raster.nlevels > 1 else 2.0)
        return cls(raster.widths, raster.heights,
                   magnif_step=step, tile_size=tile_size)

    @property
    def nlevels(self) -> int:
        return int(self._widths.size)

    @property
    def tile_size(self) -> int:
        return self._tile_size

    def downsample(self, level: int) -> float:
        """Level-L pixel size expressed in level-0 pixels."""
        return self._step ** level

    def grid_size(self, level: int) -> tuple[int, int]:
        """(nx, ny): number of tile columns/rows at a level."""
        return (ceil(self._widths[level] / self._tile_size),
                ceil(self._heights[level] / self._tile_size))

    def level_for_scale(self, scale: float, bias: float = 0.0) -> int:
        """Pick the pyramid level for a zoom factor.

        Args:
            scale: displayed device pixels per level-0 pixel (>1 means
                zoomed in beyond 1:1).
            bias: added to the ideal (fractional) level before flooring;
                0.0 renders sharper-or-equal to screen resolution, positive
                values tolerate slightly coarser levels (less data, softer).
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        ideal = log(1.0 / scale) / log(self._step) + bias
        return max(0, min(self.nlevels - 1, floor(ideal + 1e-9)))

    def tiles_in_viewport(self,
                          viewport: tuple[float, float, float, float],
                          level: int,
                          margin: int = 0) -> list[TileKey]:
        """Tiles of ``level`` intersecting a viewport, center-first.

        Args:
            viewport: (x0, y0, width, height) in level-0 pixels.
            level: pyramid level of the tiles.
            margin: extra rings of tiles around the viewport (prefetch).

        Returns:
            TileKeys ordered by distance from the viewport center, so
            fetching in list order fills the middle of the screen first.
        """
        x0, y0, w, h = viewport
        if w <= 0 or h <= 0:
            return []
        d = self.downsample(level)
        ts = self._tile_size
        lx0, ly0 = x0 / d, y0 / d
        lx1, ly1 = (x0 + w) / d, (y0 + h) / d

        nx, ny = self.grid_size(level)
        ix0 = max(0, floor(lx0 / ts) - margin)
        iy0 = max(0, floor(ly0 / ts) - margin)
        ix1 = min(nx - 1, ceil(lx1 / ts) - 1 + margin)
        iy1 = min(ny - 1, ceil(ly1 / ts) - 1 + margin)
        if ix0 > ix1 or iy0 > iy1:
            return []

        cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
        keys = [TileKey(level, ix, iy)
                for iy in range(iy0, iy1 + 1)
                for ix in range(ix0, ix1 + 1)]
        keys.sort(key=lambda k: (((k.ix + 0.5) * ts - cx) ** 2 +
                                 ((k.iy + 0.5) * ts - cy) ** 2,
                                 k.iy, k.ix))
        return keys

    def tile_bounds_px(self, key: TileKey) -> tuple[int, int, int, int]:
        """(x0, y0, width, height) of a tile in pixels at its own level,
        clipped to the level extent (edge tiles are partial)."""
        ts = self._tile_size
        x0, y0 = key.ix * ts, key.iy * ts
        w = int(min(ts, self._widths[key.level] - x0))
        h = int(min(ts, self._heights[key.level] - y0))
        if w <= 0 or h <= 0:
            raise ValueError(f"tile {key} outside level extent")
        return x0, y0, w, h

    def tile_bounds_level0(self, key: TileKey) -> tuple[float, float, float, float]:
        """(x0, y0, width, height) of a tile in level-0 pixels — where the
        tile sits in world coordinates."""
        d = self.downsample(key.level)
        x0, y0, w, h = self.tile_bounds_px(key)
        return x0 * d, y0 * d, w * d, h * d
##


#####
class TileCache:
    """Thread-safe LRU cache of decoded tile arrays with a byte budget.

    The most recently inserted/accessed tiles survive; eviction never
    removes the just-inserted entry (a single tile larger than the whole
    budget is kept rather than thrashing).
    """

    def __init__(self, max_bytes: int = 512 * 2 ** 20):
        self._max_bytes = int(max_bytes)
        self._data: OrderedDict[TileKey, np.ndarray] = OrderedDict()
        self._nbytes = 0
        self._lock = threading.RLock()

    def get(self, key: TileKey) -> np.ndarray | None:
        with self._lock:
            arr = self._data.get(key)
            if arr is not None:
                self._data.move_to_end(key)
            return arr

    def put(self, key: TileKey, arr: np.ndarray) -> None:
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._nbytes -= old.nbytes
            self._data[key] = arr
            self._nbytes += arr.nbytes
            while self._nbytes > self._max_bytes and len(self._data) > 1:
                _, evicted = self._data.popitem(last=False)
                self._nbytes -= evicted.nbytes

    def __contains__(self, key: TileKey) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._nbytes

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._nbytes = 0
##
