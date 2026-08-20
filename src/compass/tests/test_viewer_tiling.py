# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

import numpy as np
import pytest

from compass.viewer import TileCache, TileGrid, TileKey


@pytest.fixture
def grid() -> TileGrid:
    # 10000x8000 base, halving, 4 levels; 512px tiles
    return TileGrid(widths=[10000, 5000, 2500, 1250],
                    heights=[8000, 4000, 2000, 1000],
                    magnif_step=2.0, tile_size=512)


def test_basic_geometry(grid):
    assert grid.nlevels == 4
    assert grid.tile_size == 512
    assert grid.downsample(0) == 1.0 and grid.downsample(3) == 8.0
    assert grid.grid_size(0) == (20, 16)   # ceil(10000/512), ceil(8000/512)
    assert grid.grid_size(3) == (3, 2)


def test_level_for_scale(grid):
    assert grid.level_for_scale(1.0) == 0
    assert grid.level_for_scale(4.0) == 0     # zoomed past 1:1 -> clamp
    assert grid.level_for_scale(0.5) == 1     # exactly one level-1 px per screen px
    assert grid.level_for_scale(0.4) == 1     # sharper-or-equal: stay at 1
    assert grid.level_for_scale(0.25) == 2
    assert grid.level_for_scale(0.12) == 3
    assert grid.level_for_scale(0.001) == 3   # clamp to coarsest
    # positive bias tolerates a coarser level
    assert grid.level_for_scale(0.4, bias=0.5) == 1
    assert grid.level_for_scale(0.3, bias=0.8) == 2
    with pytest.raises(ValueError):
        grid.level_for_scale(0.0)


def test_tiles_in_viewport_level0(grid):
    keys = grid.tiles_in_viewport((0, 0, 1024, 1024), level=0)
    assert len(keys) == 4
    assert {(k.ix, k.iy) for k in keys} == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert all(k.level == 0 for k in keys)


def test_tiles_in_viewport_margin_and_clipping(grid):
    keys = grid.tiles_in_viewport((0, 0, 1024, 1024), level=0, margin=1)
    # margin ring clipped at the top-left image corner: 3x3 instead of 4x4
    assert {(k.ix, k.iy) for k in keys} == {(x, y) for x in range(3) for y in range(3)}


def test_tiles_in_viewport_level_conversion(grid):
    # viewport in level-0 coords maps to level-2 tiles via downsample 4
    keys = grid.tiles_in_viewport((0, 0, 4096, 4096), level=2)
    # 4096/4 = 1024 level-2 px -> 2x2 tiles
    assert {(k.ix, k.iy) for k in keys} == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_tiles_center_first_ordering(grid):
    keys = grid.tiles_in_viewport((0, 0, 3 * 512, 3 * 512), level=0)
    assert len(keys) == 9
    assert (keys[0].ix, keys[0].iy) == (1, 1)  # center tile first


def test_tiles_outside_viewport(grid):
    assert grid.tiles_in_viewport((20000, 20000, 500, 500), level=0) == []
    assert grid.tiles_in_viewport((0, 0, 0, 100), level=0) == []


def test_tile_bounds_edge_clipping(grid):
    # last column at level 0: 10000 = 19*512 + 272
    assert grid.tile_bounds_px(TileKey(0, 19, 0)) == (9728, 0, 272, 512)
    assert grid.tile_bounds_px(TileKey(0, 0, 15)) == (0, 7680, 512, 320)
    with pytest.raises(ValueError):
        grid.tile_bounds_px(TileKey(0, 20, 0))


def test_tile_bounds_level0(grid):
    x0, y0, w, h = grid.tile_bounds_level0(TileKey(1, 1, 2))
    assert (x0, y0, w, h) == (1024.0, 2048.0, 1024.0, 1024.0)


def test_from_raster():
    class FakeRaster:
        nlevels = 3
        widths = np.array([4000, 2000, 1000])
        heights = np.array([3000, 1500, 750])

        def between_level_scaling_factor(self, a, b):
            return 2.0 ** (a - b)

    g = TileGrid.from_raster(FakeRaster(), tile_size=256)
    assert g.nlevels == 3 and g.downsample(1) == 2.0 and g.tile_size == 256


def test_cache_lru_eviction():
    tile = np.zeros((512, 512, 3), np.uint8)  # 768 KiB
    cache = TileCache(max_bytes=3 * tile.nbytes)
    k = [TileKey(0, i, 0) for i in range(4)]
    for i in range(3):
        cache.put(k[i], tile)
    assert len(cache) == 3 and cache.nbytes == 3 * tile.nbytes

    cache.get(k[0])          # refresh k0 -> k1 becomes LRU
    cache.put(k[3], tile)    # over budget -> evict k1
    assert k[1] not in cache
    assert all(key in cache for key in (k[0], k[2], k[3]))


def test_cache_replace_and_oversized():
    cache = TileCache(max_bytes=100)
    big = np.zeros(1000, np.uint8)
    key = TileKey(0, 0, 0)
    cache.put(key, big)
    # a single entry above budget is kept rather than thrashing
    assert cache.get(key) is big and cache.nbytes == 1000
    small = np.zeros(10, np.uint8)
    cache.put(key, small)  # replacement updates the byte accounting
    assert cache.nbytes == 10 and len(cache) == 1
    cache.clear()
    assert len(cache) == 0 and cache.nbytes == 0
