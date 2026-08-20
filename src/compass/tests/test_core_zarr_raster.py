# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

import numpy as np
import pytest
import shapely.geometry as shg

from compass.core import ImageShape, Px, ZarrRaster, rasterize_polygon, write_pyramid


@pytest.fixture(scope="module")
def levels() -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    base = rng.integers(0, 255, size=(1200, 1600, 3), dtype=np.uint8)
    return [base, base[::2, ::2], base[::4, ::4]]


@pytest.fixture(scope="module")
def store_path(tmp_path_factory, levels):
    path = tmp_path_factory.mktemp("zarr") / "slide.zarr"
    write_pyramid(path, levels,
                  base_mpp=0.25, base_objective_power=40.0, mag_step=2.0,
                  chunk_size=256, shard_factor=2)
    return path


@pytest.fixture
def mri(store_path) -> ZarrRaster:
    return ZarrRaster(store_path)


def test_open_missing_store_raises(tmp_path):
    with pytest.raises(ValueError):
        ZarrRaster(tmp_path / "nope.zarr")


def test_geometry(mri, levels):
    assert mri.nlevels == 3
    assert mri.shape(0) == ImageShape(width=1600, height=1200)
    assert mri.shape(2) == ImageShape(width=400, height=300)
    assert mri.nchannels == 3
    assert mri.channel_names == ["R", "G", "B"]
    assert mri.native_resolution == 0.25
    assert mri.native_magnification == 40.0


def test_group_and_level_attrs(mri):
    assert mri.info["level_count"] == 3
    assert mri.info["base_mag_step"] == 2.0
    li = mri.level_info(1)
    assert li["scale_factor"] == 2.0
    assert li["mpp_x"] == 0.5
    assert li["objective_power"] == 20.0


def test_get_region_px_matches_source(mri, levels):
    reg = mri.get_region_px(100, 200, 300, 150, level=0)
    np.testing.assert_array_equal(reg, levels[0][200:350, 100:400])
    reg1 = mri.get_region_px(10, 20, 64, 32, level=1)
    np.testing.assert_array_equal(reg1, levels[1][20:52, 10:74])


def test_get_region_named_args(mri, levels):
    reg = mri.get_region(Px(x=0, y=0), ImageShape(width=16, height=8), level=2)
    np.testing.assert_array_equal(reg, levels[2][0:8, 0:16])


def test_get_plane(mri, levels):
    np.testing.assert_array_equal(mri.get_plane(2), levels[2])


def test_dask_variants(mri, levels):
    reg = mri.get_region_px_dask(100, 200, 300, 150, level=0).compute()
    np.testing.assert_array_equal(reg, levels[0][200:350, 100:400])
    np.testing.assert_array_equal(mri.get_plane_dask(1).compute(), levels[1])


def test_out_of_bounds_raises(mri):
    with pytest.raises(RuntimeError):
        mri.get_region_px(1500, 0, 200, 100, level=0)  # crosses right edge
    with pytest.raises(RuntimeError):
        mri.get_region_px(0, 0, 10, 10, level=3)  # no such level
    with pytest.raises(RuntimeError):
        mri.get_plane(-1)


def test_convert_px(mri):
    p = mri.convert_px(Px(x=100, y=200), from_level=0, to_level=1)
    assert (p.x, p.y) == (50, 100)
    p = mri.convert_px(Px(x=50, y=100), from_level=1, to_level=0)
    assert (p.x, p.y) == (100, 200)


def test_polygonal_region(mri, levels):
    # triangle inside the level-2 plane
    tri = shg.Polygon([(50, 50), (150, 50), (100, 130)])
    img = mri.get_polygonal_region_px(tri, level=2)
    assert img.shape == (80, 100, 3)
    # pixels well inside the triangle keep their values...
    np.testing.assert_array_equal(img[30, 50], levels[2][80, 100])
    # ...and the bottom corners (outside the triangle) are zeroed
    assert img[-1, 0].sum() == 0 and img[-1, -1].sum() == 0


def test_rasterize_polygon():
    sq = shg.Polygon([(2, 2), (7, 2), (7, 7), (2, 7)])
    mask = rasterize_polygon(sq, (10, 10))
    assert mask[4, 4] == 1 and mask[2, 2] == 1
    assert mask[0, 0] == 0 and mask[9, 9] == 0
    # off-canvas polygon yields empty mask
    off = shg.Polygon([(20, 20), (30, 20), (25, 30)])
    assert rasterize_polygon(off, (10, 10)).sum() == 0


def test_writer_validates_input(tmp_path):
    with pytest.raises(ValueError):
        write_pyramid(tmp_path / "x.zarr", [],
                      base_mpp=0.25, base_objective_power=40.0)
    a = np.zeros((64, 64, 3), np.uint8)
    b = np.zeros((32, 32), np.uint8)  # channel mismatch
    with pytest.raises(ValueError):
        write_pyramid(tmp_path / "y.zarr", [a, b],
                      base_mpp=0.25, base_objective_power=40.0)
    with pytest.raises(ValueError):
        write_pyramid(tmp_path / "z.zarr", [a],
                      base_mpp=0.25, base_objective_power=40.0,
                      channel_names=["only-one"])


def test_single_channel_map(tmp_path):
    seg = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 7
    write_pyramid(tmp_path / "map.zarr", [seg, seg[::2, ::2]],
                  base_mpp=0.5, base_objective_power=20.0,
                  channel_names=["label"])
    m = ZarrRaster(tmp_path / "map.zarr")
    assert m.nchannels == 1
    assert m.info["dimension_names"] == ["y", "x"]
    np.testing.assert_array_equal(m.get_plane(0), seg)
    assert m.get_plane(0).dtype == np.uint16
