# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Tests for compass.connector ingestion.

The planning helpers are tested directly. The wsi2zarr pipeline is tested
end-to-end against fake pyvips/WSI stand-ins (numpy-backed, injected via
sys.modules) because this cannot rely on libvips/OpenSlide being installed
— what matters here is that the produced store opens as a valid
compass.core.ZarrRaster with the right pixels and metadata, which is
independent of the real image codecs.
"""

import sys
import types

import numpy as np
import pytest

from compass.connector import plan_levels, resolve_crop, wsi2zarr
from compass.core import ImageShape, ZarrRaster


# ----------------------------------------------------------------------
# Pure planning helpers
# ----------------------------------------------------------------------

def test_plan_levels_halving():
    levels = plan_levels(1400, 1100, downscale_factor=2, min_size=256)
    assert [(lv['w'], lv['h']) for lv in levels] == [(1400, 1100), (700, 550), (350, 275)]


def test_plan_levels_scales_crop_origin():
    levels = plan_levels(2000, 2000, x0=101, y0=57, downscale_factor=2, min_size=500)
    assert [(lv['x0'], lv['y0']) for lv in levels] == [(101, 57), (50, 28), (25, 14)]


def test_plan_levels_tiny_image_single_level():
    assert len(plan_levels(300, 300, min_size=256)) == 1


def test_resolve_crop_whole_image():
    assert resolve_crop(1000, 800, {}, False) == (0, 0, 1000, 800)
    # auto-crop without a scan ROI falls back to the whole image
    assert resolve_crop(1000, 800, {}, True) == (0, 0, 1000, 800)


def test_resolve_crop_auto_roi():
    roi = {'x0': 10, 'y0': 20, 'width': 300, 'height': 400}
    assert resolve_crop(1000, 800, roi, True) == (10, 20, 300, 400)
    # explicit crop=False ignores the ROI
    assert resolve_crop(1000, 800, roi, False) == (0, 0, 1000, 800)


def test_resolve_crop_explicit_clamped():
    assert resolve_crop(1000, 800, {}, (100, 100, 200, 200)) == (100, 100, 200, 200)
    # origin clamped into the image, extent clamped to what remains
    assert resolve_crop(1000, 800, {}, (-5, 900, 5000, 100)) == (0, 799, 1000, 1)


# ----------------------------------------------------------------------
# wsi2zarr against numpy-backed fakes
# ----------------------------------------------------------------------

class FakeVipsImage:
    """Minimal numpy-backed stand-in for the pyvips.Image API wsi2zarr uses."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    @property
    def height(self) -> int:
        return self.arr.shape[0]

    @property
    def width(self) -> int:
        return self.arr.shape[1]

    @staticmethod
    def new_from_file(path, **kwargs):
        with open(path, "rb") as f:
            return FakeVipsImage(np.load(f))

    def write_to_file(self, path):
        with open(path, "wb") as f:
            np.save(f, self.arr)

    def crop(self, x0, y0, w, h):
        return FakeVipsImage(self.arr[y0:y0 + h, x0:x0 + w])

    def flatten(self, background=0):
        return self  # test images carry no alpha

    def resize(self, scale, gap=None):
        f = round(1.0 / scale)
        h, w = self.arr.shape[0] // f, self.arr.shape[1] // f
        return FakeVipsImage(self.arr[::f, ::f][:h, :w])

    def numpy(self):
        return self.arr


class FakeWSI:
    def __init__(self, path):
        with open(path, "rb") as f:
            arr = np.load(f)
        self.info = {
            'objective_power': 40.0,
            'width': arr.shape[1],
            'height': arr.shape[0],
            'mpp_x': 0.25,
            'mpp_y': 0.25,
            'roi': {'x0': 8, 'y0': 16, 'width': 1400, 'height': 1100},
            'background': 0xFF,
        }


@pytest.fixture
def fake_slide(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, size=(1200, 1500, 3), dtype=np.uint8)
    slide_path = tmp_path / "slide.mrxs"
    with open(slide_path, "wb") as f:
        np.save(f, arr)

    fake_pyvips = types.ModuleType("pyvips")
    fake_pyvips.Image = FakeVipsImage
    monkeypatch.setitem(sys.modules, "pyvips", fake_pyvips)

    fake_wsi_mod = types.ModuleType("compass.connector.wsi")
    fake_wsi_mod.WSI = FakeWSI
    monkeypatch.setitem(sys.modules, "compass.connector.wsi", fake_wsi_mod)

    return slide_path, arr


def test_wsi2zarr_roundtrip(fake_slide, tmp_path):
    slide_path, arr = fake_slide
    dst = tmp_path / "slide.zarr"
    wsi2zarr(slide_path, dst, crop=True,  # auto-crop to the fake scan ROI
             min_size=256, chunk_size=64, shard_factor=2)

    m = ZarrRaster(dst)
    # ROI is 1400x1100 -> levels 1400x1100, 700x550, 350x275
    assert m.nlevels == 3
    assert m.shape(0) == ImageShape(width=1400, height=1100)
    assert m.shape(2) == ImageShape(width=350, height=275)

    # pixels: level 0 must be exactly the cropped source
    np.testing.assert_array_equal(m.get_plane(0), arr[16:16 + 1100, 8:8 + 1400])

    # metadata contract with ZarrRaster / write_pyramid
    assert m.info["level_count"] == 3
    assert m.info["base_mpp"] == 0.25
    assert m.info["base_mag_step"] == 2
    assert m.info["base_objective_power"] == 40.0
    assert m.channel_names == ["R", "G", "B"]
    li = m.level_info(1)
    assert li["scale_factor"] == 2 and li["mpp_x"] == 0.5 and li["objective_power"] == 20.0

    # magnification wiring end to end
    assert m.get_mpp_for_level(2) == 1.0
    assert m.get_level_for_mpp(0.5) == 1


def test_wsi2zarr_no_crop_and_overwrite(fake_slide, tmp_path):
    slide_path, arr = fake_slide
    dst = tmp_path / "full.zarr"
    wsi2zarr(slide_path, dst, crop=False, min_size=512, chunk_size=64, shard_factor=2)
    m = ZarrRaster(dst)
    assert m.shape(0) == ImageShape(width=1500, height=1200)
    np.testing.assert_array_equal(m.get_plane(0), arr)

    # refuses to clobber without overwrite=True...
    with pytest.raises(Exception):
        wsi2zarr(slide_path, dst, crop=False, min_size=512)
    # ...and replaces cleanly with it
    wsi2zarr(slide_path, dst, crop=False, min_size=512, chunk_size=64,
             shard_factor=2, overwrite=True)
    assert ZarrRaster(dst).nlevels == m.nlevels


def test_wsi2zarr_band_size_must_align():
    with pytest.raises(ValueError):
        wsi2zarr("x", "y", chunk_size=512, shard_factor=8, band_size=1000)
