# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Tests for compass.processing.tissue on a synthetic H&E-like image."""

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("cv2")

from compass.processing.tissue import METHODS, detect_foreground, foreground_bbox  # noqa: E402


@pytest.fixture(scope="module")
def slide():
    """Near-white noisy background with a textured pink/purple ellipse."""
    rng = np.random.default_rng(0)
    img = np.full((300, 400, 3), 240, np.uint8)
    img += rng.integers(0, 10, img.shape, dtype=np.uint8)
    yy, xx = np.mgrid[:300, :400]
    blob = ((yy - 150) / 75) ** 2 + ((xx - 175) / 100) ** 2 < 1
    tex = rng.integers(0, 60, blob.shape)
    tissue = np.stack([180 - tex, 90 + tex // 2, 160 - tex // 3], -1)
    img[blob] = tissue[blob].astype(np.uint8)
    return img, blob


@pytest.mark.parametrize("method", ["simple-he", "simple", "htk"])
def test_detect_foreground_recovers_blob(slide, method):
    img, blob = slide
    mask, _ = detect_foreground(img, method=method)
    assert mask.dtype == bool and mask.shape == blob.shape
    assert (mask == blob).mean() > 0.98
    assert foreground_bbox(mask) == foreground_bbox(blob)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_all_methods_return_masks(slide, method):
    img, _ = slide
    mask, th = detect_foreground(img, method=method.upper())  # case-insensitive
    assert mask.dtype == bool and mask.shape == img.shape[:2]
    assert th is None or np.isscalar(th)


def test_simple_he_estimates_threshold(slide):
    img, _ = slide
    _, th = detect_foreground(img, method="simple-he")
    assert 200 < th < 245
    _, th = detect_foreground(img, method="simple-he", g_th=210)
    assert th == 210


def test_unknown_method_and_kwarg(slide):
    img, _ = slide
    with pytest.raises(ValueError, match="unknown"):
        detect_foreground(img, method="nope")
    with pytest.raises(TypeError):
        detect_foreground(img, method="simple-he", bogus=1)


def test_foreground_bbox():
    mask = np.zeros((50, 60), bool)
    assert foreground_bbox(mask) is None
    mask[10:20, 5:25] = True        # main object, area 200
    mask[45, 58] = True             # speck
    assert foreground_bbox(mask) == (5, 10, 54, 36)
    assert foreground_bbox(mask, min_fraction=0.01) == (5, 10, 20, 10)


# ----------------------------------------------------------------------
# Tissue regions as polygons
# ----------------------------------------------------------------------

from compass.processing.tissue import mask_to_polygons, tissue_regions  # noqa: E402


def _blobs():
    m = np.zeros((100, 120), bool)
    m[10:61, 10:91] = True            # big blob: 51 x 81 pixels
    m[30:45, 40:60] = False           # big hole: 15 x 20
    m[20:23, 20:23] = False           # small hole: 3 x 3
    m[70:81, 100:111] = True          # small blob: 11 x 11
    m[95, 5] = True                   # a single pixel
    return m


def test_mask_to_polygons_edges_holes_and_filters():
    polys = mask_to_polygons(_blobs(), 2.0, 2.0, min_area=1000, hole_area=100)
    assert len(polys) == 1            # the 11x11 blob (484) and the pixel are below min_area
    p = polys[0]
    assert p.bounds == pytest.approx((20, 20, 182, 122))     # pixel edges: [10, 91) x [10, 61), x2
    assert len(p.interiors) == 1                             # small hole (36) filled, big one (1200) kept
    assert p.interiors[0].bounds == pytest.approx((80, 60, 120, 90))
    assert p.area == pytest.approx(2 * 2 * (51 * 81 - 15 * 20), rel=1e-3)   # small hole filled


def test_mask_to_polygons_keeps_everything_when_unfiltered_and_sorts():
    polys = mask_to_polygons(_blobs(), 1.0, 1.0)
    areas = [q.area for q in polys]
    assert areas == sorted(areas, reverse=True)
    assert len(polys) == 2 and len(polys[0].interiors) == 2      # a lone pixel has no contour


def test_mask_to_polygons_simplification_and_empty():
    yy, xx = np.mgrid[:200, :200]
    disk = (yy - 100) ** 2 + (xx - 100) ** 2 < 80 ** 2
    fine = mask_to_polygons(disk)[0]
    coarse = mask_to_polygons(disk, tolerance=3.0)[0]
    assert len(coarse.exterior.coords) < len(fine.exterior.coords) / 3
    assert fine.hausdorff_distance(coarse) <= 3.0 + 1e-6
    assert mask_to_polygons(np.zeros((10, 10), bool)) == []


class _Raster:
    """Duck-typed pyramid: level k is the image subsampled by 2**k, mpp0 = 2 um."""
    def __init__(self, img, nlevels=3, mpp0=2.0):
        self.imgs = [img[::2 ** k, ::2 ** k] for k in range(nlevels)]
        self.nlevels, self.mpp0 = nlevels, mpp0

    def get_mpp_for_level(self, k):
        return self.mpp0 * 2 ** k

    def shape(self, k):
        from compass.core import ImageShape
        h, w = self.imgs[k].shape[:2]
        return ImageShape(width=w, height=h)

    def get_plane(self, k):
        return self.imgs[k]


def test_tissue_regions_on_pyramid(slide):
    img = slide[0]
    r = tissue_regions(_Raster(img), work_mpp=4.0, tolerance_um=4.0, min_area_mm2=0.01)
    assert r["info"]["work_level"] == 1 and r["info"]["work_mpp"] == 4.0
    assert len(r["pieces"]) == 1
    piece = r["pieces"][0]
    # the ellipse: centre (175, 150), radii (100, 75), in level-0 pixels
    x0, y0, x1, y1 = piece["bbox"].bounds
    assert (x0, y0, x1, y1) == pytest.approx((75, 75, 275, 225), abs=8)
    assert piece["outline"].area == pytest.approx(np.pi * 100 * 75, rel=0.08)
    assert piece["area_um2"] == pytest.approx(piece["outline"].area * 4.0)
    assert piece["bbox"].equals(__import__("shapely").box(x0, y0, x1, y1))
    # a high minimum area removes everything
    assert tissue_regions(_Raster(img), work_mpp=4.0, min_area_mm2=10.0)["pieces"] == []
