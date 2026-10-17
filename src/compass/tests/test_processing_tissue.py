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
