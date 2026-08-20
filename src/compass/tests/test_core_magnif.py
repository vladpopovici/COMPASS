# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

import pytest

from compass.core import Magnification


@pytest.fixture
def mag() -> Magnification:
    # 40x objective, 0.25 mpp at level 0, 6 levels, halving per level
    return Magnification(40.0, 0.25, level=0, n_levels=6, magnif_step=2.0)


def test_basic_properties(mag):
    assert mag.base_magnif == 40.0
    assert mag.base_mpp == 0.25
    assert mag.nlevels == 6
    assert mag.magnif_step == 2.0


def test_level_conversions(mag):
    assert mag.get_magnif_for_level(0) == 40.0
    assert mag.get_magnif_for_level(2) == 10.0
    assert mag.get_mpp_for_level(0) == 0.25
    assert mag.get_mpp_for_level(3) == 2.0


def test_lookup_by_mpp_and_magnif(mag):
    assert mag.get_level_for_mpp(0.25) == 0
    assert mag.get_level_for_mpp(1.0) == 2
    assert mag.get_level_for_magnif(5.0) == 3
    assert mag.get_magnif_for_mpp(0.5) == 20.0
    assert mag.get_mpp_for_magnif(10.0) == 1.0


def test_out_of_range_raises(mag):
    with pytest.raises(RuntimeError):
        mag.get_level_for_mpp(0.1)  # finer than level 0 by >10%
    with pytest.raises(RuntimeError):
        mag.get_mpp_for_magnif(80.0)
    with pytest.raises(RuntimeError):
        mag.get_mpp_for_level(6)


def test_close_to_range_end_clamps(mag):
    # within 10% below the base mpp -> clamps to level 0
    assert mag.get_level_for_mpp(0.24) == 0
    assert mag.get_magnif_for_mpp(0.24) == 40.0


def test_non_base_level():
    # same pyramid described relative to level 1 (20x, 0.5 mpp)
    m = Magnification(20.0, 0.5, level=1, n_levels=6, magnif_step=2.0)
    assert m.get_magnif_for_level(0) == 40.0
    assert m.get_mpp_for_level(0) == 0.25
    assert m.get_level_for_magnif(20.0) == 1
