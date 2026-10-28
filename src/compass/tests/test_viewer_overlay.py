# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from compass.core.annot import AnnotationStore
from compass.core.annot import Point as APoint, Polygon as APolygon, PolyLine
from compass.viewer.overlay import (DEFAULT_PALETTE, assign_colors, geometry_arrays,
                                    hex_to_rgba, load_overlay, load_store_groups,
                                    read_colors)


def test_hex_to_rgba():
    assert hex_to_rgba("#FF0000") == (1.0, 0.0, 0.0, 1.0)
    assert hex_to_rgba("#000000", 0.5)[3] == 0.5
    for bad in ("red", "#12345", "#gg0000", "FF0000"):
        with pytest.raises(ValueError):
            hex_to_rgba(bad)


def test_read_colors_flat_and_group_list(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[colors]\n"Pattern CT" = "#0072B2"\n\n'
                 '[[group]]\nlayer = "x"\nname = "Pattern DE"\ncolor = "#D55E00"\n'
                 '[[group]]\nname = "no colour"\n')
    assert read_colors(f) == {"Pattern CT": "#0072B2", "Pattern DE": "#D55E00"}


def test_read_colors_rejects_bad_colour(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[colors]\nA = "blue"\n')
    with pytest.raises(ValueError, match="#rrggbb"):
        read_colors(f)


def test_assign_colors_keeps_given_and_avoids_them():
    given = {"A": DEFAULT_PALETTE[0]}
    out = assign_colors(["A", "B", "C", "B"], given)
    assert out["A"] == DEFAULT_PALETTE[0]
    assert out["B"] != out["A"] and out["C"] != out["A"] and out["B"] != out["C"]
    assert set(out) == {"A", "B", "C"}


def test_assign_colors_cycles_when_palette_exhausted():
    names = [f"g{i}" for i in range(len(DEFAULT_PALETTE) + 2)]
    out = assign_colors(names)
    assert len(out) == len(names)
    assert out[names[0]] == out[names[len(DEFAULT_PALETTE)]]


def test_geometry_arrays_polygon_with_hole_is_closed_rings():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(2, 2), (4, 2), (4, 4), (2, 4)]
    seg, pts = geometry_arrays([Polygon(shell, [hole])])
    assert seg.shape == (2 * 8, 2) and seg.dtype == np.float32
    assert len(pts) == 0
    # every ring is closed: each segment end is some segment start
    starts, ends = {tuple(r) for r in seg[0::2]}, {tuple(r) for r in seg[1::2]}
    assert starts == ends


def test_geometry_arrays_lines_points_and_multipart():
    seg, pts = geometry_arrays([LineString([(0, 0), (1, 1), (2, 0)]),
                                Point(5, 5), MultiPoint([(6, 6), (7, 7)]),
                                shapely.MultiLineString([[(0, 0), (1, 0)], [(3, 3), (4, 4)]])])
    assert seg.shape == (2 * (2 + 1 + 1), 2)  # open lines: n-1 segments each
    assert pts.tolist() == [[5, 5], [6, 6], [7, 7]]


def test_geometry_arrays_empty_and_missing():
    seg, pts = geometry_arrays([])
    assert seg.shape == (0, 2) and pts.shape == (0, 2)
    seg, pts = geometry_arrays([None, Polygon()])
    assert seg.shape == (0, 2) and pts.shape == (0, 2)


@pytest.fixture
def two_stores(tmp_path):
    a = tmp_path / "S1.morphotypes.ann.sqlite"
    with AnnotationStore(a, mode="w") as st:
        st.add_objects([APolygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                        APolygon([(20, 20), (30, 20), (30, 30)])],
                       layer="morphology", group="Pattern CT")
        st.add_objects([APolygon([(50, 50), (60, 50), (60, 60)])],
                       layer="morphology", group="Pattern DE")
        st.add_objects([PolyLine([(0, 0), (5, 5), (9, 2)])],
                       layer="landmarks", group="Invasive front")
    b = tmp_path / "S1.tissue.ann.sqlite"
    with AnnotationStore(b, mode="w") as st:
        st.add_objects([APolygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
                        APoint((3, 4))], layer="tissue", group="tissue")
    return a, b


def test_load_store_groups(two_stores):
    a, _ = two_stores
    groups, warns = load_store_groups(a)
    assert warns == []
    assert sorted((g.layer, g.name, g.n_objects) for g in groups) == sorted([
        ("morphology", "Pattern CT", 2), ("morphology", "Pattern DE", 1),
        ("landmarks", "Invasive front", 1)])
    assert all(g.source == "S1.morphotypes" for g in groups)
    ct = next(g for g in groups if g.name == "Pattern CT")
    assert ct.segments.shape == (2 * (4 + 3), 2)
    front = next(g for g in groups if g.name == "Invasive front")
    assert front.segments.shape == (2 * 2, 2)
    assert len({g.uid for g in groups}) == len(groups)


def test_default_empty_groups_not_listed(two_stores):
    a, _ = two_stores
    groups, _ = load_store_groups(a)
    assert "no_group" not in {g.name for g in groups}


def test_load_overlay_two_stores_unique_uids_and_shared_colours(two_stores):
    a, b = two_stores
    ov = load_overlay([a, b], colors={"Pattern CT": "#0072B2"})
    assert len({g.uid for g in ov.groups}) == len(ov.groups) == 4
    assert ov.sources() == ["S1.morphotypes", "S1.tissue"]
    ct = next(g for g in ov.groups if g.name == "Pattern CT")
    assert ct.color == hex_to_rgba("#0072B2")
    tissue = next(g for g in ov.groups if g.name == "tissue")
    assert len(tissue.points) == 1 and len(tissue.segments) == 8
    assert len({g.color for g in ov.groups}) == 4
    assert ov.by_uid()[ct.uid] is ct


def test_max_objects_skips_big_group_with_warning(two_stores):
    a, _ = two_stores
    ov = load_overlay([a], max_objects=1)
    names = {g.name for g in ov.groups}
    assert "Pattern CT" not in names and "Pattern DE" in names
    assert any("Pattern CT" in w and "not shown" in w for w in ov.warnings)


def test_store_is_not_modified(two_stores):
    a, _ = two_stores
    before = a.read_bytes()
    load_overlay([a])
    assert a.read_bytes() == before
