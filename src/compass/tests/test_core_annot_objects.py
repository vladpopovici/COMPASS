# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Annotation object wrappers (compass.core.annot.objects) and the
vectorized GeoDataFrame transforms (compass.core.annot.transform)."""

import geopandas as gpd
import numpy as np
import pytest
import shapely

from compass.core.annot import (Circle, Point, PointSet, Polygon, PolyLine,
                                create_empty_annotation_object, gdf_to_objects,
                                make_annotation_object, objects_to_gdf, transform)


def test_construction_defaults_and_data():
    p = Point((3, 4), data={"a": 1})
    assert p.type == "POINT" and p.name == "POINT" and p.id == -1
    assert p.data == {"a": 1}
    assert Point((0, 0)).data == {}           # legacy bug: data was None
    assert Polygon([(0, 0), (1, 0), (1, 1)]).name == "POLYGON"   # legacy: None
    assert PointSet([(0, 0), (1, 1)]).name == "POINTS"
    assert PolyLine([(0, 0), (1, 1)]).name == "POLYLINE"
    assert Circle((0, 0), 2).name == "CIRCLE"


def test_coordinates_and_size():
    ps = PointSet([(0, 0), (1, 2), (3, 4)])
    np.testing.assert_array_equal(ps.x, [0, 1, 3])
    np.testing.assert_array_equal(ps.y, [0, 2, 4])
    assert ps.size() == 3 and ps.xy().shape == (3, 2)
    assert Point((1, 2)).size() == 1


def test_accepts_shapely_geometry():
    holed = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                            holes=[[(2, 2), (4, 2), (4, 4)]])
    assert Polygon(holed).geom.equals(holed)
    with pytest.raises(ValueError):
        Polygon(shapely.Point(0, 0))


def test_transforms():
    pg = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    pg.translate(1)                       # y_off defaults to x_off
    assert pg.bounding_box() == (1, 1, 3, 3)
    pg.scale(2)                           # about its center (2, 2)
    assert pg.bounding_box() == (0, 0, 4, 4)
    pg.resize(0.5)                        # about (0, 0)
    assert pg.bounding_box() == (0, 0, 2, 2)
    pg.affine([[0, -1, 10], [1, 0, 0]])   # rotate 90 deg + shift
    assert pg.bounding_box() == pytest.approx((8, 0, 10, 2))


def test_duplicate_is_independent():
    p = Point((1, 1), name="p", data={"a": 1.0})
    p._id = 7
    q = p.duplicate()                     # legacy: crashed (self.group)
    q.translate(5, 5)
    q.data["a"] = 2.0
    assert p.xy().tolist() == [[1, 1]] and p.data == {"a": 1.0}
    assert q.id == -1 and q.name == "p"


def test_circle_exact_center_radius():
    c = Circle((10, 20), 5)
    assert c.size() == 65                  # 64-vertex disk (closed ring)
    assert c.bounding_box() == (5, 15, 15, 25)
    c.translate(1)                         # legacy: crashed with y_off=None
    assert c.center == (11, 21)
    c.scale(2)                             # about its center
    assert c.center == (11, 21) and c.radius == 10
    c.resize(0.5)                          # about (0, 0)
    assert c.center == (5.5, 10.5) and c.radius == 5
    c.affine([[0, -2, 0], [2, 0, 0]])      # similarity (rotation, scale 2)
    assert c.center == pytest.approx((-21, 11)) and c.radius == pytest.approx(10)


def test_circle_rejects_deforming_transforms():
    c = Circle((0, 0), 1)
    with pytest.raises(ValueError):
        c.scale(2, 3)
    with pytest.raises(ValueError):
        c.affine([[2, 0, 0], [0, 1, 0]])
    pg = c.to_polygon()
    pg.affine([[2, 0, 0], [0, 1, 0]])
    assert pg.type == "POLYGON" and pg.bounding_box() == pytest.approx((-2, -1, 2, 1))


def test_asdict_roundtrip():
    for obj in [Point((1, 2), "p", {"a": 1.0}), PointSet([(0, 0), (1, 1)]),
                PolyLine([(0, 0), (5, 5)]), Polygon([(0, 0), (1, 0), (1, 1)]),
                Circle((3, 4), 2.5, "c", {"b": 2.0})]:
        d = obj.asdict()
        back = create_empty_annotation_object(d["annotation_type"])
        back.fromdict(d)
        assert back.type == obj.type and back.name == obj.name and back.data == obj.data
        assert back.geom.equals_exact(obj.geom, 1e-9)
    c = create_empty_annotation_object("CIRCLE")
    c.fromdict(Circle((3, 4), 2.5).asdict())   # legacy: lost center/radius
    assert c.center == (3, 4) and c.radius == 2.5


def test_make_annotation_object_and_frames():
    c = make_annotation_object("CIRCLE", shapely.Point(1, 1), circle_r=2.0)
    assert isinstance(c, Circle) and c.radius == 2.0
    disk = shapely.Point(5, 5).buffer(3)
    c2 = make_annotation_object("CIRCLE", disk)
    assert c2.center == pytest.approx((5, 5)) and c2.radius == pytest.approx(3)

    objs = [Point((0, 0), "a"), Circle((5, 5), 1, "b")]
    gdf = objects_to_gdf(objs)
    assert list(gdf["type"]) == ["POINT", "CIRCLE"]
    assert np.isnan(gdf["circle_r"].iloc[0]) and gdf["circle_r"].iloc[1] == 1
    back = gdf_to_objects(gdf, data={1: {"x": 1.0}})
    assert [o.name for o in back] == ["a", "b"] and back[1].data == {"x": 1.0}
    assert isinstance(back[1], Circle) and back[1].center == pytest.approx((5, 5))


def _frame():
    return objects_to_gdf([Point((1, 1)), Polygon([(0, 0), (2, 0), (2, 2)]),
                           Circle((10, 10), 2)])


def test_frame_translate_scale_affine():
    gdf = _frame()
    t = transform.translate(gdf, 5, -1)
    assert t.geometry.iloc[0].coords[0] == (6, 0)
    assert gdf.geometry.iloc[0].coords[0] == (1, 1)         # input untouched

    s = transform.scale(gdf, 2)
    assert s.geometry.iloc[0].coords[0] == (2, 2)
    assert s["circle_r"].iloc[2] == 4
    assert s.geometry.iloc[2].centroid.coords[0] == pytest.approx((20, 20))

    own = transform.scale(gdf, 2, origin="center")
    assert own.geometry.iloc[0].coords[0] == (1, 1)
    assert own.geometry.iloc[1].bounds == (-1, -1, 3, 3)

    r = transform.affine(gdf, [[0, -1, 0], [1, 0, 0]])
    assert r.geometry.iloc[0].coords[0] == pytest.approx((-1, 1))
    assert r["circle_r"].iloc[2] == pytest.approx(2)


def test_frame_non_similarity_circles():
    gdf = _frame()
    with pytest.raises(ValueError, match="similarity"):
        transform.scale(gdf, 2, 1)
    out = transform.scale(gdf, 2, 1, circles="to_polygon")
    assert list(out["type"]) == ["POINT", "POLYGON", "POLYGON"]
    assert np.isnan(out["circle_r"].iloc[2])
    assert out.geometry.iloc[2].bounds == pytest.approx((16, 8, 24, 12))
    # no circles: any affine is fine
    pts = gpd.GeoDataFrame(geometry=[shapely.Point(1, 1)])
    assert transform.scale(pts, 2, 3).geometry.iloc[0].coords[0] == (2, 3)


def test_import_record_matrix_convention():
    # crop at (100, 50), downsample 2: matrix maps source level-0 -> output
    M = [[0.5, 0.0, -50.0], [0.0, 0.5, -25.0]]
    gdf = gpd.GeoDataFrame(geometry=[shapely.Point(300, 250)])
    assert transform.affine(gdf, M).geometry.iloc[0].coords[0] == (100, 100)
    assert transform.affine(gdf, M + [[0, 0, 1]]).geometry.iloc[0].coords[0] == (100, 100)
