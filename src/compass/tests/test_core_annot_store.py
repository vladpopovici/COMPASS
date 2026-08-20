# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Tests for compass.core.AnnotationStore, the vectorized read layer over the
SQLite/R-tree annotation schema.

The schema DDL in compass.core.annot must stay byte-identical to the legacy
writer's (compass.legacy.annot_sqlite_storage.SQLITE_DDL) — checked here
textually against the legacy source file, without importing it (the legacy
import chain drags in pyvips/OpenSlide, which core tests must not need).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import shapely
import shapely.geometry as shg
import shapely.wkb as shapely_wkb

from compass.core import ANNOT_TYPE_CODE, SQLITE_DDL, AnnotationStore, init_db


def test_ddl_matches_legacy_writer():
    legacy_src = (Path(__file__).parent.parent / "legacy" /
                  "annot_sqlite_storage.py").read_text()
    assert SQLITE_DDL in legacy_src, \
        "core SQLITE_DDL diverged from compass.legacy.annot_sqlite_storage"


def _insert_object(conn, oid, type_str, geom, name=None, circle_r=None,
                   scalars=None, attr_ids=None):
    if type_str == "CIRCLE":
        bbox = (geom.x - circle_r, geom.y - circle_r, geom.x + circle_r, geom.y + circle_r)
    else:
        bbox = geom.bounds
    conn.execute(
        "INSERT INTO objects(object_id, type_code, name, wkb, circle_r,"
        " min_x, min_y, max_x, max_y) VALUES (?,?,?,?,?,?,?,?,?)",
        (oid, ANNOT_TYPE_CODE[type_str], name,
         sqlite3.Binary(shapely_wkb.dumps(geom)), circle_r, *bbox))
    if scalars is not None:
        ids = np.asarray([attr_ids[k] for k in scalars], dtype=np.uint32)
        vals = np.asarray(list(scalars.values()), dtype=np.float32)
        order = np.argsort(ids)
        conn.execute(
            "INSERT INTO object_scalar_sparse(object_id, ids_u32, vals_f32) VALUES (?,?,?)",
            (oid, sqlite3.Binary(ids[order].tobytes()),
             sqlite3.Binary(vals[order].tobytes())))


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("annot") / "slide.ann.sqlite"
    init_db(path)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    with conn:
        for k, v in [("name", "test-slide"), ("image_width", "10000"),
                     ("image_height", "8000"), ("mpp", "0.25")]:
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (k, v))

        conn.executemany("INSERT INTO layers(layer_id, name) VALUES (?,?)",
                         [(0, "base"), (1, "cells")])
        conn.executemany("INSERT INTO groups(group_id, layer_id, name) VALUES (?,?,?)",
                         [(0, 0, "no_group"), (1, 1, "tumor"), (2, 1, "stroma")])

        conn.executemany("INSERT INTO scalar_attributes(attr_id, attr_name) VALUES (?,?)",
                         [(1, "CD3"), (2, "CD8"), (3, "area")])
        attr_ids = {"CD3": 1, "CD8": 2, "area": 3}

        # layer 1 ("cells"): points in two groups + one circle
        _insert_object(conn, 1, "POINT", shg.Point(100, 100), name="c1",
                       scalars={"CD3": 1.5, "area": 12.0}, attr_ids=attr_ids)
        _insert_object(conn, 2, "POINT", shg.Point(500, 500), name="c2",
                       scalars={"CD8": 0.7}, attr_ids=attr_ids)
        _insert_object(conn, 3, "CIRCLE", shg.Point(1000, 1000), circle_r=50.0,
                       scalars=None)
        # layer 0 ("base"): one big region polygon
        _insert_object(conn, 4, "POLYGON",
                       shg.Polygon([(0, 0), (2000, 0), (2000, 2000), (0, 2000)]),
                       name="region")

        conn.executemany("INSERT INTO object_groups(object_id, group_id) VALUES (?,?)",
                         [(1, 1), (2, 2), (3, 1), (4, 0)])
    conn.close()
    return path


@pytest.fixture
def store(db_path):
    with AnnotationStore(db_path) as s:
        yield s


def test_open_missing_db_raises(tmp_path):
    with pytest.raises(ValueError):
        AnnotationStore(tmp_path / "nope.sqlite")


def test_meta_and_hierarchy(store):
    assert store.meta["name"] == "test-slide"
    assert store.meta["image_width"] == "10000"

    layers = store.get_layers()
    assert layers["name"].to_list() == ["base", "cells"]
    assert store.get_layer_id("cells") == 1
    with pytest.raises(ValueError):
        store.get_layer_id("nonexistent")

    groups = store.get_groups(layer_id=1)
    assert groups["name"].to_list() == ["tumor", "stroma"]
    assert store.get_group_id("stroma", layer_id=1) == 2


def test_counts(store):
    counts = store.count_objects_per_group()
    by_name = dict(zip(counts["group_name"].to_list(), counts["n_objects"].to_list()))
    assert by_name == {"no_group": 1, "tumor": 2, "stroma": 1}

    assert store.count_objects_in_roi(1, (0, 0, 600, 600)) == 2
    assert store.count_objects_in_roi(1, (0, 0, 5000, 5000)) == 3
    assert store.count_objects_in_roi(1, (2000, 2000, 3000, 3000)) == 0


def test_id_queries(store):
    np.testing.assert_array_equal(store.get_object_ids_in_layer(1), [1, 2, 3])
    np.testing.assert_array_equal(store.get_object_ids_in_group(2), [2])
    # ROI touches c1 and c2 but not the circle at (1000,1000)+-50
    np.testing.assert_array_equal(
        store.get_object_ids_in_roi(1, (0, 0, 600, 600)), [1, 2])
    # circle enters via its bbox
    np.testing.assert_array_equal(
        store.get_object_ids_in_roi(1, (940, 940, 960, 960)), [3])
    # group filter
    np.testing.assert_array_equal(
        store.get_object_ids_in_roi(1, (0, 0, 5000, 5000), group_ids=[2]), [2])


def test_get_objects_vectorized(store):
    gdf = store.get_objects([1, 2, 3, 4])
    assert list(gdf.index) == [1, 2, 3, 4]
    assert gdf.loc[1, "type"] == "POINT"
    assert gdf.loc[1, "name"] == "c1"
    assert gdf.loc[1].geometry.equals(shg.Point(100, 100))
    # circle decoded to a polygonal disk by default
    assert gdf.loc[3, "type"] == "CIRCLE"
    assert gdf.loc[3, "circle_r"] == 50.0
    assert gdf.loc[3].geometry.geom_type == "Polygon"
    assert abs(gdf.loc[3].geometry.area - np.pi * 50 ** 2) / (np.pi * 50 ** 2) < 0.01
    # ... or kept as its center point
    raw = store.get_objects([3], circles_as_disks=False)
    assert raw.loc[3].geometry.equals(shg.Point(1000, 1000))


def test_get_objects_empty(store):
    gdf = store.get_objects([])
    assert len(gdf) == 0
    assert set(gdf.columns) >= {"type", "name", "circle_r", "geometry"}


def test_get_objects_in_roi(store):
    gdf = store.get_objects_in_roi(1, (0, 0, 600, 600))
    assert list(gdf.index) == [1, 2]
    assert shapely.get_coordinates(gdf.geometry.values).shape == (2, 2)
    # nothing in an empty corner
    assert len(store.get_objects_in_roi(1, (5000, 5000, 6000, 6000))) == 0


def test_bboxes_and_memberships(store):
    bb = store.get_bboxes(layer_id=1)
    assert bb["object_id"].to_list() == [1, 2, 3]
    circle_row = bb.filter(bb["object_id"] == 3)
    assert circle_row["min_x"][0] == 950.0 and circle_row["max_x"][0] == 1050.0

    mem = store.get_memberships([1, 3])
    assert mem["object_id"].to_list() == [1, 3]
    assert mem["group_id"].to_list() == [1, 1]


def test_scalar_data(store):
    long = store.get_scalar_data_long([1, 2, 3])
    assert long.height == 3  # CD3+area for obj 1, CD8 for obj 2, none for 3

    wide = store.get_scalar_data([1, 2, 3])
    assert wide.height == 2  # object 3 has no scalars
    row1 = wide.filter(wide["object_id"] == 1)
    assert row1["CD3"][0] == pytest.approx(1.5)
    assert row1["area"][0] == pytest.approx(12.0)
    assert row1["CD8"][0] is None

    only_cd8 = store.get_scalar_data([1, 2, 3], attributes=["CD8"])
    assert only_cd8.columns == ["object_id", "CD8"]

    names = store.get_scalar_attribute_names()
    assert names["attr_name"].to_list() == ["CD3", "CD8", "area"]


def test_store_is_read_only(db_path):
    with AnnotationStore(db_path) as s, pytest.raises(sqlite3.OperationalError):
        s._conn.execute("DELETE FROM objects")
