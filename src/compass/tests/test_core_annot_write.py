# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""AnnotationStore write path: modes, hierarchy edits, adding / updating /
removing objects, sparse scalars, whole-store transforms."""

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest
import shapely

from compass.core import ImageShape
from compass.core.annot import (AnnotationStore, Circle, Point, PointSet, Polygon,
                                PolyLine)


@pytest.fixture
def store(tmp_path):
    st = AnnotationStore(tmp_path / "w.ann.sqlite", mode="w")
    yield st
    st.close()


def _cells(n, seed=0, x0=0.0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1000, size=(n, 2)) + [x0, 0]
    return gpd.GeoDataFrame({"name": [f"c{i}" for i in range(n)]},
                            geometry=shapely.points(xy))


def test_modes(tmp_path):
    p = tmp_path / "m.sqlite"
    with pytest.raises(ValueError):
        AnnotationStore(p, mode="r+")                 # must exist
    with pytest.raises(ValueError):
        AnnotationStore(":memory:")                   # memory needs "w"
    with AnnotationStore(p, mode="w") as st:
        assert st.get_layer_names() == ["base"]
        assert st.get_groups()["name"].to_list() == ["no_group"]
        st.add_object(Point((1, 1)))
    with AnnotationStore(p) as ro:                    # default read-only
        assert len(ro) == 1 and not ro.writable
        with pytest.raises(PermissionError):
            ro.add_object(Point((2, 2)))
    with AnnotationStore(p, mode="r+") as st:
        st.add_object(Point((2, 2)))
        assert len(st) == 2
    with AnnotationStore(p, mode="w") as st:          # "w" replaces
        assert len(st) == 0


def test_meta_properties(store):
    store.name = "slide"
    store.mpp = 0.25
    store.image_shape = (1000, 800)
    assert store.name == "slide" and store.mpp == 0.25
    assert store.image_shape == ImageShape(width=1000, height=800)
    store.set_meta(scanner="X")
    assert store.meta["scanner"] == "X"


def test_add_objects_instances(store):
    objs = [Point((1, 2), "p", {"CD3": 1.5}), PointSet([(0, 0), (5, 5)]),
            PolyLine([(0, 0), (9, 9)]), Polygon([(0, 0), (4, 0), (4, 4)], "roi"),
            Circle((50, 50), 10, "spot", {"CD3": 2.0, "CD8": 3.0})]
    ids = store.add_objects(objs, layer="mixed", group=["g1", "g2"])
    assert ids.tolist() == [0, 1, 2, 3, 4]
    assert [o.id for o in objs] == ids.tolist()
    lid = store.get_layer_id("mixed")
    assert store.get_groups(lid)["name"].to_list() == ["g1", "g2"]
    assert store.count_objects_per_group()["n_objects"].to_list() == [0, 5, 5]

    gdf = store.get_objects(ids, circles_as_disks=False)
    assert list(gdf["type"]) == ["POINT", "POINTSET", "POLYLINE", "POLYGON", "CIRCLE"]
    assert gdf.loc[4, "circle_r"] == 10 and gdf.geometry.loc[4].geom_type == "Point"
    # circle bbox includes the radius -> found by the R-tree
    assert store.get_object_ids_in_roi(lid, (59, 59, 70, 70)).tolist() == [4]

    back = store.fetch_objects(ids)
    assert isinstance(back[4], Circle) and back[4].radius == 10
    assert back[4].data == {"CD3": 2.0, "CD8": 3.0} and back[0].data == {"CD3": 1.5}
    assert back[1].data == {}


def test_add_objects_frame_with_wide_data(store):
    gdf = _cells(100)
    data = pd.DataFrame({"CD3": np.arange(100, dtype=float),
                         "CD8": np.where(np.arange(100) % 2, np.nan, 1.0)})
    ids = store.add_objects(gdf, layer="cells", group="all", data=data)
    assert ids.size == 100
    wide = store.get_scalar_data(ids[:4], attributes=["CD3", "CD8"]).sort("object_id")
    assert wide["CD3"].to_list() == [0, 1, 2, 3]
    assert wide["CD8"].to_list() == [1.0, None, 1.0, None]
    # polars works too, as does a frame with CIRCLE rows holding disks
    circ = gpd.GeoDataFrame({"type": ["CIRCLE"], "circle_r": [3.0]},
                            geometry=[shapely.Point(7, 7).buffer(3.0)])
    (cid,) = store.add_objects(circ, layer="cells", group="all",
                               data=pl.DataFrame({"CD3": [9.0]}))
    got = store.get_objects([cid], circles_as_disks=False)
    assert got.geometry.iloc[0].coords[0] == pytest.approx((7, 7))
    assert store.get_scalar_dicts([cid]) == {int(cid): {"CD3": 9.0}}


def test_groups_unique_per_layer(store):
    store.add_objects([Point((0, 0))], layer="A", group="tumor")
    store.add_objects([Point((1, 1))], layer="B", group="tumor")   # legacy: unsaveable
    with pytest.raises(ValueError, match="several layers"):
        store.add_to_group([0], "tumor")
    assert store.get_group_id("tumor", store.get_layer_id("B")) != \
        store.get_group_id("tumor", store.get_layer_id("A"))


def test_cross_layer_membership_rejected(store):
    (a,) = store.add_objects([Point((0, 0))], layer="A", group="g")
    store.add_group("h", layer="B")
    with pytest.raises(ValueError, match="multiple layers"):
        store.add_to_group([a], "h", layer="B")


def test_remove_semantics(store):
    ids = store.add_objects(_cells(10), layer="cells", group="g1")
    store.add_to_group(ids[:3], "g2", layer="cells")
    store.set_scalar_data(ids, [{"a": 1.0}] * 10)
    # removing g1 deletes the 7 objects that are only in g1
    assert store.remove_group("g1", layer="cells") == 7
    assert len(store) == 3
    assert store.get_scalar_data_long(ids)["object_id"].unique().sort().to_list() \
        == ids[:3].tolist()
    # removing the last membership deletes the object
    assert store.remove_from_group(ids[:1], "g2", layer="cells") == 1
    assert store.remove_objects(ids[1:2]) == 1
    assert store.get_object_ids_in_layer(store.get_layer_id("cells")).tolist() == [ids[2]]
    assert store.remove_layer("cells") == 1
    assert len(store) == 0 and store.get_layer_names() == ["base"]
    # R-tree emptied by the triggers
    assert store._conn.execute("SELECT COUNT(*) FROM objects_rtree").fetchone()[0] == 0


def test_ids_never_reused(store):
    ids = store.add_objects(_cells(5))
    store.remove_objects(ids[-2:])
    new = store.add_objects(_cells(1))
    assert new[0] == ids[-1] + 1
    gid = store.add_group("x")
    store.remove_group("x")
    assert store.add_group("y") == gid + 1


def test_batch_is_atomic(store):
    with pytest.raises(RuntimeError):
        with store.batch():
            store.add_objects(_cells(5), layer="L")
            raise RuntimeError("boom")
    assert len(store) == 0 and store.get_layer_names() == ["base"]
    with store.batch():
        store.add_objects(_cells(5), layer="L")
        store.add_objects(_cells(5), layer="L")
    assert len(store) == 10


def test_rename(store):
    store.add_objects([Point((0, 0))], layer="A", group="g")
    store.rename_layer("A", "AA")
    store.rename_group("g", "gg", layer="AA")
    assert store.get_groups(store.get_layer_id("AA"))["name"].to_list() == ["gg"]


def test_update_objects(store):
    ids = store.add_objects([Point((0, 0)), Circle((5, 5), 1, "c")], layer="L", group="g")
    lid = store.get_layer_id("L")
    upd = gpd.GeoDataFrame({"name": ["moved", "c2"], "circle_r": [np.nan, 4.0]},
                           geometry=[shapely.Point(100, 100), shapely.Point(50, 50)],
                           index=pd.Index(ids, name="object_id"))
    store.update_objects(upd)
    got = store.get_objects(ids, circles_as_disks=False)
    assert list(got["name"]) == ["moved", "c2"]
    assert list(got["type"]) == ["POINT", "CIRCLE"] and got["circle_r"].iloc[1] == 4
    assert store.get_object_ids_in_roi(lid, (99, 99, 101, 101)).tolist() == [ids[0]]
    assert store.get_object_ids_in_roi(lid, (0, 0, 10, 10)).size == 0
    with pytest.raises(ValueError, match="no objects"):
        store.update_objects(gpd.GeoSeries([shapely.Point(0, 0)], index=[999]))


def test_scalar_merge_replace_delete(store, caplog):
    (oid,) = store.add_objects([Point((0, 0), data={"a": 1.0, "b": 2.0})])
    store.set_scalar_data([oid], [{"b": 5.0, "c": 6.0}])
    assert store.get_scalar_dicts([oid])[oid] == {"a": 1.0, "b": 5.0, "c": 6.0}
    store.set_scalar_data([oid], [{"z": 1.0, "bad": "text"}], replace=True)
    assert store.get_scalar_dicts([oid])[oid] == {"z": 1.0}
    assert "not numeric" in caplog.text
    store.delete_scalar_data([oid], attributes=["z"])
    assert store.get_scalar_dicts([oid]) == {}
    with pytest.raises(ValueError, match="do not exist"):
        store.set_scalar_data([12345], [{"a": 1.0}])


def test_store_transforms(store):
    store.mpp = 0.5
    store.image_shape = (1000, 500)
    pid, cid = store.add_objects([Point((10, 20)), Circle((100, 100), 5)])
    store.translate(5, 5)
    store.scale(2, origin=(0, 0))
    got = store.get_objects([pid, cid], circles_as_disks=False)
    assert got.geometry.iloc[0].coords[0] == (30, 50)
    assert got.geometry.iloc[1].coords[0] == (210, 210) and got["circle_r"].iloc[1] == 10
    bb = store.get_bboxes().to_pandas().set_index("object_id")
    assert tuple(bb.loc[cid]) == (200, 200, 220, 220)       # R-tree follows

    with pytest.raises(ValueError, match="similarity"):
        store.scale(2, 1)
    store.scale(2, 1, object_ids=[pid])                       # no circle among them
    store.affine([[1, 0, 0], [0, 3, 0]], circles="to_polygon")
    assert list(store.get_objects([cid])["type"]) == ["POLYGON"]

    store.resize(0.5)
    assert store.mpp == 1.0 and store.image_shape == ImageShape(width=500, height=250)
    store.set_resolution(0.25)
    assert store.mpp == 0.25 and store.image_shape.width == 2000


def test_match_image(store):
    class Raster:  # duck-typed PyramidalRaster
        def shape(self, level):
            return ImageShape(width=4000, height=2002)

        def get_mpp_for_level(self, level):
            return 0.25

    store.mpp = 1.0
    store.image_shape = (1000, 500)
    (pid,) = store.add_objects([Point((100, 100))])
    store.match_image(Raster(), 0)
    assert store.image_shape == ImageShape(width=4000, height=2002) and store.mpp == 0.25
    x, y = store.get_objects([pid]).geometry.iloc[0].coords[0]
    assert x == pytest.approx(100 * 0.5 * (4 + 2002 / 500))


def test_memory_store_save_and_load(tmp_path):
    mem = AnnotationStore(":memory:", mode="w")
    mem.add_objects(_cells(20), layer="cells", group="g",
                    data=[{"a": float(i)} for i in range(20)])
    out = mem.save_as(tmp_path / "saved.sqlite")
    with pytest.raises(FileExistsError):
        mem.save_as(out, overwrite=False)
    with AnnotationStore(out) as ro:
        assert len(ro) == 20 and ro.get_scalar_dicts([5]) == {5: {"a": 5.0}}
    copy = AnnotationStore.load_in_memory(out)
    copy.remove_objects([0])
    with AnnotationStore(out) as ro:
        assert len(ro) == 20                                  # file untouched
    assert len(copy) == 19


def test_read_only_reader_sees_writer(tmp_path):
    p = tmp_path / "wal.sqlite"
    w = AnnotationStore(p, mode="w")
    w.add_objects(_cells(10))
    r = AnnotationStore(p)                                    # WAL: concurrent read
    assert len(r) == 10
    w.add_objects(_cells(5))
    assert len(r) == 15
    r.close()
    w.close()


def test_bulk_add_speed(tmp_path):
    n = 200_000
    gdf = _cells(n)
    with AnnotationStore(tmp_path / "big.sqlite", mode="w") as st:
        t = time.perf_counter()
        ids = st.add_objects(gdf, layer="cells", group="all",
                             data=pd.DataFrame({"CD3": np.ones(n)}))
        dt = time.perf_counter() - t
        assert ids.size == n
        assert st.count_objects_in_roi(st.get_layer_id("cells"), (0, 0, 1000, 1000)) == n
        t = time.perf_counter()
        st.translate(1.0)
        dt_tr = time.perf_counter() - t
    print(f"\nadd {n} points + scalars: {dt:.2f}s; translate: {dt_tr:.2f}s")
    assert dt < 30 and dt_tr < 30
