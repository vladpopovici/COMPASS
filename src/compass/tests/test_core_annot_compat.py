# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""The legacy annotation API on top of AnnotationStore
(compass.core.annot.compat), and file interchange with the legacy code."""

import numpy as np
import pytest

from compass.core import ImageShape
from compass.core import annot as ann


def _example():
    a = ann.Annotation(name="Annotation 1", image_shape=ImageShape(width=1000, height=800),
                       mpp=2.0)
    pt1 = ann.Point((200, 200), name="PT1", data={"a": 5, "b": 10})
    pt2 = ann.Point((200, 600), name="PT2", data={"b": 12.5, "c": 20.5})
    pt3 = ann.Point((600, 500), name="PT3")
    pl1 = ann.Polygon([(100, 100), (100, 500), (500, 500), (500, 100)], name="Poly1",
                      data={"e": 10.5})
    cr1 = ann.Circle((200, 400), 50, name="Circ1", data={"a": 4.5, "d": 2})
    a.add_annotation_object(pt1, group="no_group", layer="base")
    a.add_annotation_object(pt2, group="group one", layer="base")
    a.add_annotations([pt3, pl1, cr1], group="group two", layer="new layer")
    return a


def test_annotation_build_and_scan():
    a = _example()
    assert a.name == "Annotation 1" and a.get_resolution() == 2.0
    assert a.get_layer_names() == ["base", "new layer"]
    assert [o.name for o in a.scan_objects("base")] == ["PT1"]
    assert [o.name for o in a.scan_objects("base", group=None)] == ["PT1", "PT2"]
    assert [o.name for o in a.scan_objects("new layer", "group two", "POINT")] == ["PT3"]
    assert [o.name for o in a.scan_objects("new layer", None, 5)] == ["Circ1"]
    (c,) = a.scan_objects("new layer", None, "CIRCLE")
    assert c.center == (200, 400) and c.radius == 50 and c.data == {"a": 4.5, "d": 2.0}
    # vectorized access to the same content
    assert len(a.store.get_objects(a.store.get_object_ids_in_layer(1))) == 3


def test_annotation_resize_and_resolution():
    a = _example()
    a.set_resolution(0.5)                        # 4x finer
    assert a.get_resolution() == 0.5
    assert a.get_base_image_shape() == ImageShape(width=4000, height=3200)
    (c,) = a.scan_objects("new layer", None, "CIRCLE")
    assert c.center == (800, 1600) and c.radius == 200


def test_asdict_fromdict_roundtrip():
    a = _example()
    d = a.asdict()
    assert d["id_layer"] == {0: "base", 1: "new layer"}
    assert d["id_reg"]["object"] == 4
    b = ann.Annotation()
    b.fromdict(d)
    assert b.asdict() == d


def test_save_load_and_path_functions(tmp_path):
    p = tmp_path / "a.sqlite"
    a = _example()
    ann.save_annotation(a, p)
    b = ann.load_annotation(p)
    assert b.asdict() == a.asdict()

    lid = ann.query_layer_id(p, "new layer")
    gid = ann.query_group_id(p, "group two", lid)
    assert ann.query_groups_in_layer(p, lid) == [(gid, "group two")]
    assert ann.query_object_ids_in_layer(p, lid) == [2, 3, 4]
    assert ann.query_object_ids_in_group(p, gid) == [2, 3, 4]
    assert ann.query_object_ids_in_layer_roi(p, lid, (140, 340, 160, 360)) == [3, 4]
    objs = ann.fetch_objects(p, [0, 4])
    assert [o.name for o in objs] == ["PT1", "Circ1"] and objs[0].data == {"a": 5, "b": 10}
    assert ann.fetch_geometries(p, [0])[0].data == {}
    assert ann.fetch_scalar_data(p, [0, 2]) == {0: {"a": 5.0, "b": 10.0}}
    assert ann.fetch_scalar_data(p, []) == {}      # legacy returned []


def test_objects_are_copied_in():
    a = ann.Annotation()
    p = ann.Point((1, 1))
    a.add_annotation_object(p)
    p.translate(10)
    (q,) = a.scan_objects()
    assert q.xy().tolist() == [[1, 1]] and q.id == p.id == 0


# -- interchange with compass.legacy (needs its system libs: pyvips, ...) -----

@pytest.fixture
def legacy():
    return pytest.importorskip("compass.legacy.annot_sqlite_storage",
                               exc_type=ImportError)


def test_legacy_file_readable(tmp_path, legacy):
    import compass.legacy.annot as lann
    from compass.legacy.core import ImageShape as LShape

    la = lann.Annotation(name="L", image_shape=LShape(width=100, height=100), mpp=0.5)
    la.add_annotation_object(lann.Point((10, 10), name="p", data={"g": 1.0}))
    la.add_annotations([lann.Circle((50, 50), 5, name="c", data={"g": 2.0})],
                       group="spots", layer="visium")
    p = tmp_path / "legacy.sqlite"
    legacy.save_annotation(la, p)

    b = ann.load_annotation(p)
    assert b.get_layer_names() == ["base", "visium"]
    (c,) = b.scan_objects("visium", "spots")
    assert c.center == (50, 50) and c.radius == 5 and c.data == {"g": 2.0}
    # and it can be extended without id clashes
    b.add_annotation_object(ann.Point((1, 1)))
    assert b.store.get_object_ids_in_layer(0).tolist() == [0, 2]


def test_new_file_readable_by_legacy(tmp_path, legacy):
    p = tmp_path / "new.sqlite"
    _example().save(p)
    la = legacy.load_annotation(p)
    d = la.asdict()
    assert d["id_layer"] == {0: "base", 1: "new layer"}
    circ = [o for o in la._annots[2] if o.type == "CIRCLE"][0]
    assert circ.center == (200, 400) and circ.radius == 50
    assert np.isclose(circ.data["a"], 4.5)
