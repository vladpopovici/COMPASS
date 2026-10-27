# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Tests for compass.connector.asap: ASAP XML -> AnnotationStore, mapped onto
a (fake-slide) imported pyramid through its import record."""

import numpy as np
import pytest

from compass.connector import AsapTarget, asap_to_store, read_asap_xml, wsi2zarr
from compass.core import ImageShape, ZarrRaster
from compass.core.annot import AnnotationStore
from compass.tests.test_connector_ingest import FakeVipsImage, FakeWSI, fake_slide  # noqa: F401


def xml(items, groups=("Tumor", "Pattern CT", "Orientation lumen", "Invasive front",
                        "Three points")):
    """items: (type, group, [(x, y), ...] or [(order, x, y), ...])"""
    out = ['<?xml version="1.0"?>', "<ASAP_Annotations>", "<Annotations>"]
    for k, (t, g, pts) in enumerate(items):
        out.append(f'<Annotation Name="Annotation {k}" Type="{t}" PartOfGroup="{g}" Color="#F4FA58"><Coordinates>')
        for i, p in enumerate(pts):
            o, x, y = p if len(p) == 3 else (i, *p)
            out.append(f'<Coordinate Order="{o}" X="{x}" Y="{y}" />')
        out.append("</Coordinates></Annotation>")
    out += ["</Annotations>", "<AnnotationGroups>"]
    out += [f'<Group Name="{g}" PartOfGroup="None" Color="#000000"><Attributes /></Group>' for g in groups]
    out += ["</AnnotationGroups>", "</ASAP_Annotations>"]
    return "\n".join(out)


GROUP_MAP = {
    "Tumor": ("tissue", "Tumor"),
    "Pattern CT": ("morphology", "Pattern CT"),
    "Orientation lumen": ("landmarks", "Orientation lumen"),
    "Invasive front": AsapTarget("landmarks", "Invasive front", as_type="POLYLINE"),
    "Three points": None,
}
VOCAB = [("morphology", "Pattern CT"), ("morphology", "Pattern DE"), ("tissue", "Tumor"),
         ("landmarks", "Orientation lumen"), ("landmarks", "Invasive front")]


@pytest.fixture
def raster(fake_slide, tmp_path):  # noqa: F811
    slide_path, _ = fake_slide
    dst = tmp_path / "s.zarr"
    wsi2zarr(slide_path, dst, crop=(9, 17, 1400, 1100), start_level=1,
             min_size=256, chunk_size=64, shard_factor=2)
    return ZarrRaster(dst)


def src(x_out, y_out):
    """source level-0 coordinates of an output pixel (crop snapped to (8, 16), x2)."""
    return 8 + 2 * x_out, 16 + 2 * y_out


def test_import_maps_onto_pyramid(raster, tmp_path):
    f = tmp_path / "a.xml"
    f.write_text(xml([
        ("Polygon", "Tumor", [src(100, 100), src(300, 100), src(300, 250), src(100, 250)]),
        ("Polygon", "Pattern CT", [src(120, 120), src(200, 120), src(200, 200)]),
        ("Dot", "Orientation lumen", [src(37, 11)]),
        ("PointSet", "Invasive front", [src(10, 10), src(50, 60), src(90, 70)]),
        ("Dot", "Three points", [src(1, 1)]),
    ]))
    out = tmp_path / "a.ann.sqlite"
    rep = asap_to_store(f, out, raster, group_map=GROUP_MAP, groups=VOCAB,
                        data={"approx": 1.0}, meta={"annotation_quality": "approximate"})
    assert rep["n_objects"] == 4
    assert rep["dropped"] == {"Three points": 1}
    assert rep["per_group"] == {"tissue/Tumor": 1, "morphology/Pattern CT": 1,
                                "landmarks/Orientation lumen": 1, "landmarks/Invasive front": 1}
    assert not rep["out_of_bounds"] and not rep["repaired"] and not rep["skipped"]

    with AnnotationStore(str(out)) as st:
        assert st.image_shape == raster.shape(0)
        assert st.mpp == raster.get_mpp_for_level(0)
        assert st.meta["annotation_quality"] == "approximate"
        assert st.meta["annotation_source"] == "a.xml"
        # vocabulary groups exist even when empty
        names = {r["group_name"]: r["n_objects"] for r in st.count_objects_per_group().iter_rows(named=True)}
        assert names["Pattern DE"] == 0 and names["Pattern CT"] == 1
        # the dot lands where the pyramid says
        lay = st.get_layer_id("landmarks")
        dot = st.get_objects(st.get_object_ids_in_group(st.get_group_id("Orientation lumen", lay)))
        assert dot.geometry.iloc[0].x == pytest.approx(37) and dot.geometry.iloc[0].y == pytest.approx(11)
        # the front is a polyline
        fr = st.get_objects(st.get_object_ids_in_group(st.get_group_id("Invasive front", lay)))
        assert fr["type"].astype(str).iloc[0] == "POLYLINE"
        assert list(fr.geometry.iloc[0].coords) == [(10, 10), (50, 60), (90, 70)]
        # constant attribute on every object
        ids = np.concatenate([st.get_object_ids_in_layer(l) for l in
                              (st.get_layer_id("tissue"), st.get_layer_id("morphology"), lay)])
        assert all(d == {"approx": 1.0} for d in st.get_scalar_dicts(ids).values())
        tumor = st.get_objects(st.get_object_ids_in_layer(st.get_layer_id("tissue")))
        assert tumor.geometry.iloc[0].bounds == pytest.approx((100, 100, 300, 250))


def test_unmapped_group_is_an_error_and_writes_nothing(raster, tmp_path):
    f = tmp_path / "b.xml"
    f.write_text(xml([("Polygon", "Pattern DI", [(0, 0), (9, 0), (9, 9)]),
                      ("Polygon", "Pattern  CT", [(0, 0), (9, 0), (9, 9)]),
                      ("Polygon", "Tumor", [(0, 0), (9, 0), (9, 9)])]))
    out = tmp_path / "b.ann.sqlite"
    with pytest.raises(ValueError, match=r"'Pattern  CT' \(1\), 'Pattern DI' \(1\)"):
        asap_to_store(f, out, raster, group_map=GROUP_MAP)
    assert not out.exists()

    rep = asap_to_store(f, out, raster, group_map=GROUP_MAP, on_unmapped="skip")
    assert rep["unmapped"] == {"Pattern DI": 1, "Pattern  CT": 1}
    assert rep["n_objects"] == 1


def test_invalid_polygons_repaired_degenerate_skipped(tmp_path):
    f = tmp_path / "c.xml"
    f.write_text(xml([
        ("Polygon", "Tumor", [(0, 0), (100, 100), (100, 0), (0, 100)]),      # bow-tie
        ("Polygon", "Tumor", [(5, 5), (5, 5), (9, 9)]),                       # 2 distinct points
        ("Polygon", "Tumor", [(0, 0), (50, 0), (50, 50), (0, 50)]),           # fine
    ]))
    rep = asap_to_store(f, tmp_path / "c.ann.sqlite", group_map=GROUP_MAP,
                        image_shape=ImageShape(width=200, height=200), mpp=0.5)
    assert rep["n_objects"] == 2
    assert len(rep["repaired"]) == 1 and "Tumor#0" in rep["repaired"][0]
    assert len(rep["skipped"]) == 1 and "Tumor#1" in rep["skipped"][0]
    with AnnotationStore(str(tmp_path / "c.ann.sqlite")) as st:
        g = st.get_objects(st.get_object_ids_in_layer(st.get_layer_id("tissue")))
        assert g.geometry.is_valid.all()
        assert st.mpp == 0.5 and st.image_shape == ImageShape(width=200, height=200)


def test_out_of_bounds_reported(raster, tmp_path):
    w, h = raster.shape(0).width, raster.shape(0).height
    f = tmp_path / "d.xml"
    f.write_text(xml([
        ("Polygon", "Tumor", [src(w + 50, 10), src(w + 90, 10), src(w + 90, 40)]),       # outside
        ("Polygon", "Tumor", [src(w - 10, 10), src(w + 40, 10), src(w + 40, 40)]),       # partial
        ("Polygon", "Tumor", [src(10, 10), src(40, 10), src(40, 40)]),                   # inside
    ]))
    rep = asap_to_store(f, tmp_path / "d.ann.sqlite", raster, group_map=GROUP_MAP)
    kinds = sorted(o["kind"] for o in rep["out_of_bounds"])
    assert kinds == ["outside", "partial"]
    assert all(o["group"] == "Tumor" for o in rep["out_of_bounds"])


def test_coordinates_sorted_by_order(tmp_path):
    f = tmp_path / "e.xml"
    f.write_text(xml([("PointSet", "Invasive front", [(2, 30, 30), (0, 10, 10), (1, 20, 20)])]))
    parsed = read_asap_xml(f)
    assert parsed.items[0].xy.tolist() == [[10, 10], [20, 20], [30, 30]]
    assert parsed.declared_groups[0] == "Tumor"


def test_bad_files(tmp_path):
    f = tmp_path / "f.xml"
    f.write_text("<Foo/>")
    with pytest.raises(ValueError, match="not an ASAP file"):
        read_asap_xml(f)
    f.write_text(xml([("Spline", "Tumor", [(0, 0), (1, 1)])]))
    with pytest.raises(ValueError, match="unsupported type"):
        read_asap_xml(f)
    f.write_text("<ASAP_Annotations><Annotations><Annotation Type='Dot' PartOfGroup='Tumor'/>"
                 "</Annotations></ASAP_Annotations>")
    with pytest.raises(ValueError, match="no coordinates"):
        read_asap_xml(f)
    with pytest.raises(FileExistsError):
        g = tmp_path / "g.xml"
        g.write_text(xml([("Polygon", "Tumor", [(0, 0), (9, 0), (9, 9)])]))
        o = tmp_path / "g.ann.sqlite"
        asap_to_store(g, o, group_map=GROUP_MAP, image_shape=ImageShape(width=20, height=20))
        asap_to_store(g, o, group_map=GROUP_MAP, overwrite=False)


def test_prune_empty_removes_empty_groups_and_layers(raster, tmp_path):
    f = tmp_path / "p.xml"
    f.write_text(xml([("Polygon", "Tumor", [src(10, 10), src(40, 10), src(40, 40)])]))
    out = tmp_path / "p.ann.sqlite"
    rep = asap_to_store(f, out, raster, group_map=GROUP_MAP, groups=VOCAB, prune_empty=True)
    assert rep["n_objects"] == 1
    assert "base/no_group" in rep["pruned"] and "morphology/Pattern CT" in rep["pruned"]
    with AnnotationStore(str(out)) as st:
        assert st.get_layer_names() == ["tissue"]
        assert st.get_groups()["name"].to_list() == ["Tumor"]
        assert st.count_objects_per_group()["n_objects"].to_list() == [1]
    # all empty: nothing left, but still a valid store
    f.write_text(xml([("Dot", "Three points", [src(1, 1)])]))
    rep = asap_to_store(f, out, raster, group_map=GROUP_MAP, groups=VOCAB, prune_empty=True)
    assert rep["n_objects"] == 0
    with AnnotationStore(str(out)) as st:
        assert st.get_layer_names() == [] and len(st) == 0
