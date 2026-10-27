# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Import of ASAP (https://github.com/computationalpathologygroup/ASAP) XML
annotations into an :class:`~compass.core.annot.AnnotationStore`.

ASAP stores, per slide, a flat list of ``<Annotation Type=... PartOfGroup=...>``
elements with ordered ``<Coordinate>`` children, in the pixel coordinates of
the *source* slide's level 0. This module is vocabulary-agnostic: the caller
supplies ``group_map``, which sends each raw ASAP group name to a
``(layer, group)`` of the store (optionally changing the geometry type) or
to ``None`` to drop it. Unmapped group names are an error by default, so
typos in the source files cannot pass silently.

Coordinates are mapped onto an imported pyramid with the source -> output
transform of its import record (``raster.info["import"]["transform"]``), the
recipe of ``docs/annotations.md``. Only the standard library parser is used.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__all__ = ["AsapItem", "AsapTarget", "asap_to_store", "read_asap_xml"]

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import shapely
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShPolygon

from compass.core import ImageShape
from compass.core.annot import AnnotationStore, Point, PointSet, PolyLine, Polygon

#: ASAP annotation types understood here -> COMPASS type
_TYPES = {"dot": "POINT", "pointset": "POINTSET", "polygon": "POLYGON",
          "rectangle": "POLYGON", "polyline": "POLYLINE"}
#: geometry conversions allowed through ``AsapTarget.as_type``
_CONVERSIONS = {("POINTSET", "POLYLINE"), ("POLYLINE", "POINTSET"),
                ("POLYGON", "POLYGON"), ("POINT", "POINT"),
                ("POINTSET", "POINTSET"), ("POLYLINE", "POLYLINE")}
#: tolerance (pixels, output level 0) for the inside-the-image check
_BOUNDS_TOL = 2.0


@dataclass(frozen=True)
class AsapItem:
    """One ``<Annotation>`` of an ASAP file."""
    type: str                 # COMPASS type: POINT, POINTSET, POLYLINE, POLYGON
    asap_type: str            # as written in the file
    group: str                # raw PartOfGroup
    name: str
    xy: np.ndarray            # (n, 2), sorted by the Order attribute


@dataclass(frozen=True)
class AsapTarget:
    """Destination of a raw ASAP group: layer, group and, optionally, the
    COMPASS type to store it as (``as_type="POLYLINE"`` turns an ASAP
    point set into an ordered line)."""
    layer: str
    group: str
    as_type: Optional[str] = None


@dataclass
class AsapFile:
    """Parsed ASAP file: declared group names and the annotation items."""
    path: Path
    declared_groups: list = field(default_factory=list)
    items: list = field(default_factory=list)


def read_asap_xml(path: Union[str, Path]) -> AsapFile:
    """Parse an ASAP XML file.

    Raises:
        ValueError: not an ``ASAP_Annotations`` document, an unknown
            annotation type, or an annotation without coordinates.
    """
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"{path.name}: not valid XML: {e}") from e
    if root.tag != "ASAP_Annotations":
        raise ValueError(f"{path.name}: not an ASAP file (root <{root.tag}>)")

    out = AsapFile(path=path)
    out.declared_groups = [g.get("Name", "") for g in root.iter("Group")]
    for k, a in enumerate(root.iter("Annotation")):
        atype = a.get("Type", "")
        if atype.lower() not in _TYPES:
            raise ValueError(f"{path.name}: annotation {k}: unsupported type {atype!r}")
        coords = a.findall("./Coordinates/Coordinate")
        if not coords:
            raise ValueError(f"{path.name}: annotation {k} ({atype}) has no coordinates")
        coords = sorted(coords, key=lambda c: int(c.get("Order", "0")))
        xy = np.array([[float(c.get("X")), float(c.get("Y"))] for c in coords])
        out.items.append(AsapItem(type=_TYPES[atype.lower()], asap_type=atype,
                                  group=a.get("PartOfGroup", ""),
                                  name=a.get("Name", f"Annotation {k}"), xy=xy))
    return out


def _as_target(v) -> Optional[AsapTarget]:
    if v is None or isinstance(v, AsapTarget):
        return v
    return AsapTarget(*v)


def _repair_polygon(xy: np.ndarray):
    """Valid polygonal geometry from a vertex ring, or ``(None, reason)``.
    Returns ``(geom, repaired)``."""
    if len(np.unique(xy, axis=0)) < 3:
        return None, False
    g = ShPolygon(xy)
    if g.is_valid and not g.is_empty:
        return g, False
    fixed = shapely.make_valid(g)
    parts = [p for p in getattr(fixed, "geoms", [fixed]) if isinstance(p, (ShPolygon, MultiPolygon))]
    parts = [q for p in parts for q in getattr(p, "geoms", [p])]
    parts = [p for p in parts if not p.is_empty and p.area > 0]
    if not parts:
        return None, True
    return (parts[0] if len(parts) == 1 else MultiPolygon(parts)), True


def _make_object(typ: str, xy: np.ndarray, name: str):
    """(COMPASS object or None, repaired flag)."""
    if typ == "POINT":
        return Point(tuple(xy[0]), name=name), False
    if typ == "POINTSET":
        return PointSet([tuple(p) for p in xy], name=name), False
    if typ == "POLYLINE":
        return PolyLine([tuple(p) for p in xy], name=name), False
    g, repaired = _repair_polygon(xy)
    return (None if g is None else Polygon(g, name=name)), repaired


def asap_to_store(xml_path: Union[str, Path],
                  store_path: Union[str, Path],
                  raster=None,
                  *,
                  group_map: Mapping[str, Union[AsapTarget, Sequence, None]],
                  groups: Iterable[Sequence[str]] = (),
                  on_unmapped: str = "error",
                  matrix: Optional[Sequence[Sequence[float]]] = None,
                  image_shape: Optional[ImageShape] = None,
                  mpp: Optional[float] = None,
                  data: Optional[Mapping[str, float]] = None,
                  meta: Optional[Mapping[str, object]] = None,
                  prune_empty: bool = False,
                  overwrite: bool = True) -> dict:
    """Import an ASAP XML file into a new annotation store.

    Args:
        xml_path: the ASAP file.
        store_path: the store to create (``.ann.sqlite``; ``":memory:"`` is
            not supported here because the store is closed on return).
        raster: the imported pyramid (a :class:`~compass.core.ZarrRaster`).
            Its import record gives the source -> output transform and its
            level 0 gives ``image_shape`` and ``mpp``; without an import
            record the identity is used. Instead of ``raster`` the transform
            and the image can be given as ``matrix`` (2 x 3), ``image_shape``
            and ``mpp``.
        group_map: raw ASAP group name -> ``(layer, group)`` /
            :class:`AsapTarget`, or ``None`` to drop the group on purpose.
        groups: ``(layer, group)`` pairs created in the store even when
            empty (the vocabulary), in this order.
        on_unmapped: ``"error"`` (default) raises ``ValueError`` listing all
            unmapped group names before anything is written; ``"skip"`` drops
            them and lists them in the report.
        data: constant scalar attributes given to every object.
        meta: extra metadata keys written to the store.
        prune_empty: remove groups without objects (also the default
            ``base``/``no_group``) and the layers left without groups.
        overwrite: replace an existing ``store_path``.

    Returns:
        A JSON-serializable report: ``n_objects``, ``per_group``
        (``"layer/group"`` -> count), ``dropped`` and ``unmapped`` (raw
        group -> count), ``repaired`` / ``skipped`` / ``out_of_bounds``
        (lists of descriptions), ``declared_groups``, ``source``, ``pruned``
        (``"layer/group"`` of the empty groups removed).
    """
    if on_unmapped not in ("error", "skip"):
        raise ValueError("on_unmapped must be 'error' or 'skip'")
    store_path = Path(store_path)
    if str(store_path) == ":memory:":
        raise ValueError("a file path is required")
    if store_path.exists() and not overwrite:
        raise FileExistsError(store_path)

    parsed = read_asap_xml(xml_path)
    targets = {k: _as_target(v) for k, v in group_map.items()}

    unmapped = Counter(it.group for it in parsed.items if it.group not in targets)
    if unmapped and on_unmapped == "error":
        raise ValueError(f"{parsed.path.name}: unmapped ASAP groups: "
                         + ", ".join(f"{g!r} ({n})" for g, n in sorted(unmapped.items())))
    for t in targets.values():
        if t is not None and t.as_type is not None and t.as_type not in {
                "POINT", "POINTSET", "POLYLINE", "POLYGON"}:
            raise ValueError(f"unsupported as_type {t.as_type!r}")

    # transform and target image
    rec = None
    if raster is not None:
        rec = (raster.info or {}).get("import")
        if image_shape is None:
            image_shape = raster.shape(0)
        if mpp is None:
            mpp = raster.get_mpp_for_level(0)
    if matrix is None:
        matrix = rec["transform"]["matrix"] if rec else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    report = {"source": parsed.path.name, "declared_groups": parsed.declared_groups,
              "per_group": {}, "dropped": {}, "unmapped": dict(unmapped),
              "repaired": [], "skipped": [], "out_of_bounds": [], "n_objects": 0,
              "pruned": []}

    # build the objects per destination first, so a failure writes nothing
    by_dest: dict = {}
    for k, it in enumerate(parsed.items):
        if it.group not in targets:
            continue
        t = targets[it.group]
        if t is None:
            report["dropped"][it.group] = report["dropped"].get(it.group, 0) + 1
            continue
        typ = t.as_type or it.type
        if (it.type, typ) not in _CONVERSIONS:
            raise ValueError(f"{parsed.path.name}: cannot store ASAP {it.asap_type} "
                             f"of group {it.group!r} as {typ}")
        obj, repaired = _make_object(typ, it.xy, it.name)
        label = f"{it.group}#{k} ({it.asap_type}, {len(it.xy)} vertices)"
        if obj is None:
            report["skipped"].append(label + ": degenerate geometry")
            continue
        if repaired:
            report["repaired"].append(label)
        by_dest.setdefault((t.layer, t.group), []).append(obj)

    store_path.parent.mkdir(parents=True, exist_ok=True)
    with AnnotationStore(str(store_path), mode="w") as store:
        store.set_meta(name=parsed.path.stem, annotation_format="ASAP",
                       annotation_source=parsed.path.name, **{k: str(v) for k, v in (meta or {}).items()})
        with store.batch():
            for layer, group in groups:
                store.add_layer(layer)
                store.add_group(group, layer=layer)
            for (layer, group), objs in by_dest.items():
                store.add_layer(layer)
                store.add_group(group, layer=layer)
                ids = store.add_objects(objs, layer=layer, group=group,
                                        data=[dict(data) for _ in objs] if data else None)
                report["per_group"][f"{layer}/{group}"] = len(ids)
                report["n_objects"] += len(ids)
        if prune_empty:
            lname = {r["layer_id"]: r["name"] for r in store.get_layers().iter_rows(named=True)}
            for r in store.count_objects_per_group().iter_rows(named=True):
                if r["n_objects"] == 0:
                    store.remove_group(r["group_id"], layer=r["layer_id"])
                    report["pruned"].append(f"{lname[r['layer_id']]}/{r['group_name']}")
            for lid, name in lname.items():
                if store.get_groups(layer_id=lid).height == 0:
                    store.remove_layer(lid)
        if store.n_objects:
            store.affine(matrix)
        if image_shape is not None:
            store.image_shape = image_shape
        if mpp is not None:
            store.mpp = mpp

        if image_shape is not None and store.n_objects:
            bb = store.get_bboxes()
            w, h = image_shape.width, image_shape.height
            ids = bb["object_id"].to_numpy()
            names = store.get_objects(ids)["name"]
            gname = {r["group_id"]: r["name"] for r in store.get_groups().iter_rows(named=True)}
            omap = dict(zip(*(c.to_list() for c in store.get_memberships(ids).get_columns())))
            for row in bb.iter_rows(named=True):
                outside = (row["max_x"] < -_BOUNDS_TOL or row["max_y"] < -_BOUNDS_TOL
                           or row["min_x"] > w + _BOUNDS_TOL or row["min_y"] > h + _BOUNDS_TOL)
                partial = (row["min_x"] < -_BOUNDS_TOL or row["min_y"] < -_BOUNDS_TOL
                           or row["max_x"] > w + _BOUNDS_TOL or row["max_y"] > h + _BOUNDS_TOL)
                if outside or partial:
                    oid = row["object_id"]
                    report["out_of_bounds"].append(
                        {"object_id": int(oid), "name": str(names.loc[oid]),
                         "group": gname.get(omap.get(oid)),
                         "kind": "outside" if outside else "partial"})
    return report
