# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""The legacy annotation API (``compass.legacy.annot.Annotation`` and the
module functions of ``compass.legacy.annot_sqlite_storage``) on top of
``AnnotationStore``.

Existing scripts port by changing their imports::

    # before
    import compass.annot as ann
    from compass.annot_sqlite_storage import save_annotation, load_annotation
    # after
    import compass.core.annot as ann
    from compass.core.annot import save_annotation, load_annotation

Differences from the legacy behaviour:
  - ``Annotation`` keeps its objects in an in-memory SQLite store. Objects
    are *copied* in when added (legacy kept references): changing an object
    after adding it does not change the annotation. Edit through
    ``annotation.store`` instead.
  - group names are scoped to a layer (as in the file format); legacy
    treated them as global, which could produce unsaveable annotations.
  - ``scan_objects(group=None)`` yields each object once, even if it is in
    several groups of the layer.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..types import ImageShape
from .objects import AnnotationObject, create_empty_annotation_object
from .schema import ANNOT_CODE_TYPE, ANNOT_TYPE_CODE
from .store import DEFAULT_GROUP, DEFAULT_LAYER, AnnotationStore

if TYPE_CHECKING:
    from ..pyramid import PyramidalRaster

__all__ = [
    "Annotation",
    "save_annotation",
    "load_annotation",
    "query_layer_id",
    "query_group_id",
    "query_groups_in_layer",
    "query_object_ids_in_layer",
    "query_object_ids_in_group",
    "query_object_ids_in_layer_roi",
    "fetch_objects",
    "fetch_geometries",
    "fetch_scalar_data",
]

_SCAN_BATCH = 10_000


#####
class Annotation:
    """A collection of annotation objects on one coordinate system, in
    layers and groups (legacy API). Defaults: layer "base", group
    "no_group". Backed by an in-memory ``AnnotationStore`` (``.store``)
    for vectorized access."""

    def __init__(self, name: str = "", image_shape: ImageShape | None = None,
                 mpp: float = 1.0, *, store: AnnotationStore | None = None):
        if store is not None:
            self._store = store
            return
        self._store = AnnotationStore(":memory:", mode="w")
        shape = image_shape if image_shape is not None else ImageShape(width=0, height=0)
        self._store.set_meta(name=name, image_width=shape.width,
                             image_height=shape.height, mpp=float(mpp))

    @property
    def store(self) -> AnnotationStore:
        """The underlying store: GeoDataFrame queries, bulk edits."""
        return self._store

    @property
    def name(self) -> str:
        return self._store.name

    @property
    def type(self) -> str:
        return "Annotation"

    def get_base_image_shape(self) -> ImageShape:
        return self._store.image_shape or ImageShape(width=0, height=0)

    def get_resolution(self) -> float:
        mpp = self._store.mpp
        return 1.0 if mpp is None else mpp

    def get_layer_names(self, sorted_list: bool = False) -> list[str]:
        names = self._store.get_layer_names()
        return sorted(names) if sorted_list else names

    def add_annotation_object(self, a: AnnotationObject,
                              group: str | list[str] = DEFAULT_GROUP,
                              layer: str = DEFAULT_LAYER) -> None:
        """Add (a copy of) one object; groups/layer created if needed.
        Sets ``a.id``."""
        self._store.add_objects([a], layer=layer, group=group)

    def add_annotations(self, a_list: list[AnnotationObject],
                        group: str | list[str] = DEFAULT_GROUP,
                        layer: str = DEFAULT_LAYER) -> None:
        """Add (copies of) several objects in one transaction."""
        self._store.add_objects(a_list, layer=layer, group=group)

    def scan_objects(self, layer: str = DEFAULT_LAYER,
                     group: str | None = DEFAULT_GROUP,
                     filter_object_type: int | str | None = None
                     ) -> Iterator[AnnotationObject]:
        """Iterate over the objects of a layer (``group=None``) or of one of
        its groups, optionally only those of one type (name or code).
        Objects are fetched in batches, in id order."""
        lid = self._store.get_layer_id(layer)
        if group is None:
            ids = self._store.get_object_ids_in_layer(lid)
        else:
            ids = self._store.get_object_ids_in_group(self._store.get_group_id(group, lid))
        if filter_object_type is not None:
            if isinstance(filter_object_type, (int, np.integer)):
                filter_object_type = ANNOT_CODE_TYPE[int(filter_object_type)]
            filter_object_type = str(filter_object_type).upper()
            if filter_object_type not in ANNOT_TYPE_CODE:
                raise ValueError(f"unknown annotation type {filter_object_type!r}")
        for i in range(0, ids.size, _SCAN_BATCH):
            for obj in self._store.fetch_objects(ids[i:i + _SCAN_BATCH]):
                if filter_object_type is None or obj.type == filter_object_type:
                    yield obj

    def resize(self, factor: float) -> None:
        """Rescale all objects by ``factor``; mpp and image shape follow."""
        self._store.resize(factor)

    def set_resolution(self, mpp: float) -> None:
        self._store.set_resolution(mpp)

    def match_image(self, pyr_img: "PyramidalRaster", pyr_level: int = 0) -> None:
        self._store.match_image(pyr_img, pyr_level)

    # -- (de)serialization ---------------------------------------------------

    def asdict(self) -> dict:
        """Legacy dictionary form (fully materialized: small annotations)."""
        st = self._store
        layers = st.get_layers()
        groups = st.get_groups()
        meta = st.meta
        annots: dict[int, list[dict]] = {}
        for gid in groups["group_id"].to_list():
            ids = st.get_object_ids_in_group(gid)
            annots[gid] = [o.asdict() for o in st.fetch_objects(ids)]
        return {
            "name": self.name,
            "image_shape": self.get_base_image_shape(),
            "mpp": self.get_resolution(),
            "id_layer": dict(zip(layers["layer_id"].to_list(), layers["name"].to_list())),
            "id_group": dict(zip(groups["group_id"].to_list(), groups["name"].to_list())),
            "id_reg": {"layer": int(meta.get("last_layer_id", 0)),
                       "group": int(meta.get("last_group_id", 0)),
                       "object": int(meta.get("last_object_id", -1))},
            "layers": {lid: groups.filter(groups["layer_id"] == lid)["group_id"].to_list()
                       for lid in layers["layer_id"].to_list()},
            "annotations": annots,
        }

    def fromdict(self, d: dict) -> None:
        """Replace the content with a legacy dictionary (``asdict()``)."""
        shape = d["image_shape"]
        if isinstance(shape, dict):
            shape = ImageShape(**shape)
        st = AnnotationStore(":memory:", mode="w")
        with st.batch():
            st.set_meta(name=d["name"], image_width=shape.width,
                        image_height=shape.height, mpp=float(d["mpp"]))
            st._conn.execute("DELETE FROM groups")
            st._conn.execute("DELETE FROM layers")
            st._conn.executemany("INSERT INTO layers(layer_id, name) VALUES (?, ?)",
                                 [(int(k), str(v)) for k, v in d["id_layer"].items()])
            g2l = {int(g): int(lid) for lid, gs in d["layers"].items() for g in gs}
            st._conn.executemany(
                "INSERT INTO groups(group_id, layer_id, name) VALUES (?, ?, ?)",
                [(int(g), g2l[int(g)], str(nm)) for g, nm in d["id_group"].items()])
            objs: dict[int, AnnotationObject] = {}
            members: list[tuple[int, int]] = []
            for gid, lst in d["annotations"].items():
                for od in lst:
                    if int(od["id"]) not in objs:
                        o = create_empty_annotation_object(od["annotation_type"])
                        o.fromdict(od)
                        objs[int(od["id"])] = o
                    members.append((int(od["id"]), int(gid)))
            _insert_with_ids(st, objs, members)
            reg = d.get("id_reg", {})
            st.set_meta(last_layer_id=int(reg.get("layer", 0)),
                        last_group_id=int(reg.get("group", 0)),
                        last_object_id=max(int(reg.get("object", -1)),
                                           max(objs, default=-1)))
        self._store.close()
        self._store = st

    # -- files ---------------------------------------------------------------

    def save(self, path: str | Path, overwrite: bool = True) -> Path:
        """Write to a SQLite annotation file."""
        return self._store.save_as(path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> "Annotation":
        """Load a SQLite annotation file (into memory)."""
        return cls(store=AnnotationStore.load_in_memory(path))
##


def _insert_with_ids(st: AnnotationStore, objs: dict[int, AnnotationObject],
                     members: list[tuple[int, int]]) -> None:
    """Insert objects keeping their given ids (fromdict)."""
    if not objs:
        return
    ids = sorted(objs)
    tmp = AnnotationStore(":memory:", mode="w")
    try:
        # encode via the regular path, then copy rows with the wanted ids
        new = tmp.add_objects([objs[i] for i in ids])
        rows = tmp._conn.execute(
            "SELECT object_id, type_code, name, wkb, circle_r, min_x, min_y, max_x, max_y "
            "FROM objects ORDER BY object_id").fetchall()
        remap = dict(zip(new.tolist(), ids))
        st._conn.executemany(
            "INSERT INTO objects(object_id, type_code, name, wkb, circle_r, "
            "min_x, min_y, max_x, max_y) VALUES (?,?,?,?,?,?,?,?,?)",
            [(remap[r[0]], *r[1:]) for r in rows])
    finally:
        tmp.close()
    st._conn.executemany(
        "INSERT OR IGNORE INTO object_groups(object_id, group_id) VALUES (?, ?)", members)
    st._write_scalars(np.array(ids, dtype=np.int64), [objs[i].data for i in ids],
                      replace=True)
    for i in ids:
        objs[i]._id = i


# ---------------------------------------------------------------------------
# legacy annot_sqlite_storage functions
# ---------------------------------------------------------------------------

def save_annotation(annotation: Annotation, path: str | Path, *,
                    overwrite: bool = True) -> None:
    """Write an ``Annotation`` to a SQLite file (replacing its content)."""
    annotation.save(path, overwrite=overwrite)


def load_annotation(path: str | Path) -> Annotation:
    """Read a SQLite annotation file into an in-memory ``Annotation``."""
    return Annotation.load(path)


def query_layer_id(path: str | Path, layer_name: str) -> int:
    with AnnotationStore(path) as st:
        return st.get_layer_id(layer_name)


def query_group_id(path: str | Path, group_name: str, layer_id: int | None = None) -> int:
    with AnnotationStore(path) as st:
        return st.get_group_id(group_name, layer_id)


def query_groups_in_layer(path: str | Path, layer_id: int) -> list[tuple[int, str]]:
    with AnnotationStore(path) as st:
        g = st.get_groups(layer_id)
        return list(zip(g["group_id"].to_list(), g["name"].to_list()))


def query_object_ids_in_layer(path: str | Path, layer_id: int) -> list[int]:
    with AnnotationStore(path) as st:
        return st.get_object_ids_in_layer(layer_id).tolist()


def query_object_ids_in_group(path: str | Path, group_id: int) -> list[int]:
    with AnnotationStore(path) as st:
        return st.get_object_ids_in_group(group_id).tolist()


def query_object_ids_in_layer_roi(path: str | Path, layer_id: int,
                                  roi: tuple[float, float, float, float]) -> list[int]:
    with AnnotationStore(path) as st:
        return st.get_object_ids_in_roi(layer_id, roi).tolist()


def fetch_objects(path: str | Path, object_ids: list[int]) -> list[AnnotationObject]:
    """Objects (with data) for the ids, as instances."""
    with AnnotationStore(path) as st:
        return st.fetch_objects(object_ids)


def fetch_geometries(path: str | Path, object_ids: list[int]) -> list[AnnotationObject]:
    """Objects for the ids, without their data."""
    with AnnotationStore(path) as st:
        return st.fetch_objects(object_ids, with_data=False)


def fetch_scalar_data(path: str | Path, object_ids: list[int]) -> dict[int, dict[str, float]]:
    """``{object_id: {attr_name: value}}`` for the ids."""
    with AnnotationStore(path) as st:
        return st.get_scalar_dicts(object_ids)
