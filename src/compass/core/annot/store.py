# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Vectorized read/write access to the SQLite/R-tree annotation store.

The physical schema (``schema.py``) is the legacy one, kept as-is: WKB
geometries + bbox columns in ``objects``, an ``objects_rtree`` R-tree kept
in sync by triggers, a layer -> group -> object hierarchy, and sparse
per-object scalar attributes (``object_scalar_sparse``: parallel uint32
attribute-id / float32 value blobs, with a shared ``scalar_attributes``
name<->id dictionary).

Query results come back as **geopandas GeoDataFrames** (geometry decoded in
one vectorized ``shapely.from_wkb`` pass) or **polars DataFrames** — never
as one Python object per annotation, except through ``fetch_objects`` (for
inspecting a few selected objects).

Usage
-----
    # read-only (the viewer)
    store = AnnotationStore("slide_001.ann.sqlite")
    gdf = store.get_objects_in_roi(layer_id=0, roi=(x0, y0, x1, y1))

    # build / edit
    with AnnotationStore("slide_001.ann.sqlite", mode="w") as store:
        ids = store.add_objects(cells_gdf, layer="cells", group="tumor",
                                data=expression_frame)
        store.remove_objects(ids[:10])
        store.affine(import_record["transform"]["matrix"])
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import logging
import os
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import shapely

from ..types import ImageShape
from .objects import AnnotationObject, Circle, make_annotation_object
from .schema import (ANNOT_CODE_TYPE, ANNOT_TYPE_CODE, SCHEMA_VERSION, SQLITE_DDL,
                     geometry_type_names)
from .transform import CircleMode, as_affine_matrix, similarity_scale, transform_rows

if TYPE_CHECKING:
    from ..pyramid import PyramidalRaster

logger = logging.getLogger(__name__)

_OBJECT_COLUMNS = "o.object_id, o.type_code, o.name, o.wkb, o.circle_r"
_CIRCLE = ANNOT_TYPE_CODE["CIRCLE"]

#: default layer / group, as in the legacy ``Annotation`` model
DEFAULT_LAYER = "base"
DEFAULT_GROUP = "no_group"

#: objects per batch for whole-store transforms
_TRANSFORM_BATCH = 100_000

LayerRef = int | str
GroupRef = int | str
Objects = AnnotationObject | Iterable[AnnotationObject] | gpd.GeoDataFrame | gpd.GeoSeries
ScalarData = Sequence[Mapping[str, Any] | None] | pd.DataFrame | pl.DataFrame


#####
class AnnotationStore:
    """Vectorized access to one SQLite annotation database.

    ``mode``:
      - ``"r"`` (default): read-only — what the viewer uses.
      - ``"r+"``: read/write an existing database.
      - ``"w"``: create a new database (an existing file is replaced).
        Pass ``path=":memory:"`` for an in-memory store; ``save_as()``
        writes it to disk.

    New databases get the legacy default layer ``"base"`` with group
    ``"no_group"`` (ids 0). All bulk queries return geopandas/polars frames
    indexed by / keyed on ``object_id``.

    Every write method is atomic. Wrap several in ``with store.batch():``
    to make them one transaction (much faster for many small writes).

    Rules kept by the write methods:
      - an object belongs to one layer, through one or more of its groups;
        an object left in no group (``remove_group``, ``remove_layer``,
        ``remove_from_group``) is deleted;
      - object, group and layer ids are never reused after a delete (the
        last ids handed out are kept in ``meta``, as the legacy writer did).

    For circles the stored geometry is the center POINT plus a ``circle_r``
    column; reads decode them to polygonal disks by default
    (``circles_as_disks=False`` keeps the centers). Writes accept either.
    """

    def __init__(self, path: str | Path, mode: Literal["r", "r+", "w"] = "r"):
        if mode not in ("r", "r+", "w"):
            raise ValueError(f"mode must be 'r', 'r+' or 'w', got {mode!r}")
        self._mode = mode
        self._tx_depth = 0
        memory = str(path) == ":memory:"
        self._path: Path | None = None if memory else Path(path)

        if memory:
            if mode != "w":
                raise ValueError("an in-memory store must be opened with mode='w'")
            self._conn = sqlite3.connect(":memory:", isolation_level=None)
        elif mode == "r":
            if not self._path.exists():
                raise ValueError(f"no such annotation database: {path}")
            uri = self._path.resolve().as_uri() + "?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
            self._conn.execute("PRAGMA query_only = ON;")
        else:
            if mode == "r+" and not self._path.exists():
                raise ValueError(f"no such annotation database: {path}")
            if mode == "w":
                _remove_db_files(self._path)
            self._conn = sqlite3.connect(str(self._path), isolation_level=None)
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")

        if self.writable:
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA temp_store = MEMORY;")
            # Up to 256 MB page cache (allocated as used): R-tree upkeep on
            # bulk writes thrashes the 2 MB default — ~30% faster at 1M objects.
            self._conn.execute("PRAGMA cache_size = -262144;")
            self._conn.executescript(SQLITE_DDL)
            with self._tx():
                self._meta_set("schema_version", SCHEMA_VERSION)
                if mode == "w":
                    self._conn.execute("INSERT INTO layers(layer_id, name) VALUES (0, ?)",
                                       (DEFAULT_LAYER,))
                    self._conn.execute(
                        "INSERT INTO groups(group_id, layer_id, name) VALUES (0, 0, ?)",
                        (DEFAULT_GROUP,))
                    self._meta_set("last_layer_id", "0")
                    self._meta_set("last_group_id", "0")
                    self._meta_set("last_object_id", "-1")

    @classmethod
    def load_in_memory(cls, path: str | Path) -> "AnnotationStore":
        """A writable in-memory copy of a database file (the file itself is
        not modified; ``save_as()`` writes the copy back)."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"no such annotation database: {path}")
        store = cls(":memory:", mode="w")
        src = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            src.backup(store._conn)
        finally:
            src.close()
        store._conn.execute("PRAGMA foreign_keys = ON;")
        store._conn.executescript(SQLITE_DDL)
        return store

    def save_as(self, path: str | Path, overwrite: bool = True) -> Path:
        """Write a complete copy of the database to ``path`` (SQLite backup
        API). Typical use: persist an in-memory store."""
        if self._tx_depth:
            raise RuntimeError("save_as() inside an open batch()")
        path = Path(path)
        if self._path is not None and path.resolve() == self._path.resolve():
            raise ValueError("save_as() target is the store's own file")
        if path.exists():
            if not overwrite:
                raise FileExistsError(path)
            _remove_db_files(path)
        dst = sqlite3.connect(str(path))
        try:
            self._conn.backup(dst)
        finally:
            dst.close()
        return path

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AnnotationStore":
        return self

    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    @property
    def path(self) -> Path | None:
        """Database file; None for an in-memory store."""
        return self._path

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def writable(self) -> bool:
        return self._mode != "r"

    def __len__(self) -> int:
        return self.n_objects

    @property
    def n_objects(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0])

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    @contextmanager
    def batch(self) -> Iterator["AnnotationStore"]:
        """Group several writes into one transaction: all are kept, or (on
        an exception) none."""
        with self._tx():
            yield self

    @contextmanager
    def _tx(self) -> Iterator[None]:
        """Nestable transaction (SQLite savepoints)."""
        self._require_writable()
        name = f"sp{self._tx_depth}"
        self._conn.execute(f"SAVEPOINT {name}")
        self._tx_depth += 1
        try:
            yield
        except BaseException:
            self._tx_depth -= 1
            self._conn.execute(f"ROLLBACK TO {name}")
            self._conn.execute(f"RELEASE {name}")
            raise
        else:
            self._tx_depth -= 1
            self._conn.execute(f"RELEASE {name}")

    def _require_writable(self) -> None:
        if not self.writable:
            raise PermissionError("annotation store is open read-only (mode='r'); "
                                  "open it with mode='r+' to edit")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def meta(self) -> dict[str, str]:
        """The ``meta`` key/value table (name, image_width, image_height, mpp, ...)."""
        return dict(self._conn.execute("SELECT key, value FROM meta").fetchall())

    def get_layers(self) -> pl.DataFrame:
        """All layers: columns ``layer_id``, ``name``."""
        rows = self._conn.execute(
            "SELECT layer_id, name FROM layers ORDER BY layer_id").fetchall()
        return pl.DataFrame(rows, schema={"layer_id": pl.Int64, "name": pl.Utf8}, orient="row")

    def get_groups(self, layer_id: int | None = None) -> pl.DataFrame:
        """Groups (optionally restricted to one layer): columns ``group_id``,
        ``layer_id``, ``name``."""
        sql = "SELECT group_id, layer_id, name FROM groups"
        params: tuple = ()
        if layer_id is not None:
            sql += " WHERE layer_id = ?"
            params = (int(layer_id),)
        rows = self._conn.execute(sql + " ORDER BY group_id", params).fetchall()
        return pl.DataFrame(rows,
                            schema={"group_id": pl.Int64, "layer_id": pl.Int64, "name": pl.Utf8},
                            orient="row")

    def get_layer_id(self, layer_name: str) -> int:
        """Resolve a layer name to its id (raises ValueError if unknown)."""
        row = self._conn.execute(
            "SELECT layer_id FROM layers WHERE name = ?", (str(layer_name),)).fetchone()
        if row is None:
            names = ", ".join(self.get_layers()["name"].to_list()) or "<none>"
            raise ValueError(f"Unknown layer name '{layer_name}'. Available layers: {names}")
        return int(row[0])

    def get_group_id(self, group_name: str, layer_id: int | None = None) -> int:
        """Resolve a group name to its id (raises ValueError if unknown).
        Group names are unique only within a layer; pass ``layer_id`` to
        disambiguate."""
        sql = "SELECT group_id FROM groups WHERE name = ?"
        params: list = [str(group_name)]
        if layer_id is not None:
            sql += " AND layer_id = ?"
            params.append(int(layer_id))
        row = self._conn.execute(sql, params).fetchone()
        if row is None:
            names = ", ".join(self.get_groups(layer_id)["name"].to_list()) or "<none>"
            raise ValueError(f"Unknown group name '{group_name}'. Available groups: {names}")
        return int(row[0])

    def count_objects_per_group(self) -> pl.DataFrame:
        """Object counts per group: columns ``group_id``, ``layer_id``,
        ``group_name``, ``n_objects``. Cheap; use it to drive the LOD
        (live-vector vs. datashader) decision before fetching anything."""
        rows = self._conn.execute(
            """
            SELECT g.group_id, g.layer_id, g.name, COUNT(og.object_id)
            FROM groups g
            LEFT JOIN object_groups og ON og.group_id = g.group_id
            GROUP BY g.group_id
            ORDER BY g.group_id
            """).fetchall()
        return pl.DataFrame(
            rows,
            schema={"group_id": pl.Int64, "layer_id": pl.Int64,
                    "group_name": pl.Utf8, "n_objects": pl.Int64},
            orient="row")

    def count_objects_in_roi(self, layer_id: int,
                             roi: tuple[float, float, float, float]) -> int:
        """Number of objects in the layer whose bbox intersects the ROI
        (R-tree only — no geometry decoding). roi = (x0, y0, x1, y1), base
        pixel coordinates."""
        x0, y0, x1, y1 = roi
        row = self._conn.execute(
            """
            SELECT COUNT(DISTINCT og.object_id)
            FROM objects_rtree r
            JOIN object_groups og ON og.object_id = r.object_id
            JOIN groups g         ON g.group_id = og.group_id
            WHERE g.layer_id = ?
              AND r.min_x <= ? AND r.max_x >= ?
              AND r.min_y <= ? AND r.max_y >= ?
            """,
            (int(layer_id), float(x1), float(x0), float(y1), float(y0))).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # ID queries (R-tree / hierarchy only, no geometry decoding)
    # ------------------------------------------------------------------

    def get_object_ids_in_layer(self, layer_id: int) -> np.ndarray:
        """Distinct object_ids belonging to any group of the layer."""
        rows = self._conn.execute(
            """
            SELECT DISTINCT og.object_id
            FROM groups g
            JOIN object_groups og ON og.group_id = g.group_id
            WHERE g.layer_id = ?
            ORDER BY og.object_id
            """, (int(layer_id),)).fetchall()
        return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))

    def get_object_ids_in_group(self, group_id: int) -> np.ndarray:
        """object_ids in a specific group."""
        rows = self._conn.execute(
            "SELECT object_id FROM object_groups WHERE group_id = ? ORDER BY object_id",
            (int(group_id),)).fetchall()
        return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))

    def get_object_ids_in_roi(self, layer_id: int,
                              roi: tuple[float, float, float, float],
                              group_ids: list[int] | np.ndarray | None = None) -> np.ndarray:
        """Distinct object_ids in the layer whose bbox intersects the ROI,
        optionally restricted to a subset of groups (the viewer's visible
        planes). roi = (x0, y0, x1, y1), base pixel coordinates."""
        x0, y0, x1, y1 = roi
        sql = """
            SELECT DISTINCT og.object_id
            FROM objects_rtree r
            JOIN object_groups og ON og.object_id = r.object_id
            JOIN groups g         ON g.group_id = og.group_id
            WHERE g.layer_id = ?
              AND r.min_x <= ? AND r.max_x >= ?
              AND r.min_y <= ? AND r.max_y >= ?
            """
        params: list = [int(layer_id), float(x1), float(x0), float(y1), float(y0)]
        if group_ids is not None:
            ids = [int(g) for g in group_ids]
            sql += f" AND og.group_id IN ({','.join('?' * len(ids))})"
            params.extend(ids)
        rows = self._conn.execute(sql + " ORDER BY og.object_id", params).fetchall()
        return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))

    # ------------------------------------------------------------------
    # Bulk geometry fetch -> GeoDataFrame
    # ------------------------------------------------------------------

    def get_objects(self, object_ids: np.ndarray | list[int],
                    circles_as_disks: bool = True) -> gpd.GeoDataFrame:
        """Fetch geometries for the given ids as a GeoDataFrame.

        Returns a GeoDataFrame indexed by ``object_id`` with columns
        ``geometry``, ``type`` (POINT/POINTSET/POLYLINE/POLYGON/CIRCLE),
        ``name`` and — for circles — ``circle_r`` (NaN elsewhere).
        Geometries are decoded in one vectorized ``shapely.from_wkb`` pass.
        """
        object_ids = np.asarray(object_ids, dtype=np.int64)
        if object_ids.size == 0:
            return self._empty_gdf()

        frames = []
        for chunk in _chunked(object_ids):
            qmarks = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT {_OBJECT_COLUMNS} FROM objects o "
                f"WHERE o.object_id IN ({qmarks}) ORDER BY o.object_id",
                [int(x) for x in chunk]).fetchall()
            frames.append(self._decode(rows, circles_as_disks))
        return frames[0] if len(frames) == 1 else gpd.GeoDataFrame(pd.concat(frames))

    def get_objects_in_roi(self, layer_id: int,
                           roi: tuple[float, float, float, float],
                           group_ids: list[int] | np.ndarray | None = None,
                           circles_as_disks: bool = True) -> gpd.GeoDataFrame:
        """The viewport query: geometries of all objects in the layer
        (optionally: in the given groups) whose bbox intersects the ROI, as a
        single GeoDataFrame. Combines the R-tree filter and the geometry
        fetch in one SQL pass.

        roi = (x0, y0, x1, y1), base pixel coordinates.
        """
        x0, y0, x1, y1 = roi
        sql = f"""
            SELECT DISTINCT {_OBJECT_COLUMNS}
            FROM objects_rtree r
            JOIN objects o        ON o.object_id = r.object_id
            JOIN object_groups og ON og.object_id = o.object_id
            JOIN groups g         ON g.group_id = og.group_id
            WHERE g.layer_id = ?
              AND r.min_x <= ? AND r.max_x >= ?
              AND r.min_y <= ? AND r.max_y >= ?
            """
        params: list = [int(layer_id), float(x1), float(x0), float(y1), float(y0)]
        if group_ids is not None:
            ids = [int(g) for g in group_ids]
            sql += f" AND og.group_id IN ({','.join('?' * len(ids))})"
            params.extend(ids)
        rows = self._conn.execute(sql + " ORDER BY o.object_id", params).fetchall()
        return self._decode(rows, circles_as_disks)

    def get_bboxes(self, layer_id: int | None = None) -> pl.DataFrame:
        """Bounding boxes (no geometry decoding): columns ``object_id``,
        ``min_x``, ``min_y``, ``max_x``, ``max_y``. Enough for datashader
        point-rendering of bbox centers or coarse hit-testing."""
        if layer_id is None:
            sql = "SELECT o.object_id, o.min_x, o.min_y, o.max_x, o.max_y FROM objects o ORDER BY o.object_id"
            params: tuple = ()
        else:
            sql = """
                SELECT DISTINCT o.object_id, o.min_x, o.min_y, o.max_x, o.max_y
                FROM objects o
                JOIN object_groups og ON og.object_id = o.object_id
                JOIN groups g         ON g.group_id = og.group_id
                WHERE g.layer_id = ?
                ORDER BY o.object_id
                """
            params = (int(layer_id),)
        rows = self._conn.execute(sql, params).fetchall()
        return pl.DataFrame(
            rows,
            schema={"object_id": pl.Int64, "min_x": pl.Float64, "min_y": pl.Float64,
                    "max_x": pl.Float64, "max_y": pl.Float64},
            orient="row")

    def get_memberships(self, object_ids: np.ndarray | list[int] | None = None) -> pl.DataFrame:
        """The object<->group membership table: columns ``object_id``,
        ``group_id`` (an object may appear in several groups of one layer)."""
        if object_ids is None:
            rows = self._conn.execute(
                "SELECT object_id, group_id FROM object_groups ORDER BY object_id, group_id"
            ).fetchall()
        else:
            rows = []
            for chunk in _chunked(np.asarray(object_ids, dtype=np.int64)):
                qmarks = ",".join("?" * len(chunk))
                rows.extend(self._conn.execute(
                    f"SELECT object_id, group_id FROM object_groups "
                    f"WHERE object_id IN ({qmarks}) ORDER BY object_id, group_id",
                    [int(x) for x in chunk]).fetchall())
        return pl.DataFrame(rows, schema={"object_id": pl.Int64, "group_id": pl.Int64},
                            orient="row")

    # ------------------------------------------------------------------
    # Sparse scalar attributes (e.g. molecular expression)
    # ------------------------------------------------------------------

    def get_scalar_attribute_names(self) -> pl.DataFrame:
        """The attribute dictionary: columns ``attr_id``, ``attr_name``."""
        rows = self._conn.execute(
            "SELECT attr_id, attr_name FROM scalar_attributes ORDER BY attr_id").fetchall()
        return pl.DataFrame(rows, schema={"attr_id": pl.Int64, "attr_name": pl.Utf8},
                            orient="row")

    def get_scalar_data_long(self, object_ids: np.ndarray | list[int]) -> pl.DataFrame:
        """Sparse per-object scalars in long (COO) form: columns
        ``object_id``, ``attr_id``, ``value``. The packed uint32/float32
        blobs are concatenated with numpy — no per-value Python loop."""
        object_ids = np.asarray(object_ids, dtype=np.int64)
        if object_ids.size == 0:
            return pl.DataFrame(schema={"object_id": pl.Int64, "attr_id": pl.Int64,
                                        "value": pl.Float32})

        oid_parts: list[np.ndarray] = []
        aid_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for chunk in _chunked(object_ids):
            qmarks = ",".join("?" * len(chunk))
            for oid, ids_blob, vals_blob in self._conn.execute(
                    f"SELECT object_id, ids_u32, vals_f32 FROM object_scalar_sparse "
                    f"WHERE object_id IN ({qmarks})",
                    [int(x) for x in chunk]):
                aids = np.frombuffer(ids_blob, dtype=np.uint32)
                if aids.size == 0:
                    continue
                oid_parts.append(np.full(aids.size, oid, dtype=np.int64))
                aid_parts.append(aids.astype(np.int64))
                val_parts.append(np.frombuffer(vals_blob, dtype=np.float32))

        if not oid_parts:
            return pl.DataFrame(schema={"object_id": pl.Int64, "attr_id": pl.Int64,
                                        "value": pl.Float32})
        return pl.DataFrame({
            "object_id": np.concatenate(oid_parts),
            "attr_id": np.concatenate(aid_parts),
            "value": np.concatenate(val_parts),
        })

    def get_scalar_data(self, object_ids: np.ndarray | list[int],
                        attributes: list[str] | None = None) -> pl.DataFrame:
        """Sparse per-object scalars pivoted to wide form: one row per
        ``object_id``, one column per attribute name, missing values null.
        Restrict with ``attributes`` (names) when only a few of thousands of
        attributes are needed — e.g. the genes driving a color map."""
        long = self.get_scalar_data_long(object_ids)
        names = self.get_scalar_attribute_names()
        long = long.join(names, on="attr_id", how="left").drop("attr_id")
        if attributes is not None:
            long = long.filter(pl.col("attr_name").is_in(list(attributes)))
        return long.pivot(on="attr_name", index="object_id", values="value")

    def get_layer_names(self) -> list[str]:
        """Layer names, in id order."""
        return [r[0] for r in self._conn.execute(
            "SELECT name FROM layers ORDER BY layer_id").fetchall()]

    # ------------------------------------------------------------------
    # Per-object instances (small selections only)
    # ------------------------------------------------------------------

    def fetch_objects(self, object_ids: np.ndarray | list[int],
                      with_data: bool = True) -> list[AnnotationObject]:
        """The given objects as ``Point`` / ``Polygon`` / ``Circle`` / ...
        instances (ordered by id), with their scalar data unless
        ``with_data=False``. One Python object per annotation: meant for
        inspecting a selection, not for bulk work — use ``get_objects``."""
        gdf = self.get_objects(object_ids, circles_as_disks=False)
        data = self.get_scalar_dicts(gdf.index.to_numpy()) if with_data else {}
        out = []
        for oid, tp, nm, r, geom in zip(gdf.index, gdf["type"].astype(str), gdf["name"],
                                        gdf["circle_r"], gdf.geometry):
            obj = make_annotation_object(tp, geom, name=None if nm is pd.NA else nm,
                                         data=data.get(int(oid)), circle_r=r)
            obj._id = int(oid)
            out.append(obj)
        return out

    def get_scalar_dicts(self, object_ids: np.ndarray | list[int]
                         ) -> dict[int, dict[str, float]]:
        """Scalars as ``{object_id: {attr_name: value}}`` (objects without
        values are absent)."""
        long = self.get_scalar_data_long(object_ids)
        if long.height == 0:
            return {}
        names = dict(self._conn.execute(
            "SELECT attr_id, attr_name FROM scalar_attributes").fetchall())
        out: dict[int, dict[str, float]] = {}
        for oid, aid, v in long.iter_rows():
            out.setdefault(int(oid), {})[names[aid]] = float(v)
        return out

    # ------------------------------------------------------------------
    # Metadata (write)
    # ------------------------------------------------------------------

    def _meta_get(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def _meta_set(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))

    def set_meta(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """Set ``meta`` keys (values are stored as text)."""
        items = {**(values or {}), **kwargs}
        with self._tx():
            for k, v in items.items():
                self._meta_set(str(k), v)

    @property
    def name(self) -> str:
        return self._meta_get("name", "") or ""

    @name.setter
    def name(self, value: str) -> None:
        self.set_meta(name=value)

    @property
    def mpp(self) -> float | None:
        """Resolution (microns per pixel) of the annotation coordinates."""
        v = self._meta_get("mpp")
        return None if v in (None, "") else float(v)

    @mpp.setter
    def mpp(self, value: float | None) -> None:
        self.set_meta(mpp="" if value is None else float(value))

    @property
    def image_shape(self) -> ImageShape | None:
        """Size of the image the coordinates refer to (meta ``image_width``,
        ``image_height``); None if not recorded."""
        w, h = self._meta_get("image_width"), self._meta_get("image_height")
        if w in (None, "") or h in (None, ""):
            return None
        return ImageShape(width=int(float(w)), height=int(float(h)))

    @image_shape.setter
    def image_shape(self, shape: ImageShape | tuple[int, int]) -> None:
        if not isinstance(shape, ImageShape):
            shape = ImageShape(width=int(shape[0]), height=int(shape[1]))
        self.set_meta(image_width=shape.width, image_height=shape.height)

    # ------------------------------------------------------------------
    # Layers and groups (write)
    # ------------------------------------------------------------------

    def _alloc_ids(self, table: str, column: str, meta_key: str, n: int) -> int:
        """First of ``n`` fresh ids: above both the current maximum and the
        last id ever handed out (so deleted ids are not reused)."""
        last = int(self._meta_get(meta_key, "-1") or "-1")
        mx = self._conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()[0]
        start = max(last, -1 if mx is None else int(mx)) + 1
        self._meta_set(meta_key, start + n - 1)
        return start

    def _resolve_layer(self, layer: LayerRef, create: bool = False) -> int:
        if isinstance(layer, (int, np.integer)):
            row = self._conn.execute("SELECT layer_id FROM layers WHERE layer_id = ?",
                                     (int(layer),)).fetchone()
            if row is None:
                raise ValueError(f"no layer with id {layer}")
            return int(layer)
        row = self._conn.execute("SELECT layer_id FROM layers WHERE name = ?",
                                 (str(layer),)).fetchone()
        if row is not None:
            return int(row[0])
        if create:
            lid = self._alloc_ids("layers", "layer_id", "last_layer_id", 1)
            self._conn.execute("INSERT INTO layers(layer_id, name) VALUES (?, ?)",
                               (lid, str(layer)))
            return lid
        names = ", ".join(self.get_layer_names()) or "<none>"
        raise ValueError(f"Unknown layer name '{layer}'. Available layers: {names}")

    def _resolve_group(self, group: GroupRef, layer: LayerRef | None = None,
                       create: bool = False) -> int:
        lid = None if layer is None else self._resolve_layer(layer, create=create)
        if isinstance(group, (int, np.integer)):
            row = self._conn.execute("SELECT layer_id FROM groups WHERE group_id = ?",
                                     (int(group),)).fetchone()
            if row is None:
                raise ValueError(f"no group with id {group}")
            if lid is not None and int(row[0]) != lid:
                raise ValueError(f"group {group} is not in layer {layer!r}")
            return int(group)
        sql, params = "SELECT group_id FROM groups WHERE name = ?", [str(group)]
        if lid is not None:
            sql += " AND layer_id = ?"
            params.append(lid)
        rows = self._conn.execute(sql, params).fetchall()
        if len(rows) == 1:
            return int(rows[0][0])
        if len(rows) > 1:
            raise ValueError(f"group name '{group}' exists in several layers; "
                             f"pass layer= to choose")
        if create and lid is not None:
            gid = self._alloc_ids("groups", "group_id", "last_group_id", 1)
            self._conn.execute("INSERT INTO groups(group_id, layer_id, name) VALUES (?, ?, ?)",
                               (gid, lid, str(group)))
            return gid
        names = ", ".join(self.get_groups(lid)["name"].to_list()) or "<none>"
        raise ValueError(f"Unknown group name '{group}'. Available groups: {names}")

    def add_layer(self, name: str, exist_ok: bool = True) -> int:
        """Create a layer; returns its id (the existing one if ``exist_ok``)."""
        with self._tx():
            row = self._conn.execute("SELECT layer_id FROM layers WHERE name = ?",
                                     (str(name),)).fetchone()
            if row is not None:
                if not exist_ok:
                    raise ValueError(f"layer '{name}' already exists")
                return int(row[0])
            return self._resolve_layer(str(name), create=True)

    def add_group(self, name: str, layer: LayerRef = DEFAULT_LAYER,
                  exist_ok: bool = True) -> int:
        """Create a group in ``layer`` (created too if it is a new name);
        returns the group id."""
        with self._tx():
            lid = self._resolve_layer(layer, create=True)
            row = self._conn.execute(
                "SELECT group_id FROM groups WHERE name = ? AND layer_id = ?",
                (str(name), lid)).fetchone()
            if row is not None:
                if not exist_ok:
                    raise ValueError(f"group '{name}' already exists in layer {layer!r}")
                return int(row[0])
            return self._resolve_group(str(name), lid, create=True)

    def rename_layer(self, layer: LayerRef, new_name: str) -> None:
        with self._tx():
            lid = self._resolve_layer(layer)
            self._conn.execute("UPDATE layers SET name = ? WHERE layer_id = ?",
                               (str(new_name), lid))

    def rename_group(self, group: GroupRef, new_name: str,
                     layer: LayerRef | None = None) -> None:
        with self._tx():
            gid = self._resolve_group(group, layer)
            self._conn.execute("UPDATE groups SET name = ? WHERE group_id = ?",
                               (str(new_name), gid))

    def remove_group(self, group: GroupRef, layer: LayerRef | None = None) -> int:
        """Delete a group. Its objects that are in no other group are
        deleted with it; returns how many."""
        with self._tx():
            return self._delete_groups([self._resolve_group(group, layer)])

    def remove_layer(self, layer: LayerRef) -> int:
        """Delete a layer, its groups and its objects; returns the number of
        objects deleted."""
        with self._tx():
            lid = self._resolve_layer(layer)
            gids = [r[0] for r in self._conn.execute(
                "SELECT group_id FROM groups WHERE layer_id = ?", (lid,)).fetchall()]
            n = self._delete_groups(gids)
            self._conn.execute("DELETE FROM layers WHERE layer_id = ?", (lid,))
            return n

    def _delete_groups(self, group_ids: list[int]) -> int:
        if not group_ids:
            return 0
        q = ",".join("?" * len(group_ids))
        self._candidates_reset()
        self._conn.execute(f"INSERT OR IGNORE INTO temp._cand SELECT object_id "
                           f"FROM object_groups WHERE group_id IN ({q})", group_ids)
        self._conn.execute(f"DELETE FROM groups WHERE group_id IN ({q})", group_ids)
        return self._delete_orphan_candidates()

    def _candidates_reset(self) -> None:
        self._conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _cand (object_id INTEGER PRIMARY KEY)")
        self._conn.execute("DELETE FROM temp._cand")

    def _delete_orphan_candidates(self) -> int:
        cur = self._conn.execute(
            "DELETE FROM objects WHERE object_id IN (SELECT object_id FROM temp._cand) "
            "AND NOT EXISTS (SELECT 1 FROM object_groups og "
            "                WHERE og.object_id = objects.object_id)")
        self._conn.execute("DELETE FROM temp._cand")
        return cur.rowcount

    # ------------------------------------------------------------------
    # Objects (write)
    # ------------------------------------------------------------------

    def add_objects(self, objects: Objects, layer: LayerRef = DEFAULT_LAYER,
                    group: GroupRef | Sequence[GroupRef] = DEFAULT_GROUP,
                    data: ScalarData | None = None) -> np.ndarray:
        """Insert objects into one or more groups of a layer (layer and
        groups given by name are created if new). Returns the new object ids.

        ``objects``: annotation objects (``Point``, ``Polygon``, ``Circle``,
        ...) or a GeoDataFrame / GeoSeries. A frame may carry ``type`` (else
        inferred from the geometry), ``name`` and ``circle_r``; CIRCLE rows
        need ``circle_r`` and may hold the center point or the disk.

        ``data``: per-object scalars, aligned with the objects — a sequence
        of dicts (None for none) or a wide pandas/polars frame, one column
        per attribute, null/NaN meaning absent. Defaults to the objects'
        own ``data`` dicts. Objects passed as instances get their ``id`` set.
        """
        codes, names, geoms, radii, objs = _prepare_objects(objects)
        if data is None and objs is not None:
            data = [o.data for o in objs]
        groups = [group] if isinstance(group, (str, int, np.integer)) else list(group)
        if not groups:
            raise ValueError("an object needs at least one group")
        n = codes.size
        with self._tx():
            lid = self._resolve_layer(layer, create=True)
            gids = [self._resolve_group(g, lid, create=True) for g in groups]
            if n == 0:
                return np.empty(0, dtype=np.int64)
            wkb, bounds = _encode_rows(codes, geoms, radii)
            start = self._alloc_ids("objects", "object_id", "last_object_id", n)
            ids = np.arange(start, start + n, dtype=np.int64)
            r_db = [None if np.isnan(r) else float(r) for r in radii]
            self._conn.executemany(
                "INSERT INTO objects(object_id, type_code, name, wkb, circle_r, "
                "min_x, min_y, max_x, max_y) VALUES (?,?,?,?,?,?,?,?,?)",
                zip(ids.tolist(), codes.tolist(), names, wkb, r_db,
                    *bounds.T.tolist()))
            for gid in gids:
                self._conn.executemany(
                    "INSERT INTO object_groups(object_id, group_id) VALUES (?, ?)",
                    ((i, gid) for i in ids.tolist()))
            if data is not None:
                self._write_scalars(ids, data, replace=True)
        if objs is not None:
            for o, i in zip(objs, ids.tolist()):
                o._id = i
        return ids

    def add_object(self, obj: AnnotationObject, layer: LayerRef = DEFAULT_LAYER,
                   group: GroupRef | Sequence[GroupRef] = DEFAULT_GROUP) -> int:
        """Insert one object; returns its id."""
        return int(self.add_objects([obj], layer=layer, group=group)[0])

    def remove_objects(self, object_ids: np.ndarray | list[int]) -> int:
        """Delete objects (with their memberships and scalars); returns how
        many existed."""
        ids = np.asarray(object_ids, dtype=np.int64)
        n = 0
        with self._tx():
            for chunk in _chunked(ids):
                q = ",".join("?" * len(chunk))
                n += self._conn.execute(f"DELETE FROM objects WHERE object_id IN ({q})",
                                        chunk.tolist()).rowcount
        return n

    def update_objects(self, gdf: gpd.GeoDataFrame | gpd.GeoSeries) -> None:
        """Replace the geometry (and, if the columns are present, ``type``,
        ``name``, ``circle_r``) of existing objects. Indexed by object_id.
        Without a ``type`` column, circles stay circles (given as center
        point or disk) and other objects take the type of their new
        geometry. Bounding boxes / R-tree follow automatically."""
        if isinstance(gdf, gpd.GeoSeries):
            gdf = gpd.GeoDataFrame(geometry=gdf)
        ids = gdf.index.to_numpy(dtype=np.int64)
        if ids.size == 0:
            return
        with self._tx():
            old = self._get_type_radius(ids)
            missing = np.setdiff1d(ids, old.index.to_numpy())
            if missing.size:
                raise ValueError(f"no objects with ids {missing[:10].tolist()}"
                                 f"{' ...' if missing.size > 10 else ''}")
            old = old.loc[ids]
            geoms = gdf.geometry.to_numpy()
            if "type" in gdf.columns:
                codes = _type_codes(gdf["type"].astype(str).to_numpy())
            else:
                codes = _type_codes(geometry_type_names(geoms))
                codes[old["type_code"].to_numpy() == _CIRCLE] = _CIRCLE
            radii = (gdf["circle_r"].to_numpy(dtype=np.float64) if "circle_r" in gdf.columns
                     else old["circle_r"].to_numpy(dtype=np.float64))
            radii = np.where(codes == _CIRCLE, radii, np.nan)
            wkb, bounds = _encode_rows(codes, geoms, radii)
            r_db = [None if np.isnan(r) else float(r) for r in radii]
            self._conn.executemany(
                "UPDATE objects SET type_code = ?, wkb = ?, circle_r = ?, "
                "min_x = ?, min_y = ?, max_x = ?, max_y = ? WHERE object_id = ?",
                zip(codes.tolist(), wkb, r_db, *bounds.T.tolist(), ids.tolist()))
            if "name" in gdf.columns:
                self._conn.executemany(
                    "UPDATE objects SET name = ? WHERE object_id = ?",
                    zip([None if v is None or v is pd.NA else str(v) for v in gdf["name"]],
                        ids.tolist()))

    def _get_type_radius(self, ids: np.ndarray) -> pd.DataFrame:
        rows: list[tuple] = []
        for chunk in _chunked(ids):
            q = ",".join("?" * len(chunk))
            rows.extend(self._conn.execute(
                f"SELECT object_id, type_code, circle_r FROM objects "
                f"WHERE object_id IN ({q})", chunk.tolist()).fetchall())
        df = pd.DataFrame(rows, columns=["object_id", "type_code", "circle_r"])
        df["circle_r"] = df["circle_r"].astype(float)
        return df.set_index("object_id")

    def add_to_group(self, object_ids: np.ndarray | list[int], group: GroupRef,
                     layer: LayerRef | None = None) -> None:
        """Add existing objects to a(nother) group of their layer. A group
        given by name together with ``layer`` is created if new. Objects
        cannot join a group of a different layer (ValueError)."""
        ids = np.asarray(object_ids, dtype=np.int64)
        with self._tx():
            gid = self._resolve_group(group, layer,
                                      create=layer is not None and isinstance(group, str))
            self._check_ids_exist(ids)
            try:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO object_groups(object_id, group_id) VALUES (?, ?)",
                    ((i, gid) for i in ids.tolist()))
            except sqlite3.IntegrityError as e:
                raise ValueError(f"cannot add to group {group!r}: {e}") from None

    def remove_from_group(self, object_ids: np.ndarray | list[int], group: GroupRef,
                          layer: LayerRef | None = None) -> int:
        """Take objects out of a group. Objects left in no group are
        deleted; returns how many."""
        ids = np.asarray(object_ids, dtype=np.int64)
        with self._tx():
            gid = self._resolve_group(group, layer)
            self._candidates_reset()
            self._conn.executemany("INSERT OR IGNORE INTO temp._cand VALUES (?)",
                                   ((i,) for i in ids.tolist()))
            self._conn.execute(
                "DELETE FROM object_groups WHERE group_id = ? "
                "AND object_id IN (SELECT object_id FROM temp._cand)", (gid,))
            return self._delete_orphan_candidates()

    def _check_ids_exist(self, ids: np.ndarray) -> None:
        found = 0
        for chunk in _chunked(np.unique(ids)):
            q = ",".join("?" * len(chunk))
            found += self._conn.execute(
                f"SELECT COUNT(*) FROM objects WHERE object_id IN ({q})",
                chunk.tolist()).fetchone()[0]
        if found != np.unique(ids).size:
            raise ValueError(f"{np.unique(ids).size - found} of the object ids do not exist")

    # ------------------------------------------------------------------
    # Sparse scalar attributes (write)
    # ------------------------------------------------------------------

    def set_scalar_data(self, object_ids: np.ndarray | list[int], data: ScalarData,
                        replace: bool = False) -> None:
        """Set per-object scalars. ``data`` is aligned with ``object_ids``:
        a sequence of dicts or a wide pandas/polars frame (null/NaN =
        absent). By default the given attributes are merged into what each
        object already has; ``replace=True`` replaces each object's whole
        vector."""
        ids = np.asarray(object_ids, dtype=np.int64)
        with self._tx():
            self._check_ids_exist(ids)
            self._write_scalars(ids, data, replace=replace)

    def delete_scalar_data(self, object_ids: np.ndarray | list[int],
                           attributes: Sequence[str] | None = None) -> None:
        """Remove all scalars of the objects, or only the named attributes."""
        ids = np.asarray(object_ids, dtype=np.int64)
        with self._tx():
            if attributes is None:
                for chunk in _chunked(ids):
                    q = ",".join("?" * len(chunk))
                    self._conn.execute(
                        f"DELETE FROM object_scalar_sparse WHERE object_id IN ({q})",
                        chunk.tolist())
                return
            drop = np.array([r[0] for r in self._conn.execute(
                f"SELECT attr_id FROM scalar_attributes WHERE attr_name IN "
                f"({','.join('?' * len(attributes))})", list(attributes)).fetchall()],
                dtype=np.uint32)
            old = self._read_sparse(ids)
            rows = []
            for oid, (aid, val) in old.items():
                keep = ~np.isin(aid, drop)
                rows.append((oid, aid[keep], val[keep]))
            self._store_sparse(rows)

    def _attr_ids(self, names: Sequence[str]) -> dict[str, int]:
        """attr_name -> attr_id, adding new names to the dictionary."""
        names = list(dict.fromkeys(str(n) for n in names))
        out: dict[str, int] = {}
        for i in range(0, len(names), 30000):
            part = names[i:i + 30000]
            self._conn.executemany(
                "INSERT OR IGNORE INTO scalar_attributes(attr_name) VALUES (?)",
                ((n,) for n in part))
            q = ",".join("?" * len(part))
            out.update(self._conn.execute(
                f"SELECT attr_name, attr_id FROM scalar_attributes "
                f"WHERE attr_name IN ({q})", part).fetchall())
        return out

    def _read_sparse(self, ids: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for chunk in _chunked(ids):
            q = ",".join("?" * len(chunk))
            for oid, a, v in self._conn.execute(
                    f"SELECT object_id, ids_u32, vals_f32 FROM object_scalar_sparse "
                    f"WHERE object_id IN ({q})", chunk.tolist()):
                out[int(oid)] = (np.frombuffer(a, dtype=np.uint32),
                                 np.frombuffer(v, dtype=np.float32))
        return out

    def _store_sparse(self, rows: Iterable[tuple[int, np.ndarray, np.ndarray]]) -> None:
        """(object_id, attr ids, values) -> rows; empty vectors delete."""
        put, drop = [], []
        for oid, aid, val in rows:
            if aid.size == 0:
                drop.append((oid,))
                continue
            order = np.argsort(aid, kind="stable")
            put.append((oid, aid[order].astype(np.uint32).tobytes(),
                        val[order].astype(np.float32).tobytes()))
        if put:
            self._conn.executemany(
                "INSERT OR REPLACE INTO object_scalar_sparse(object_id, ids_u32, vals_f32) "
                "VALUES (?, ?, ?)", put)
        if drop:
            self._conn.executemany("DELETE FROM object_scalar_sparse WHERE object_id = ?",
                                   drop)

    def _write_scalars(self, ids: np.ndarray, data: ScalarData, replace: bool) -> None:
        new = _sparse_rows(ids, data, self._attr_ids)
        if not replace:
            old = self._read_sparse(ids)
            merged = []
            for oid, aid, val in new:
                if oid in old:
                    oa, ov = old[oid]
                    keep = ~np.isin(oa, aid)
                    aid = np.concatenate([oa[keep], aid])
                    val = np.concatenate([ov[keep], val])
                merged.append((oid, aid, val))
            new = merged
        self._store_sparse(new)

    # ------------------------------------------------------------------
    # Geometric transforms (write, whole store or a subset)
    # ------------------------------------------------------------------

    def translate(self, x_off: float, y_off: float | None = None,
                  object_ids: np.ndarray | list[int] | None = None) -> None:
        """Shift objects (all, or ``object_ids``) by (x_off, y_off)."""
        if y_off is None:
            y_off = x_off
        self._transform(np.eye(2), np.array([x_off, y_off], dtype=float),
                        (0.0, 0.0), "raise", object_ids)

    def scale(self, x_scale: float, y_scale: float | None = None,
              origin: str | tuple[float, float] = (0.0, 0.0),
              object_ids: np.ndarray | list[int] | None = None,
              circles: CircleMode = "raise") -> None:
        """Scale about a fixed point, or each object about its own
        "center" / "centroid". Does not change ``mpp`` — see ``resize``."""
        if y_scale is None:
            y_scale = x_scale
        self._transform(np.diag([float(x_scale), float(y_scale)]), None, origin,
                        circles, object_ids)

    def affine(self, M: Any, object_ids: np.ndarray | list[int] | None = None,
               circles: CircleMode = "raise") -> None:
        """Apply a 2 x 3 affine matrix (row-major, on (x, y, 1)) — e.g. the
        ``transform.matrix`` of an ingestion import record, to bring
        source-slide annotations onto the imported pyramid. Circles under a
        non-similarity: ``circles="raise"`` or ``"to_polygon"``."""
        m = as_affine_matrix(M)
        self._transform(m[:, :2], m[:, 2], (0.0, 0.0), circles, object_ids)

    def resize(self, factor: float) -> None:
        """Rescale all coordinates by ``factor`` about (0, 0) and update
        ``mpp`` (divided by factor) and ``image_shape`` (multiplied,
        rounded) to match."""
        with self._tx():
            self._transform(np.eye(2) * float(factor), None, (0.0, 0.0), "raise", None)
            if self.mpp is not None:
                self._meta_set("mpp", self.mpp / factor)
            shp = self.image_shape
            if shp is not None:
                self._meta_set("image_width", int(np.round(shp.width * factor)))
                self._meta_set("image_height", int(np.round(shp.height * factor)))

    def set_resolution(self, mpp: float) -> None:
        """Rescale the annotation to a new resolution (microns per pixel)."""
        cur = self.mpp
        if cur is None:
            raise ValueError("the store has no mpp; set store.mpp first")
        if mpp != cur:
            with self._tx():
                self.resize(cur / mpp)
                self._meta_set("mpp", float(mpp))

    def match_image(self, raster: "PyramidalRaster", level: int = 0) -> None:
        """Rescale to fit ``raster`` at ``level``: the factor is the mean of
        the width and height ratios; afterwards ``image_shape`` and ``mpp``
        are exactly the raster's."""
        shp = self.image_shape
        if shp is None or shp.width == 0 or shp.height == 0:
            raise ValueError("the store has no image_shape to match from")
        target = raster.shape(level)
        factor = 0.5 * (target.width / shp.width + target.height / shp.height)
        with self._tx():
            self.resize(factor)
            self._meta_set("image_width", target.width)
            self._meta_set("image_height", target.height)
            self._meta_set("mpp", raster.get_mpp_for_level(level))

    def _transform(self, A: np.ndarray, t: np.ndarray | None,
                   origin: str | tuple[float, float], circles: CircleMode,
                   object_ids: np.ndarray | list[int] | None) -> None:
        with self._tx():
            if circles == "raise" and similarity_scale(A) is None:
                n = self._count_circles(object_ids)
                if n:
                    raise ValueError(
                        f"the transform is not a similarity: it would deform {n} "
                        f"circle(s). Pass circles='to_polygon' to convert them.")
            for rows in self._iter_geometry_rows(object_ids):
                oids = np.array([r[0] for r in rows], dtype=np.int64)
                codes = np.array([r[1] for r in rows], dtype=np.int64)
                wkbs = np.empty(len(rows), dtype=object)
                wkbs[:] = [r[2] for r in rows]
                radii = np.array([np.nan if r[3] is None else r[3] for r in rows],
                                 dtype=np.float64)
                geoms, radii, conv = transform_rows(
                    shapely.from_wkb(wkbs), radii, codes == _CIRCLE, A, t, origin, circles)
                codes[conv] = ANNOT_TYPE_CODE["POLYGON"]
                wkb, bounds = _encode_rows(codes, geoms, radii)
                r_db = [None if np.isnan(r) else float(r) for r in radii]
                self._conn.executemany(
                    "UPDATE objects SET type_code = ?, wkb = ?, circle_r = ?, "
                    "min_x = ?, min_y = ?, max_x = ?, max_y = ? WHERE object_id = ?",
                    zip(codes.tolist(), wkb, r_db, *bounds.T.tolist(), oids.tolist()))

    def _count_circles(self, object_ids: np.ndarray | list[int] | None) -> int:
        if object_ids is None:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM objects WHERE type_code = ?", (_CIRCLE,)).fetchone()[0])
        n = 0
        for chunk in _chunked(np.asarray(object_ids, dtype=np.int64)):
            q = ",".join("?" * len(chunk))
            n += self._conn.execute(
                f"SELECT COUNT(*) FROM objects WHERE type_code = ? AND object_id IN ({q})",
                [_CIRCLE, *chunk.tolist()]).fetchone()[0]
        return int(n)

    def _iter_geometry_rows(self, object_ids: np.ndarray | list[int] | None
                            ) -> Iterator[list[tuple]]:
        cols = "object_id, type_code, wkb, circle_r"
        if object_ids is None:
            last = None
            while True:
                if last is None:
                    rows = self._conn.execute(
                        f"SELECT {cols} FROM objects ORDER BY object_id LIMIT ?",
                        (_TRANSFORM_BATCH,)).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT {cols} FROM objects WHERE object_id > ? "
                        f"ORDER BY object_id LIMIT ?", (last, _TRANSFORM_BATCH)).fetchall()
                if not rows:
                    return
                yield rows
                last = rows[-1][0]
        else:
            for chunk in _chunked(np.unique(np.asarray(object_ids, dtype=np.int64))):
                q = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT {cols} FROM objects WHERE object_id IN ({q})",
                    chunk.tolist()).fetchall()
                if rows:
                    yield rows

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_gdf() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"type": pd.Series(dtype="string"),
             "name": pd.Series(dtype="string"),
             "circle_r": pd.Series(dtype="float64")},
            geometry=gpd.GeoSeries([]),
            index=pd.Index([], dtype="int64", name="object_id"))

    @staticmethod
    def _decode(rows: list[tuple], circles_as_disks: bool) -> gpd.GeoDataFrame:
        """Decode (object_id, type_code, name, wkb, circle_r) rows into a
        GeoDataFrame in one vectorized pass."""
        if not rows:
            return AnnotationStore._empty_gdf()

        oids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        codes = np.fromiter((r[1] for r in rows), dtype=np.int16, count=len(rows))
        names = pd.array([r[2] for r in rows], dtype="string")
        wkbs = np.empty(len(rows), dtype=object)
        wkbs[:] = [r[3] for r in rows]
        radii = np.array([np.nan if r[4] is None else float(r[4]) for r in rows],
                         dtype=np.float64)

        geoms = shapely.from_wkb(wkbs)

        # Circles are stored as center POINT + radius; optionally decode to
        # polygonal disks so the frame is uniformly drawable.
        is_circle = codes == ANNOT_TYPE_CODE["CIRCLE"]
        if circles_as_disks and is_circle.any():
            geoms[is_circle] = shapely.buffer(geoms[is_circle], radii[is_circle])

        types = pd.Categorical.from_codes(
            codes - 1, categories=[ANNOT_CODE_TYPE[c] for c in sorted(ANNOT_CODE_TYPE)])

        # The GeoSeries must carry the object_id index itself: the
        # GeoDataFrame constructor aligns it by label, not by position.
        index = pd.Index(oids, name="object_id")
        return gpd.GeoDataFrame(
            {"type": types, "name": names, "circle_r": radii},
            geometry=gpd.GeoSeries(geoms, index=index),
            index=index)
##


def _chunked(ids: np.ndarray, size: int = 30000):
    """Yield id chunks below SQLite's bound-parameter limit (32766 since 3.32)."""
    for i in range(0, ids.size, size):
        yield ids[i:i + size]


def _remove_db_files(path: Path) -> None:
    """Remove a database file and its WAL/SHM/journal companions."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(path) + suffix)
        if p.exists():
            os.remove(p)


def _type_codes(types: np.ndarray) -> np.ndarray:
    try:
        return np.array([ANNOT_TYPE_CODE[str(t).upper()] for t in types], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"unknown annotation type {e.args[0]!r}; expected one of "
                         f"{', '.join(ANNOT_TYPE_CODE)}") from None


def _prepare_objects(objects: Objects) -> tuple[np.ndarray, list, np.ndarray, np.ndarray,
                                                 list[AnnotationObject] | None]:
    """-> (type codes, names, geometries, radii, objects-or-None). Circles:
    geometry is the center point or the disk, radius in ``radii``."""
    if isinstance(objects, gpd.GeoSeries):
        objects = gpd.GeoDataFrame(geometry=objects)
    if isinstance(objects, gpd.GeoDataFrame):
        geoms = objects.geometry.to_numpy()
        types = (objects["type"].astype(str).to_numpy() if "type" in objects.columns
                 else geometry_type_names(geoms))
        codes = _type_codes(types)
        names = ([None if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))
                  else str(v) for v in objects["name"]]
                 if "name" in objects.columns else [None] * len(objects))
        radii = (objects["circle_r"].to_numpy(dtype=np.float64)
                 if "circle_r" in objects.columns else np.full(len(objects), np.nan))
        objs = None
    else:
        objs = [objects] if isinstance(objects, AnnotationObject) else list(objects)
        for o in objs:
            if not isinstance(o, AnnotationObject):
                raise TypeError(f"expected AnnotationObject instances, got {type(o).__name__}")
        codes = _type_codes(np.array([o.type for o in objs], dtype=object))
        names = [o.name for o in objs]
        geoms = np.empty(len(objs), dtype=object)
        geoms[:] = [shapely.Point(o.center) if isinstance(o, Circle) else o.geom
                    for o in objs]
        radii = np.array([o.radius if isinstance(o, Circle) else np.nan for o in objs],
                         dtype=np.float64)
    is_c = codes == _CIRCLE
    bad = is_c & ~(radii >= 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} CIRCLE row(s) without a valid circle_r")
    radii = np.where(is_c, radii, np.nan)
    return codes, names, geoms, radii, objs


def _encode_rows(codes: np.ndarray, geoms: np.ndarray, radii: np.ndarray
                 ) -> tuple[list[bytes], np.ndarray]:
    """Geometries -> (WKB list, (n, 4) bboxes). Circles are stored as their
    center point; their bbox includes the radius."""
    geoms = np.array(geoms, dtype=object, copy=True)
    is_c = codes == _CIRCLE
    if is_c.any():
        not_pt = is_c & (shapely.get_type_id(geoms) != 0)
        if not_pt.any():  # disk given -> center
            geoms[not_pt] = shapely.centroid(geoms[not_pt])
    if shapely.is_missing(geoms).any() or shapely.is_empty(geoms).any():
        raise ValueError("missing or empty geometries cannot be stored")
    bounds = shapely.bounds(geoms)
    if is_c.any():
        r = radii[is_c][:, None]
        bounds[is_c] = bounds[is_c] + np.hstack([-r, -r, r, r])
    return list(shapely.to_wkb(geoms)), bounds


def _sparse_rows(ids: np.ndarray, data: ScalarData, attr_ids
                 ) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Aligned per-object scalars -> [(object_id, attr ids, values)]."""
    n = ids.size
    if isinstance(data, (pd.DataFrame, pl.DataFrame)):
        if len(data) != n:
            raise ValueError(f"data has {len(data)} rows for {n} objects")
        cols = [str(c) for c in data.columns]
        if isinstance(data, pd.DataFrame):
            V = data.to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            V = data.select(pl.all().cast(pl.Float32)).to_numpy()
        name_id = attr_ids(cols)
        col_ids = np.array([name_id[c] for c in cols], dtype=np.uint32)
        present = ~np.isnan(V)
        return [(int(oid), col_ids[present[i]], V[i, present[i]])
                for i, oid in enumerate(ids.tolist())]

    rows = list(data)
    if len(rows) != n:
        raise ValueError(f"data has {len(rows)} entries for {n} objects")
    keys = [k for d in rows if d for k in d]
    name_id = attr_ids(keys) if keys else {}
    out = []
    for oid, d in zip(ids.tolist(), rows):
        aid, val = [], []
        for k, v in (d or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                logger.warning("object %d: attribute %r = %r is not numeric; skipped",
                               oid, k, v)
                continue
            aid.append(name_id[str(k)])
            val.append(fv)
        out.append((int(oid), np.asarray(aid, dtype=np.uint32),
                    np.asarray(val, dtype=np.float32)))
    return out
