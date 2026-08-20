# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Vectorized read access to the SQLite/R-tree annotation store.

The physical schema (see ``compass.legacy.annot_sqlite_storage``) is kept
as-is: WKB geometries + bbox columns in ``objects``, an ``objects_rtree``
R-tree virtual table, a layer -> group -> object hierarchy, and sparse
per-object scalar attributes (``object_scalar_sparse``: parallel uint32
attribute-id / float32 value blobs, with a shared ``scalar_attributes``
name<->id dictionary).

What changes relative to the legacy access layer (``compass.legacy.annot``):
query results come back as **geopandas GeoDataFrames** (geometry decoded in
one vectorized ``shapely.from_wkb`` pass) or **polars DataFrames** — never
as one Python object per annotation. At millions of objects the per-object
materialization is the bottleneck, not SQLite.

Usage
-----
    store = AnnotationStore("slide_001.ann.sqlite")
    gdf = store.get_objects_in_roi(layer_id=0, roi=(x0, y0, x1, y1))
    counts = store.count_objects_per_group()
    expr = store.get_scalar_data(gdf.index.to_numpy(), attributes=["CD3", "CD8"])
"""

import sqlite3
from pathlib import Path
from types import TracebackType

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import shapely

# Geometry type codes, identical to the legacy store
# (compass.legacy.annot.ANNOT_TYPE_CODE) for on-disk compatibility.
ANNOT_TYPE_CODE: dict[str, int] = {
    "POINT": 1,
    "POINTSET": 2,
    "POLYLINE": 3,
    "POLYGON": 4,
    "CIRCLE": 5,
}
ANNOT_CODE_TYPE: dict[int, str] = {v: k for k, v in ANNOT_TYPE_CODE.items()}

SCHEMA_VERSION = "1"

# Physical schema — byte-for-byte the SQLITE_DDL of
# compass.legacy.annot_sqlite_storage (checked by a test); the legacy writer
# and this reader must keep agreeing on it.
SQLITE_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS layers (
  layer_id INTEGER PRIMARY KEY,
  name     TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS groups (
  group_id INTEGER PRIMARY KEY,
  layer_id INTEGER NOT NULL REFERENCES layers(layer_id) ON DELETE CASCADE,
  name     TEXT NOT NULL,
  UNIQUE(layer_id, name)
);
CREATE INDEX IF NOT EXISTS idx_groups_layer ON groups(layer_id);

CREATE TABLE IF NOT EXISTS objects (
  object_id  INTEGER PRIMARY KEY,
  type_code  INTEGER NOT NULL,
  name       TEXT,
  wkb        BLOB    NOT NULL,
  circle_r   REAL,
  min_x      REAL    NOT NULL,
  min_y      REAL    NOT NULL,
  max_x      REAL    NOT NULL,
  max_y      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(type_code);

CREATE TABLE IF NOT EXISTS object_groups (
  object_id INTEGER NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
  group_id  INTEGER NOT NULL REFERENCES groups(group_id)  ON DELETE CASCADE,
  PRIMARY KEY (object_id, group_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_object_groups_group  ON object_groups(group_id);
CREATE INDEX IF NOT EXISTS idx_object_groups_object ON object_groups(object_id);

CREATE TRIGGER IF NOT EXISTS trg_object_groups_one_layer
BEFORE INSERT ON object_groups
BEGIN
  SELECT CASE
    WHEN EXISTS (
      SELECT 1
      FROM object_groups og
      JOIN groups g_existing ON g_existing.group_id = og.group_id
      JOIN groups g_new      ON g_new.group_id      = NEW.group_id
      WHERE og.object_id = NEW.object_id
        AND g_existing.layer_id != g_new.layer_id
    )
    THEN RAISE(ABORT, 'Object cannot belong to groups from multiple layers')
  END;
END;

CREATE VIRTUAL TABLE IF NOT EXISTS objects_rtree USING rtree(
  object_id,
  min_x, max_x,
  min_y, max_y
);

CREATE TRIGGER IF NOT EXISTS trg_objects_insert AFTER INSERT ON objects
BEGIN
  INSERT INTO objects_rtree(object_id, min_x, max_x, min_y, max_y)
  VALUES (NEW.object_id, NEW.min_x, NEW.max_x, NEW.min_y, NEW.max_y);
END;

CREATE TRIGGER IF NOT EXISTS trg_objects_delete AFTER DELETE ON objects
BEGIN
  DELETE FROM objects_rtree WHERE object_id = OLD.object_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_objects_bbox_update
AFTER UPDATE OF min_x, max_x, min_y, max_y ON objects
BEGIN
  UPDATE objects_rtree
  SET min_x = NEW.min_x, max_x = NEW.max_x, min_y = NEW.min_y, max_y = NEW.max_y
  WHERE object_id = NEW.object_id;
END;

CREATE TABLE IF NOT EXISTS scalar_attributes (
  attr_id   INTEGER PRIMARY KEY,
  attr_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS object_scalar_sparse (
  object_id INTEGER PRIMARY KEY REFERENCES objects(object_id) ON DELETE CASCADE,
  ids_u32   BLOB NOT NULL,
  vals_f32  BLOB NOT NULL
);
"""


def init_db(path: str | Path) -> None:
    """Create / initialize the SQLite annotation database schema."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SQLITE_DDL)
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()

_OBJECT_COLUMNS = "o.object_id, o.type_code, o.name, o.wkb, o.circle_r"


#####
class AnnotationStore:
    """Read-oriented, vectorized access to one SQLite annotation database.

    The store is opened read-only and can be used as a context manager. All
    bulk queries return geopandas/polars frames indexed by ``object_id``;
    no per-annotation Python objects are created.

    For circles the stored geometry is the center POINT plus a ``circle_r``
    column; by default they are decoded to polygonal disks
    (``point.buffer(r)``) so a returned GeoDataFrame is uniformly drawable.
    Pass ``circles_as_disks=False`` to keep the raw center points (the
    ``circle_r`` column always carries the radius either way).
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise ValueError(f"no such annotation database: {path}")
        # Read-only URI: the viewer never writes annotations.
        self._conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        self._conn.execute("PRAGMA query_only = ON;")

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
    def path(self) -> Path:
        return self._path

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
