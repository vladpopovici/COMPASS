# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Physical schema of the SQLite/R-tree annotation database.

Unchanged from ``compass.legacy.annot_sqlite_storage`` (a test checks the
DDL text), so files written by the legacy code and by
``compass.core.annot.AnnotationStore`` are interchangeable.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import sqlite3
from pathlib import Path

import numpy as np
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


def geometry_type_names(geoms: np.ndarray) -> np.ndarray:
    """Annotation type names inferred from shapely geometries (vectorized
    over ``shapely.get_type_id``). Raises ValueError for unsupported kinds
    (e.g. GeometryCollection) or missing geometries."""
    geoms = np.asarray(geoms, dtype=object)
    ids = shapely.get_type_id(geoms)
    # shapely type ids: 0 Point, 1 LineString, 2 LinearRing, 3 Polygon,
    # 4 MultiPoint, 5 MultiLineString, 6 MultiPolygon, 7 GeometryCollection
    lut = np.array(["POINT", "POLYLINE", "POLYLINE", "POLYGON", "POINTSET",
                    "POLYLINE", "POLYGON", ""], dtype=object)
    bad = (ids < 0) | (ids > 6)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"unsupported geometry at row {i}: {geoms[i]!r}")
    return lut[ids]
