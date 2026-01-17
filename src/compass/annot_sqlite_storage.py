# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""
annot_sqlite_storage.py

SQLite-backed persistence, retrieval, and querying for the Annotation / AnnotationObject
structures.

Usage
-----
from annot_sqlite_storage import (
    init_db,
    save_annotation,
    load_annotation,
    query_object_ids_in_layer,
    query_object_ids_in_group,
    query_object_ids_in_layer_roi,
    fetch_objects,
)

init_db("slide_001.ann.sqlite")

save_annotation(ann, "slide_001.ann.sqlite")
ann2 = load_annotation("slide_001.ann.sqlite")

ids = query_object_ids_in_layer("slide_001.ann.sqlite", layer_id=0)
objs = fetch_objects("slide_001.ann.sqlite", ids)

"""
import sqlite3
from pathlib import Path

import numpy as np
import shapely.geometry as shg
import shapely.wkb as shapely_wkb

from .annot import (
    Annotation,
    AnnotationObject,
    Circle,
    createEmptyAnnotationObject,
    ANNOT_TYPE_CODE,
    ANNOT_CODE_TYPE,
)
from .core import ImageShape

SCHEMA_VERSION = "1"

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


# =============================================================================
# Connection / meta helpers
# =============================================================================

def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    return conn


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _meta_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def init_db(path: str | Path) -> None:
    """
    Create / initialize the SQLite annotation database schema.

    :param path: Path to the annotation database
    """
    conn = _connect(path)
    try:
        conn.executescript(SQLITE_DDL)
        _meta_set(conn, "schema_version", SCHEMA_VERSION)
        conn.commit()
    finally:
        conn.close()


# =============================================================================
# Scalar attribute dictionary + sparse packing
# =============================================================================

def _ensure_scalar_attribute_ids(conn: sqlite3.Connection, attr_names: list[str]) -> dict[str, int]:
    names = list(dict.fromkeys(attr_names))
    if not names:
        return {}

    qmarks = ",".join(["?"] * len(names))
    existing = dict(
        conn.execute(
            f"SELECT attr_name, attr_id FROM scalar_attributes WHERE attr_name IN ({qmarks})",
            names,
        ).fetchall()
    )

    missing = [n for n in names if n not in existing]
    if missing:
        conn.executemany(
            "INSERT OR IGNORE INTO scalar_attributes(attr_name) VALUES (?)",
            [(n,) for n in missing],
        )
        existing = dict(
            conn.execute(
                f"SELECT attr_name, attr_id FROM scalar_attributes WHERE attr_name IN ({qmarks})",
                names,
            ).fetchall()
        )

    return {str(k): int(v) for k, v in existing.items()}


def _pack_sparse(data: dict[str, object] | None, name_to_id: dict[str, int]) -> tuple[bytes, bytes]:
    """
    Pack dict[str,scalar] -> (uint32 attr_ids blob, float32 values blob)
    Non-numeric values are ignored.
    """
    if not data:
        return np.array([], np.uint32).tobytes(), np.array([], np.float32).tobytes()

    ids: list[int] = []
    vals: list[float] = []

    for k, v in data.items():
        aid = name_to_id.get(str(k))
        if aid is None:
            continue
        try:
            fv = float(v)  # enforce numeric scalar
        except Exception:
            continue
        ids.append(aid)
        vals.append(fv)

    if not ids:
        return np.array([], np.uint32).tobytes(), np.array([], np.float32).tobytes()

    ids_arr = np.asarray(ids, dtype=np.uint32)
    vals_arr = np.asarray(vals, dtype=np.float32)

    order = np.argsort(ids_arr)
    ids_arr = ids_arr[order]
    vals_arr = vals_arr[order]

    return ids_arr.tobytes(order="C"), vals_arr.tobytes(order="C")


def _unpack_sparse(conn: sqlite3.Connection, ids_blob: bytes, vals_blob: bytes) -> dict[str, float]:
    ids = np.frombuffer(ids_blob, dtype=np.uint32)
    vals = np.frombuffer(vals_blob, dtype=np.float32)
    if ids.size == 0:
        return {}

    qmarks = ",".join(["?"] * int(ids.size))
    rows = conn.execute(
        f"SELECT attr_id, attr_name FROM scalar_attributes WHERE attr_id IN ({qmarks})",
        [int(x) for x in ids],
    ).fetchall()
    id_to_name = {int(aid): str(name) for aid, name in rows}

    out: dict[str, float] = {}
    for aid, v in zip(ids.tolist(), vals.tolist()):
        nm = id_to_name.get(int(aid))
        if nm is not None:
            out[nm] = float(v)
    return out


# =============================================================================
# Geometry encoding/decoding (WKB) with Circle fidelity
# =============================================================================

def _encode_geometry(obj) -> tuple[int, bytes, float | None, tuple[float, float, float, float]]:
    """
    Return (type_code, wkb, circle_r, bbox).
    For Circle: store center as POINT WKB + circle_r.
    For others: store obj.geom WKB.
    """
    type_str = obj.type
    type_code = ANNOT_TYPE_CODE[type_str]

    if type_str == "CIRCLE":
        cx, cy = obj.center
        r = float(obj.radius)
        wkb = shapely_wkb.dumps(shg.Point(cx, cy))
        bbox = (cx - r, cy - r, cx + r, cy + r)
        return type_code, wkb, r, bbox

    wkb = shapely_wkb.dumps(obj.geom)
    bbox = obj.geom.bounds
    return type_code, wkb, None, bbox


def _decode_geometry(obj, type_code: int, wkb: bytes, circle_r: float | None) -> None:
    type_str = ANNOT_CODE_TYPE[int(type_code)]

    if type_str == "CIRCLE":
        center_pt = shapely_wkb.loads(wkb)
        cx, cy = float(center_pt.x), float(center_pt.y)
        r = float(circle_r) if circle_r is not None else 0.0

        rebuilt = Circle(center=(cx, cy), radius=r, name=obj.name, data=obj.data)
        obj._geom = rebuilt.geom
        obj._center = rebuilt._center
        obj._radius = rebuilt._radius
        obj._annotation_type = "CIRCLE"
        return

    geom = shapely_wkb.loads(wkb)
    obj._geom = geom
    obj._annotation_type = type_str


# =============================================================================
# Save / Load
# =============================================================================

def save_annotation(annotation, path: str | Path, *, overwrite: bool = True) -> None:
    """
    Save an Annotation instance to SQLite.
    :param annotation: Annotation instance
    :param path: file name to save the annotation to.
    :param overwrite: if True clears existing slide-local content
        (layers, groups, objects, memberships, sparse data), while
        keeping the global scalar_attributes vocabulary
    """
    conn = _connect(path)
    try:
        conn.executescript(SQLITE_DDL)
        _meta_set(conn, "schema_version", SCHEMA_VERSION)

        with conn:
            if overwrite:
                conn.execute("DELETE FROM object_scalar_sparse;")
                conn.execute("DELETE FROM object_groups;")
                conn.execute("DELETE FROM objects;")
                conn.execute("DELETE FROM groups;")
                conn.execute("DELETE FROM layers;")

            # meta
            _meta_set(conn, "name", str(annotation.name))
            _meta_set(conn, "image_width", str(annotation.get_base_image_shape().width))
            _meta_set(conn, "image_height", str(annotation.get_base_image_shape().height))
            _meta_set(conn, "mpp", str(annotation.get_resolution()))

            d = annotation.asdict()
            id_reg = d["id_reg"]
            _meta_set(conn, "last_layer_id", str(id_reg["layer"]))
            _meta_set(conn, "last_group_id", str(id_reg["group"]))
            _meta_set(conn, "last_object_id", str(id_reg["object"]))

            # layers
            conn.executemany(
                "INSERT INTO layers(layer_id, name) VALUES (?, ?)",
                [(int(lid), str(nm)) for lid, nm in annotation._id_layer.items()],
            )

            # group -> layer mapping derived from annotation._layers
            group_to_layer: dict[int, int] = {}
            for lid, gids in annotation._layers.items():
                for gid in gids:
                    group_to_layer[int(gid)] = int(lid)

            # groups
            conn.executemany(
                "INSERT INTO groups(group_id, layer_id, name) VALUES (?, ?, ?)",
                [
                    (int(gid), int(group_to_layer[int(gid)]), str(nm))
                    for gid, nm in annotation._id_group.items()
                ],
            )

            # unique objects by id, preserving shared references
            obj_by_id: dict[int, object] = {}
            for gid, objs in annotation._annots.items():
                for obj in objs:
                    obj_by_id[int(obj._id)] = obj

            # scalar attribute dictionary for all keys encountered
            all_attr_names: list[str] = []
            for obj in obj_by_id.values():
                if obj.data:
                    all_attr_names.extend([str(k) for k in obj.data.keys()])
            name_to_id = _ensure_scalar_attribute_ids(conn, all_attr_names)

            # objects + sparse data
            object_rows: list[tuple] = []
            sparse_rows: list[tuple] = []

            for oid, obj in obj_by_id.items():
                type_code, wkb, circle_r, (minx, miny, maxx, maxy) = _encode_geometry(obj)
                object_rows.append(
                    (
                        int(oid),
                        int(type_code),
                        obj.name,
                        sqlite3.Binary(wkb),
                        circle_r,
                        float(minx),
                        float(miny),
                        float(maxx),
                        float(maxy),
                    )
                )

                ids_blob, vals_blob = _pack_sparse(obj.data, name_to_id)
                sparse_rows.append((int(oid), sqlite3.Binary(ids_blob), sqlite3.Binary(vals_blob)))

            conn.executemany(
                "INSERT INTO objects(object_id, type_code, name, wkb, circle_r, min_x, min_y, max_x, max_y) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                object_rows,
            )
            conn.executemany(
                "INSERT INTO object_scalar_sparse(object_id, ids_u32, vals_f32) VALUES (?,?,?)",
                sparse_rows,
            )

            # memberships
            membership_rows: list[tuple[int, int]] = []
            for gid, objs in annotation._annots.items():
                for obj in objs:
                    membership_rows.append((int(obj._id), int(gid)))

            conn.executemany(
                "INSERT OR IGNORE INTO object_groups(object_id, group_id) VALUES (?,?)",
                membership_rows,
            )

    finally:
        conn.close()


def load_annotation(path: str | Path):
    """
    Load an Annotation from SQLite, reconstructing:
      - id registries
      - layers/groups mappings
      - group lists containing shared object references
      - object geometries (Circle fidelity preserved)
      - object scalar attributes (dict[str,float])

    :param path: path to Annotation file
    """
    conn = _connect(path)
    try:
        conn.executescript(SQLITE_DDL)

        name = _meta_get(conn, "name", "") or ""
        w = int(_meta_get(conn, "image_width", "0") or "0")
        h = int(_meta_get(conn, "image_height", "0") or "0")
        mpp = float(_meta_get(conn, "mpp", "1.0") or "1.0")

        # Create and initialize a basic Annotation object. This ensures some assumptions are met:
        # - there is a default "base" layer (ID:0)
        # - there is a default "no_group" group (ID:0) in the base layer
        ann = Annotation(name=name, image_shape=ImageShape(width=w, height=h), mpp=mpp)

        # layers
        layers = conn.execute("SELECT layer_id, name FROM layers ORDER BY layer_id").fetchall()
        # add layers
        for lid, nm in layers:
            ann._id_layer[int(lid)] = str(nm)

        # groups
        groups = conn.execute("SELECT group_id, name FROM groups ORDER BY group_id").fetchall()
        # add groups
        for gid, nm in groups:
            ann._id_group[int(gid)] = str(nm)


        # _layers: derive from groups table (there is always a default one!)
        for lid, gid in conn.execute("SELECT layer_id, group_id FROM groups").fetchall():
            ann._layers.setdefault(int(lid), set()).add(int(gid))

        # id register
        ann._update_register("layer", int(_meta_get(conn, "last_layer_id", "0") or "0"))
        ann._update_register("group", int(_meta_get(conn, "last_group_id", "0") or "0"))
        ann._update_register("object", int(_meta_get(conn, "last_object_id", "-1") or "-1"))

        # instantiate objects once
        obj_rows = conn.execute(
            "SELECT object_id, type_code, name, wkb, circle_r FROM objects ORDER BY object_id"
        ).fetchall()

        obj_by_id: dict[int, object] = {}
        for oid, type_code, obj_name, wkb_blob, circle_r in obj_rows:
            type_str = ANNOT_CODE_TYPE[int(type_code)]
            obj = createEmptyAnnotationObject(type_str)
            obj._id = int(oid)
            if obj_name is not None:
                obj._name = str(obj_name)
            obj._data = {}
            _decode_geometry(obj, int(type_code), bytes(wkb_blob), circle_r)
            obj_by_id[int(oid)] = obj

        # attach sparse scalar data
        for oid, ids_blob, vals_blob in conn.execute(
            "SELECT object_id, ids_u32, vals_f32 FROM object_scalar_sparse"
        ).fetchall():
            obj_by_id[int(oid)]._data = _unpack_sparse(conn, bytes(ids_blob), bytes(vals_blob))

        # memberships -> ann._annots
        for gid, oid in conn.execute(
            "SELECT group_id, object_id FROM object_groups ORDER BY group_id, object_id"
        ).fetchall():
            ann._annots.setdefault(int(gid), []).append(obj_by_id[int(oid)])

        return ann

    finally:
        conn.close()


# =============================================================================
# Query helpers (IDs first; fetch payloads separately)
# =============================================================================

def query_object_ids_in_layer(path: str | Path, layer_id: int) -> list[int]:
    """
    Return distinct object_ids that belong to any group in the given layer.
    """
    conn = _connect(path)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT o.object_id
            FROM groups g
            JOIN object_groups og ON og.group_id = g.group_id
            JOIN objects o        ON o.object_id = og.object_id
            WHERE g.layer_id = ?
            ORDER BY o.object_id
            """,
            (int(layer_id),),
        ).fetchall()
        return [int(r[0]) for r in rows]
    finally:
        conn.close()


def query_object_ids_in_group(path: str | Path, group_id: int) -> list[int]:
    """
    Return object_ids in a specific group.
    """
    conn = _connect(path)
    try:
        rows = conn.execute(
            """
            SELECT og.object_id
            FROM object_groups og
            WHERE og.group_id = ?
            ORDER BY og.object_id
            """,
            (int(group_id),),
        ).fetchall()
        return [int(r[0]) for r in rows]
    finally:
        conn.close()


def query_object_ids_in_layer_roi(
    path: str | Path,
    layer_id: int,
    roi: tuple[float, float, float, float],
) -> list[int]:
    """
    ROI query in base pixel coordinates. roi = (x0, y0, x1, y1).
    Returns distinct object_ids in the layer whose bbox intersects the ROI.
    """
    x0, y0, x1, y1 = roi
    conn = _connect(path)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT o.object_id
            FROM objects_rtree r
            JOIN objects o        ON o.object_id = r.object_id
            JOIN object_groups og ON og.object_id = o.object_id
            JOIN groups g         ON g.group_id = og.group_id
            WHERE g.layer_id = ?
              AND r.min_x <= ? AND r.max_x >= ?
              AND r.min_y <= ? AND r.max_y >= ?
            ORDER BY o.object_id
            """,
            (int(layer_id), float(x1), float(x0), float(y1), float(y0)),
        ).fetchall()
        return [int(r[0]) for r in rows]
    finally:
        conn.close()


def query_groups_in_layer(path: str | Path, layer_id: int) -> list[tuple[int, str]]:
    """
    Return [(group_id, group_name), ...] for a layer.
    """
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT group_id, name FROM groups WHERE layer_id=? ORDER BY group_id",
            (int(layer_id),),
        ).fetchall()
        return [(int(gid), str(name)) for gid, name in rows]
    finally:
        conn.close()


# =============================================================================
# Fetch helpers (materialize objects by id)
# =============================================================================

def fetch_objects(path: str | Path, object_ids: list[int]) -> list[object]:
    """
    Fetch objects (AnnotationObject instances) for the provided IDs.
    This does not reconstruct group membership; it returns independent objects.

    If you need full Annotation structures with shared references across groups,
    use load_annotation(...).
    """
    if not object_ids:
        return []

    conn = _connect(path)
    try:
        qmarks = ",".join(["?"] * len(object_ids))

        obj_rows = conn.execute(
            f"""
            SELECT object_id, type_code, name, wkb, circle_r
            FROM objects
            WHERE object_id IN ({qmarks})
            ORDER BY object_id
            """,
            [int(x) for x in object_ids],
        ).fetchall()

        # sparse payload
        sparse_rows = dict(
            conn.execute(
                f"""
                SELECT object_id, ids_u32, vals_f32
                FROM object_scalar_sparse
                WHERE object_id IN ({qmarks})
                """,
                [int(x) for x in object_ids],
            ).fetchall()
        )

        out: list[object] = []
        for oid, type_code, obj_name, wkb_blob, circle_r in obj_rows:
            type_str = ANNOT_CODE_TYPE[int(type_code)]
            obj = createEmptyAnnotationObject(type_str)
            obj._id = int(oid)
            if obj_name is not None:
                obj._name = str(obj_name)

            # data
            ids_blob_vals = sparse_rows.get(int(oid))
            if ids_blob_vals is None:
                obj._data = {}
            else:
                ids_blob, vals_blob = ids_blob_vals
                obj._data = _unpack_sparse(conn, bytes(ids_blob), bytes(vals_blob))

            _decode_geometry(obj, int(type_code), bytes(wkb_blob), circle_r)
            out.append(obj)

        return out

    finally:
        conn.close()
