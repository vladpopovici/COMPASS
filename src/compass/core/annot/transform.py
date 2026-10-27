# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Vectorized geometric transforms for annotation GeoDataFrames.

The frame-level functions (``translate``, ``scale``, ``resize``,
``affine``) take a GeoDataFrame with the columns ``AnnotationStore``
returns (``geometry``, and optionally ``type`` / ``circle_r``) and return a
transformed copy. All coordinates are transformed in one numpy pass over
``shapely.get_coordinates`` — no per-object Python loop.

Affine matrices are 2 x 3, row-major, acting on (x, y, 1) — the same
convention as the ``transform.matrix`` of an ingestion import record
(``compass.connector.ingest``), so annotations drawn on a source slide
can be mapped onto its Zarr pyramid with
``affine(gdf, record["transform"]["matrix"])``.

Circles (``type == "CIRCLE"``) stay circles under similarity transforms
(shift, rotation, uniform scale, reflection): the geometry is transformed
and ``circle_r`` is scaled. Any other transform would deform them; then
``circles="raise"`` (default) raises ValueError and
``circles="to_polygon"`` turns them into POLYGON disks first.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

__all__ = ["as_affine_matrix", "similarity_scale", "transform_geometries",
           "translate", "scale", "resize", "affine"]

CircleMode = Literal["raise", "to_polygon"]
Origin = str | tuple[float, float]


def as_affine_matrix(M: Any) -> np.ndarray:
    """Normalize a 2 x 3 (or 3 x 3 homogeneous) matrix to a float 2 x 3."""
    m = np.asarray(M, dtype=np.float64)
    if m.shape == (3, 3):
        if not np.allclose(m[2], [0.0, 0.0, 1.0]):
            raise ValueError("a 3 x 3 matrix must be affine (last row 0, 0, 1)")
        m = m[:2]
    if m.shape != (2, 3):
        raise ValueError(f"expected a 2 x 3 affine matrix, got shape {m.shape}")
    return m


def similarity_scale(A: np.ndarray, rtol: float = 1e-9) -> float | None:
    """The scale factor s if the 2 x 2 linear part A is s * (rotation or
    reflection), else None."""
    a, b, c, d = float(A[0, 0]), float(A[0, 1]), float(A[1, 0]), float(A[1, 1])
    n1, n2 = a * a + c * c, b * b + d * d
    tol = rtol * max(n1, n2, 1e-300)
    if abs(n1 - n2) <= tol and abs(a * b + c * d) <= tol:
        return float(np.sqrt(n1))
    return None


def transform_geometries(geoms: np.ndarray, A: np.ndarray,
                         t: np.ndarray | None = None,
                         origins: np.ndarray | None = None) -> np.ndarray:
    """Apply p' = A (p - o) + o + t to every vertex of every geometry.

    ``origins`` is None (o = 0) or an (n, 2) array — one origin per
    geometry, e.g. for scaling each object about its own center. Returns a
    new geometry array; the input is not modified.
    """
    geoms = np.array(geoms, dtype=object, copy=True)
    if geoms.size == 0:
        return geoms
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    if coords.size == 0:
        return geoms
    A = np.asarray(A, dtype=np.float64)
    if origins is not None:
        o = np.asarray(origins, dtype=np.float64)[idx]
        coords = (coords - o) @ A.T + o
    else:
        coords = coords @ A.T
    if t is not None:
        coords = coords + np.asarray(t, dtype=np.float64)
    return shapely.set_coordinates(geoms, coords)


def _origins(geoms: np.ndarray, origin: Origin) -> np.ndarray | None:
    """Per-geometry origins for "center" (bbox center) / "centroid", or
    None + a global shift for a fixed point (handled by the caller)."""
    if isinstance(origin, str):
        if origin == "center":
            b = shapely.bounds(geoms)
            return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])
        if origin == "centroid":
            return shapely.get_coordinates(shapely.centroid(geoms))
        raise ValueError(f"origin must be 'center', 'centroid' or (x, y), got {origin!r}")
    return None


def transform_rows(geoms: np.ndarray, radii: np.ndarray, is_circle: np.ndarray,
                   A: np.ndarray, t: np.ndarray | None = None,
                   origin: Origin = (0.0, 0.0),
                   circles: CircleMode = "raise"
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform annotation rows (geometry + circle radius).

    Circle rows may hold either the center point (storage form) or the disk
    polygon; both work. Returns (geoms, radii, converted) where
    ``converted`` flags circles turned into polygons (circles="to_polygon"
    with a non-similarity transform); their radius is NaN.
    """
    if circles not in ("raise", "to_polygon"):
        raise ValueError(f"circles must be 'raise' or 'to_polygon', got {circles!r}")
    A = np.asarray(A, dtype=np.float64)
    geoms = np.asarray(geoms, dtype=object)
    radii = np.array(radii, dtype=np.float64, copy=True)
    is_circle = np.asarray(is_circle, dtype=bool)
    converted = np.zeros(geoms.size, dtype=bool)

    s = similarity_scale(A)
    if is_circle.any():
        if s is not None:
            radii[is_circle] *= s
        elif circles == "raise":
            raise ValueError(
                f"the transform is not a similarity: it would deform "
                f"{int(is_circle.sum())} circle(s). Pass circles='to_polygon' to "
                f"convert them to polygons.")
        else:
            converted = is_circle.copy()
            pts = shapely.get_type_id(geoms) == 0
            to_buf = converted & pts
            if to_buf.any():
                geoms = geoms.copy()
                geoms[to_buf] = shapely.buffer(geoms[to_buf], radii[to_buf])
            radii[converted] = np.nan

    origins = _origins(geoms, origin)
    if origins is None:  # fixed point (x0, y0): p' = A (p - o) + o + t
        o = np.asarray(origin, dtype=np.float64)
        t = (np.zeros(2) if t is None else np.asarray(t, dtype=np.float64)) + o - A @ o
    return transform_geometries(geoms, A, t, origins), radii, converted


def _transform_frame(gdf: gpd.GeoDataFrame, A: np.ndarray, t: np.ndarray | None,
                     origin: Origin, circles: CircleMode) -> gpd.GeoDataFrame:
    out = gdf.copy()
    n = len(out)
    is_circle = (out["type"].astype(str).to_numpy() == "CIRCLE"
                 if "type" in out.columns else np.zeros(n, dtype=bool))
    radii = (out["circle_r"].to_numpy(dtype=np.float64)
             if "circle_r" in out.columns else np.full(n, np.nan))
    geoms, radii, converted = transform_rows(
        out.geometry.to_numpy(), radii, is_circle, A, t, origin, circles)
    out[out.geometry.name] = gpd.GeoSeries(geoms, index=out.index, crs=gdf.crs)
    if "circle_r" in out.columns:
        out["circle_r"] = radii
    if converted.any():
        tp = out["type"].astype(object).to_numpy(copy=True)
        tp[converted] = "POLYGON"
        out["type"] = (pd.Categorical(tp, categories=gdf["type"].cat.categories)
                       if isinstance(gdf["type"].dtype, pd.CategoricalDtype) else tp)
    return out


def translate(gdf: gpd.GeoDataFrame, x_off: float,
              y_off: float | None = None) -> gpd.GeoDataFrame:
    """Shift all objects by (x_off, y_off); y_off defaults to x_off."""
    if y_off is None:
        y_off = x_off
    return _transform_frame(gdf, np.eye(2), np.array([x_off, y_off], dtype=float),
                            (0.0, 0.0), "raise")


def scale(gdf: gpd.GeoDataFrame, x_scale: float, y_scale: float | None = None,
          origin: Origin = (0.0, 0.0), circles: CircleMode = "raise") -> gpd.GeoDataFrame:
    """Scale about ``origin``: a fixed (x, y) point (default (0, 0)), or
    "center" / "centroid" to scale each object about its own."""
    if y_scale is None:
        y_scale = x_scale
    return _transform_frame(gdf, np.diag([float(x_scale), float(y_scale)]), None,
                            origin, circles)


def resize(gdf: gpd.GeoDataFrame, factor: float) -> gpd.GeoDataFrame:
    """Uniform scaling about (0, 0): a change of resolution."""
    return scale(gdf, factor)


def affine(gdf: gpd.GeoDataFrame, M: Any,
           circles: CircleMode = "raise") -> gpd.GeoDataFrame:
    """Apply a 2 x 3 affine matrix (row-major, on (x, y, 1))."""
    m = as_affine_matrix(M)
    return _transform_frame(gdf, m[:, :2], m[:, 2], (0.0, 0.0), circles)
