# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Single-object annotation wrappers: Point, PointSet, PolyLine, Polygon,
Circle.

Ported from ``compass.legacy.annot`` with the same constructors and
methods. Use them to build or inspect a handful of objects; bulk work
(thousands to millions of objects) goes through GeoDataFrames and
``AnnotationStore`` instead — see ``objects_to_gdf`` / ``gdf_to_objects``
for the bridge between the two.

Coordinates are (x, y) pairs, never (row, column).
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import copy
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.affinity as sha
import shapely.geometry as shg

from .transform import as_affine_matrix, similarity_scale

__all__ = [
    "AnnotationObject",
    "Point",
    "PointSet",
    "PolyLine",
    "Polygon",
    "Circle",
    "create_empty_annotation_object",
    "createEmptyAnnotationObject",
    "make_annotation_object",
    "objects_to_gdf",
    "gdf_to_objects",
]

#: Default number of vertices of a circle's polygonal disk. Matches
#: shapely's default ``buffer`` (quad_segs=16), which is also what
#: ``AnnotationStore`` uses when it decodes circles to disks.
CIRCLE_N_POINTS = 64


#####
class AnnotationObject:
    """Common interface of all annotation objects: a shapely geometry, a
    name, and a dict of scalar attributes (``data``, values stored as
    float32). Transformations act in place.

    ``id`` is -1 until the object is added to a store (or an ``Annotation``),
    which assigns it. Not instantiable itself: use one of the subclasses.
    """

    TYPE: ClassVar[str] = ""
    _GEOM_TYPES: ClassVar[tuple[str, ...]] = ()

    def __init__(self, name: str | None = None,
                 data: Mapping[str, Any] | None = None):
        if type(self) is AnnotationObject:
            raise TypeError("AnnotationObject is abstract; use Point, PointSet, "
                            "PolyLine, Polygon or Circle")
        self._geom: shapely.Geometry = shg.Point()
        self._name: str | None = name if name is not None else self.TYPE
        self._data: dict[str, Any] = dict(data) if data is not None else {}
        self._id: int = -1

    # -- construction helpers ---------------------------------------------

    def _set_geometry(self, coords: Any) -> None:
        """Build the geometry from coordinates, or take a ready shapely
        geometry of an accepted type."""
        if isinstance(coords, shapely.Geometry):
            if coords.geom_type not in self._GEOM_TYPES:
                raise ValueError(f"{self.TYPE} needs a geometry of type "
                                 f"{' or '.join(self._GEOM_TYPES)}, got {coords.geom_type}")
            self._geom = coords
            return
        if not isinstance(coords, Iterable):
            raise ValueError("coords cannot be interpreted as 2D coordinates")
        self._geom = self._from_coords(coords)

    @staticmethod
    def _from_coords(coords: Any) -> shapely.Geometry:
        raise NotImplementedError

    @classmethod
    def from_geometry(cls, geom: shapely.Geometry, name: str | None = None,
                      data: Mapping[str, Any] | None = None) -> "AnnotationObject":
        """Wrap an existing shapely geometry."""
        return cls(geom, name=name, data=data)

    def duplicate(self) -> "AnnotationObject":
        """An independent copy (geometry, name, data) with no id."""
        obj = copy.copy(self)
        obj._data = copy.deepcopy(self._data)
        obj._id = -1
        return obj

    # -- properties --------------------------------------------------------

    @property
    def id(self) -> int:
        """Object id in its store; -1 if not stored."""
        return self._id

    @property
    def geom(self) -> shapely.Geometry:
        """The shapely geometry of the object."""
        return self._geom

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        self._name = value

    @property
    def type(self) -> str:
        """Annotation type: POINT, POINTSET, POLYLINE, POLYGON or CIRCLE."""
        return self.TYPE

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def set_data(self, data: Mapping[str, Any] | None) -> None:
        self._data = dict(data) if data is not None else {}

    @property
    def x(self) -> np.ndarray:
        """x coordinate(s); always a vector, even for a single point."""
        return shapely.get_coordinates(self.geom)[:, 0]

    @property
    def y(self) -> np.ndarray:
        """y coordinate(s); always a vector, even for a single point."""
        return shapely.get_coordinates(self.geom)[:, 1]

    def xy(self) -> np.ndarray:
        """(n, 2) array of the coordinates defining the object."""
        return shapely.get_coordinates(self.geom)

    def size(self) -> int:
        """Number of points defining the object."""
        return int(shapely.get_num_coordinates(self.geom))

    def bounding_box(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y)."""
        return tuple(float(v) for v in self.geom.bounds)

    def __str__(self) -> str:
        return f"{self.type} <{self.name}>: {self.geom}"

    __repr__ = __str__

    # -- transformations (in place) ----------------------------------------

    def translate(self, x_off: float, y_off: float | None = None) -> None:
        """Shift by (x_off, y_off); y_off defaults to x_off."""
        if y_off is None:
            y_off = x_off
        self._geom = sha.translate(self._geom, x_off, y_off)

    def scale(self, x_scale: float, y_scale: float | None = None,
              origin: str | tuple[float, float] | shg.Point = "center") -> None:
        """Scale about ``origin``: "center" (of the bounding box),
        "centroid", or a point / (x, y) tuple. y_scale defaults to x_scale."""
        if y_scale is None:
            y_scale = x_scale
        self._geom = sha.scale(self._geom, xfact=x_scale, yfact=y_scale, origin=origin)

    def resize(self, factor: float) -> None:
        """Scale both axes by ``factor`` about (0, 0) — a change of
        resolution, e.g. moving the annotation to another pyramid level."""
        self.scale(factor, origin=(0.0, 0.0))

    def affine(self, M: Any) -> None:
        """Apply the 2 x 3 affine matrix M (row-major, acting on (x, y, 1)):
        x' = M[0,0] x + M[0,1] y + M[0,2];  y' = M[1,0] x + M[1,1] y + M[1,2].
        A 3 x 3 homogeneous matrix is accepted too."""
        m = as_affine_matrix(M)
        self._geom = sha.affine_transform(
            self._geom, [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]])

    # -- serialization -----------------------------------------------------

    def asdict(self) -> dict:
        """Dictionary representation (geometry as WKT), legacy-compatible."""
        return {
            "id": self._id,
            "annotation_type": self.TYPE,
            "name": self._name,
            "data": dict(self._data),
            "geom": shapely.to_wkt(self._geom),
        }

    def fromdict(self, d: Mapping[str, Any]) -> None:
        """Initialize from ``asdict()`` output."""
        if d.get("annotation_type", self.TYPE) != self.TYPE:
            raise ValueError(f"cannot load a {d['annotation_type']} into a {self.TYPE}")
        self._id = int(d.get("id", -1))
        self._name = d.get("name")
        self._data = dict(d.get("data") or {})
        self._geom = shapely.from_wkt(d["geom"])
##


class Point(AnnotationObject):
    """A single position in the image."""

    TYPE = "POINT"
    _GEOM_TYPES = ("Point",)

    def __init__(self, coords: Any = (0.0, 0.0), name: str | None = None,
                 data: Mapping[str, Any] | None = None):
        super().__init__(name, data)
        self._set_geometry(coords)

    @staticmethod
    def _from_coords(coords: Any) -> shapely.Geometry:
        xy = np.asarray(coords, dtype=float).ravel()
        if xy.size < 2:
            raise ValueError("a point needs (x, y) coordinates")
        return shg.Point(xy[:2])


class PointSet(AnnotationObject):
    """An ordered collection of points."""

    TYPE = "POINTSET"
    _GEOM_TYPES = ("MultiPoint",)

    def __init__(self, coords: Any, name: str | None = None,
                 data: Mapping[str, Any] | None = None):
        super().__init__(name if name is not None else "POINTS", data)
        self._set_geometry(coords)

    @staticmethod
    def _from_coords(coords: Any) -> shapely.Geometry:
        return shg.MultiPoint(np.asarray(coords, dtype=float).reshape(-1, 2))


class PolyLine(AnnotationObject):
    """A polygonal line: segments (x0,y0)->(x1,y1)->(x2,y2)->..."""

    TYPE = "POLYLINE"
    _GEOM_TYPES = ("LineString", "MultiLineString")

    def __init__(self, coords: Any, name: str | None = None,
                 data: Mapping[str, Any] | None = None):
        super().__init__(name, data)
        self._set_geometry(coords)

    @staticmethod
    def _from_coords(coords: Any) -> shapely.Geometry:
        return shg.LineString(np.asarray(coords, dtype=float).reshape(-1, 2))


class Polygon(AnnotationObject):
    """A closed polygon given by its outer ring [(x0,y0), (x1,y1), ...]
    (closing the ring is optional). Polygons with holes, or multipolygons,
    can be passed as a shapely geometry."""

    TYPE = "POLYGON"
    _GEOM_TYPES = ("Polygon", "MultiPolygon")

    def __init__(self, coords: Any, name: str | None = None,
                 data: Mapping[str, Any] | None = None):
        super().__init__(name, data)
        self._set_geometry(coords)

    @staticmethod
    def _from_coords(coords: Any) -> shapely.Geometry:
        return shg.Polygon(np.asarray(coords, dtype=float).reshape(-1, 2))


class Circle(AnnotationObject):
    """A circle, kept exactly as center + radius. ``geom`` is its polygonal
    disk (``n_points`` vertices), so it draws and intersects like any
    polygon; it is stored as the center point plus ``circle_r``.

    Only similarity transforms (shift, rotation, uniform scale, reflection)
    keep a circle a circle; anything else raises ValueError — convert with
    ``to_polygon()`` first.
    """

    TYPE = "CIRCLE"
    _GEOM_TYPES = ("Point",)

    def __init__(self, center: Any = (0.0, 0.0), radius: float = 1.0,
                 name: str | None = None,
                 data: Mapping[str, Any] | None = None,
                 n_points: int = CIRCLE_N_POINTS):
        super().__init__(name, data)
        if isinstance(center, shapely.Geometry):
            if center.geom_type != "Point":
                raise ValueError("a circle's center must be a Point")
            center = (center.x, center.y)
        c = np.asarray(center, dtype=float).ravel()
        if c.size < 2:
            raise ValueError("a circle needs (x, y) center coordinates")
        if radius < 0:
            raise ValueError("circle radius must be >= 0")
        self._center = (float(c[0]), float(c[1]))
        self._radius = float(radius)
        self._quad_segs = max(1, int(round(n_points / 4)))
        self._update_geom()

    @classmethod
    def from_geometry(cls, geom: shapely.Geometry, name: str | None = None,
                      data: Mapping[str, Any] | None = None,
                      radius: float | None = None) -> "Circle":
        """From a center point + ``radius``, or (radius omitted) a disk
        polygon: center = centroid, radius = mean vertex distance."""
        if geom.geom_type == "Point":
            if radius is None:
                raise ValueError("a circle from a point needs a radius")
            return cls(geom, radius, name=name, data=data)
        c = geom.centroid
        if radius is None:
            xy = shapely.get_coordinates(geom)
            radius = float(np.hypot(xy[:, 0] - c.x, xy[:, 1] - c.y).mean())
        return cls(c, radius, name=name, data=data)

    def _update_geom(self) -> None:
        self._geom = shg.Point(self._center).buffer(self._radius, quad_segs=self._quad_segs)

    @property
    def center(self) -> tuple[float, float]:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def bounding_box(self) -> tuple[float, float, float, float]:
        (x, y), r = self._center, self._radius
        return (x - r, y - r, x + r, y + r)

    def __str__(self) -> str:
        return f"CIRCLE <{self.name}>: center={self._center}, radius={self._radius}"

    __repr__ = __str__

    def to_polygon(self) -> Polygon:
        """The disk as a Polygon object (same name, a copy of the data)."""
        return Polygon(self._geom, name=self._name, data=copy.deepcopy(self._data))

    def translate(self, x_off: float, y_off: float | None = None) -> None:
        if y_off is None:
            y_off = x_off
        self._center = (self._center[0] + x_off, self._center[1] + y_off)
        self._update_geom()

    def scale(self, x_scale: float, y_scale: float | None = None,
              origin: str | tuple[float, float] | shg.Point = "center") -> None:
        if y_scale is not None and y_scale != x_scale:
            raise ValueError("non-uniform scaling would deform the circle; "
                             "use to_polygon() first")
        if not (isinstance(origin, str) and origin in ("center", "centroid")):
            c = sha.scale(shg.Point(self._center), xfact=x_scale, yfact=x_scale,
                          origin=origin)
            self._center = (c.x, c.y)
        self._radius *= abs(x_scale)
        self._update_geom()

    def affine(self, M: Any) -> None:
        m = as_affine_matrix(M)
        s = similarity_scale(m[:, :2])
        if s is None:
            raise ValueError("the transform is not a similarity, so it would deform "
                             "the circle; use to_polygon() first")
        x, y = self._center
        self._center = (float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
                        float(m[1, 0] * x + m[1, 1] * y + m[1, 2]))
        self._radius *= s
        self._update_geom()

    def asdict(self) -> dict:
        d = super().asdict()
        d["center"] = self._center
        d["radius"] = self._radius
        return d

    def fromdict(self, d: Mapping[str, Any]) -> None:
        super().fromdict(d)
        if "center" in d and "radius" in d:
            self._center = tuple(float(v) for v in d["center"])
            self._radius = float(d["radius"])
        else:  # only the disk is known
            c = Circle.from_geometry(self._geom)
            self._center, self._radius = c._center, c._radius
        self._update_geom()


_CLASSES: dict[str, type[AnnotationObject]] = {
    c.TYPE: c for c in (Point, PointSet, PolyLine, Polygon, Circle)}


def create_empty_annotation_object(annot_type: str) -> AnnotationObject:
    """A placeholder object of the given type (POINT, POINTSET,
    POLYLINE/LINESTRING, POLYGON, CIRCLE), e.g. to ``fromdict()`` into."""
    tp = annot_type.upper()
    if tp == "POINT":
        return Point((0.0, 0.0))
    if tp == "POINTSET":
        return PointSet([[0.0, 0.0]])
    if tp in ("POLYLINE", "LINESTRING"):
        return PolyLine([[0.0, 0.0], [1.0, 1.0]])
    if tp == "POLYGON":
        return Polygon([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    if tp == "CIRCLE":
        return Circle((0.0, 0.0), 1.0)
    raise ValueError(f"unknown annotation type: {annot_type}")


#: legacy name
createEmptyAnnotationObject = create_empty_annotation_object


def make_annotation_object(annot_type: str, geom: shapely.Geometry,
                           name: str | None = None,
                           data: Mapping[str, Any] | None = None,
                           circle_r: float | None = None) -> AnnotationObject:
    """Wrap a shapely geometry as an object of the given type. For CIRCLE,
    ``geom`` is the center point (with ``circle_r``) or the disk."""
    cls = _CLASSES.get(annot_type.upper())
    if cls is None:
        raise ValueError(f"unknown annotation type: {annot_type}")
    if cls is Circle:
        r = None if circle_r is None or np.isnan(circle_r) else float(circle_r)
        return Circle.from_geometry(geom, name=name, data=data, radius=r)
    return cls.from_geometry(geom, name=name, data=data)


def objects_to_gdf(objects: Iterable[AnnotationObject],
                   circles_as_disks: bool = True) -> gpd.GeoDataFrame:
    """Objects -> GeoDataFrame with the same columns ``AnnotationStore``
    returns (``type``, ``name``, ``circle_r``, ``geometry``). Indexed by the
    object ids when all objects have one, else by position. ``data`` is not
    included."""
    objs = list(objects)
    types = [o.type for o in objs]
    radii = np.array([o.radius if isinstance(o, Circle) else np.nan for o in objs],
                     dtype=np.float64)
    geoms = [o.geom if (circles_as_disks or not isinstance(o, Circle))
             else shg.Point(o.center) for o in objs]
    ids = [o.id for o in objs]
    index = (pd.Index(ids, dtype="int64", name="object_id")
             if objs and all(i >= 0 for i in ids) else pd.RangeIndex(len(objs)))
    return gpd.GeoDataFrame(
        {"type": pd.Categorical(types, categories=list(_CLASSES)),
         "name": pd.array([o.name for o in objs], dtype="string"),
         "circle_r": radii},
        geometry=gpd.GeoSeries(geoms, index=index), index=index)


def gdf_to_objects(gdf: gpd.GeoDataFrame,
                   data: Mapping[int, Mapping[str, Any]] | None = None
                   ) -> list[AnnotationObject]:
    """GeoDataFrame -> objects (one per row). The ``type`` column decides
    the class (inferred from the geometry when absent); ``name`` and
    ``circle_r`` are used when present. ``data`` maps an index label to the
    object's scalar attributes. If the index is ``object_id`` the objects
    get those ids."""
    from .schema import geometry_type_names

    types = (gdf["type"].astype(str).to_numpy() if "type" in gdf.columns
             else geometry_type_names(gdf.geometry.to_numpy()))
    names = gdf["name"].to_numpy() if "name" in gdf.columns else [None] * len(gdf)
    radii = (gdf["circle_r"].to_numpy(dtype=float) if "circle_r" in gdf.columns
             else np.full(len(gdf), np.nan))
    use_ids = gdf.index.name == "object_id"
    out = []
    for label, tp, geom, nm, r in zip(gdf.index, types, gdf.geometry, names, radii):
        nm = None if nm is None or nm is pd.NA else str(nm)
        obj = make_annotation_object(tp, geom, name=nm,
                                     data=None if data is None else data.get(label),
                                     circle_r=r)
        if use_ids:
            obj._id = int(label)
        out.append(obj)
    return out
