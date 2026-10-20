# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""COMPASS core: raster access (PyramidalRaster protocol + Zarr backend) and
the annotation layer (object wrappers, vectorized transforms, SQLite/R-tree
store -> geopandas / polars; see ``compass.core.annot``).

Minimal dependencies (numpy, zarr, shapely, geopandas, polars, pydantic,
dask); depends on nothing else in this project. Exposes plain numpy /
pandas / polars / geopandas / dask objects only.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from .annot import (ANNOT_CODE_TYPE, ANNOT_TYPE_CODE, SCHEMA_VERSION,
                    SQLITE_DDL, Annotation, AnnotationObject, AnnotationStore,
                    Circle, Point, PointSet, PolyLine, Polygon, init_db)
from .magnif import Magnification
from .pyramid import PyramidalRaster, rasterize_polygon
from .types import ImageShape, Px
from .zarr_raster import (ZarrRaster, create_level_array, write_pyramid,
                          write_pyramid_attrs)

__all__ = [
    "ANNOT_CODE_TYPE",
    "ANNOT_TYPE_CODE",
    "Annotation",
    "AnnotationObject",
    "AnnotationStore",
    "Circle",
    "ImageShape",
    "Magnification",
    "Point",
    "PointSet",
    "PolyLine",
    "Polygon",
    "Px",
    "PyramidalRaster",
    "SCHEMA_VERSION",
    "SQLITE_DDL",
    "ZarrRaster",
    "create_level_array",
    "init_db",
    "rasterize_polygon",
    "write_pyramid",
    "write_pyramid_attrs",
]
