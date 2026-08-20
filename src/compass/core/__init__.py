# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""COMPASS core: raster access (PyramidalRaster protocol + Zarr backend) and
the vectorized annotation data-access layer (SQLite/R-tree -> geopandas /
polars).

Minimal dependencies (numpy, zarr, shapely, geopandas, polars, pydantic,
dask); depends on nothing else in this project. Exposes plain numpy /
pandas / polars / geopandas / dask objects only.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from .annot import (ANNOT_CODE_TYPE, ANNOT_TYPE_CODE, SCHEMA_VERSION,
                    SQLITE_DDL, AnnotationStore, init_db)
from .magnif import Magnification
from .pyramid import PyramidalRaster, rasterize_polygon
from .types import ImageShape, Px
from .zarr_raster import ZarrRaster, write_pyramid

__all__ = [
    "ANNOT_CODE_TYPE",
    "ANNOT_TYPE_CODE",
    "AnnotationStore",
    "ImageShape",
    "Magnification",
    "Px",
    "PyramidalRaster",
    "SCHEMA_VERSION",
    "SQLITE_DDL",
    "ZarrRaster",
    "init_db",
    "rasterize_polygon",
    "write_pyramid",
]
