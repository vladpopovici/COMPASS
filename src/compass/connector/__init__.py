# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Ingestion and interchange: WSI reading (OpenSlide/Bio-Formats), pyramid
baking into the core Zarr format, AnnData/SpatialData conversion.

Heavy/volatile dependencies are isolated here. This package is never
imported by ``compass.viewer`` or ``compass.processing``; it runs as
one-shot conversion tooling that writes into ``compass.core`` formats.

pyvips is imported lazily inside the functions that use it; OpenSlide only
by the ``wsi`` submodule (``from compass.connector.wsi import WSI``), so
importing ``compass.connector`` itself needs neither system library.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from .export import build_omexml, raster2tiff
from .ingest import plan_levels, resolve_crop, wsi2zarr

__all__ = [
    "build_omexml",
    "plan_levels",
    "raster2tiff",
    "resolve_crop",
    "wsi2zarr",
]
