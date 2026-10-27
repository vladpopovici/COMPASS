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

pyvips and tifffile are imported lazily inside the functions that use
them; OpenSlide only by the ``wsi`` submodule (``from compass.connector.wsi
import WSI``), so importing ``compass.connector`` itself needs none of them.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from .asap import AsapTarget, asap_to_store, read_asap_xml
from .export import build_omexml, raster2tiff
from .ingest import (TRANSFORM_TYPES, box_downsample, closest_level_for_mpp,
                     crop_scale_transform, import_record, ometiff2zarr,
                     plan_import, plan_levels, resolve_crop, wsi2zarr)

__all__ = [
    "AsapTarget",
    "asap_to_store",
    "read_asap_xml",
    "TRANSFORM_TYPES",
    "box_downsample",
    "closest_level_for_mpp",
    "crop_scale_transform",
    "import_record",
    "plan_import",
    "build_omexml",
    "ometiff2zarr",
    "plan_levels",
    "raster2tiff",
    "resolve_crop",
    "wsi2zarr",
]
