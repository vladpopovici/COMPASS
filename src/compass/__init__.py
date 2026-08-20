# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""COMPASS: computational pathology and spatial statistics.

Subpackages
-----------
core
    PyramidalRaster protocol + Zarr backend, annotation data-access layer
    (SQLite/R-tree -> geopandas/polars). Minimal dependencies; everything
    else depends on this.
connector
    AnnData/SpatialData conversion, WSI ingestion (OpenSlide/Bio-Formats),
    pyramid writing. Heavy/volatile dependencies isolated here; never
    imported by the running viewer.
processing
    Domain algorithms: tissue detection, stain normalization, patch
    sampling, image registration, spatial-transcriptomics tools.
viewer
    vispy/Qt canvas, psygnal reactive state, analytical panels.
legacy
    The original flat ``compass`` modules, kept unchanged while they are
    migrated into the packages above (see docs/architecture.md for the
    mapping). Do not add new code here.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "0.2.0"
