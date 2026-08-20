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
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
