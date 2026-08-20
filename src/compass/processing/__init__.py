# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Domain algorithms: tissue detection, stain normalization, patch sampling,
image registration, spatial-transcriptomics (Visium) tools.

numpy/geometry in -> numpy/geometry out. Depends only on ``compass.core``;
knows nothing about ingestion formats or the UI.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
