# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from pydantic import BaseModel


class ImageShape(BaseModel):
    width: int
    height: int


class Px(BaseModel):
    """A pixel position: integer (x, y) coordinates at some pyramid level."""
    x: int
    y: int
