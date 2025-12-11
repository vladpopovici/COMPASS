# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import openslide as osl
import pathlib
import numpy as np

from os import PathLike
from math import floor

from _pyr import PyramidalImage
from _magnif import Magnification
from _misc import ImageShape, Px


#####
class WSI(PyramidalImage):
    """An extended version of OpenSlide, with more handy methods for dealing with
    microscopy slide images.

    Args:
        path (str): full path to the image file

    Attributes:
        _path (str): full path to WSI file
        _info (dict): slide image metadata
    """

    def __init__(self, path: str | pathlib.Path | PathLike):
        self._slide = slide_src = osl.OpenSlide(path)
        # keep also full meta info:
        self._original_meta = slide_meta = slide_src.properties
        # rename some of the most important properties:
        self._info = {
            'objective_power': float(slide_meta[osl.PROPERTY_NAME_OBJECTIVE_POWER]),
            'width': slide_src.dimensions[0],
            'height': slide_src.dimensions[1],
            'mpp_x': float(slide_meta[osl.PROPERTY_NAME_MPP_X]),  # microns/pixel
            'mpp_y': float(slide_meta[osl.PROPERTY_NAME_MPP_Y]),
            'n_levels': slide_src.level_count,  # no. of levels in pyramid
            'magnification_step': slide_src.level_downsamples[1] / slide_src.level_downsamples[0],
            'roi': dict(),
            'background': 0xFF
        }
        # optional properties:
        if osl.PROPERTY_NAME_BOUNDS_X in slide_meta:
            self._info['roi'] = {
                'x0': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_X]),
                'y0': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_Y]),
                'width': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_WIDTH]),
                'height': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_HEIGHT]),
            }
        if osl.PROPERTY_NAME_BACKGROUND_COLOR in slide_meta:
            self._info['background'] = 0xFF if slide_meta[osl.PROPERTY_NAME_BACKGROUND_COLOR] == 'FFFFFF' else 0

        super().__init__(path, ImageShape(width=slide_src.dimensions[0], height=slide_src.dimensions[1]),
                         Magnification(self._info['objective_power'],
                                       mpp=0.5 * (self._info['mpp_x'] + self._info['mpp_y']),
                                       level=0,
                                       n_levels=self._info['n_levels'],
                                       magnif_step=float(self._info['magnification_step'])))

    @property
    def info(self) -> dict:
        return self._info

    @property
    def level_count(self) -> int:  # TODO: remove in future versions
        """Return the number of levels in the multi-resolution pyramid."""
        return self.nlevels

    def downsample_factor(self, level: int) -> int:
        """Return the down-sampling factor (relative to level 0) for a given level."""
        if level < 0 or level >= self.nlevels:
            return -1

        return int(floor(self._mag.magnif_step ** level))

    def get_region_px(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a region from the image source. The region is specified in
            pixel coordinates.

            Args:
                x0, y0 (int): top left corner of the region (in pixels, at the specified
                level)
                width, height (int): width and height (in pixels) of the region.
                level (int): the level in the image pyramid to read from
                as_type: type of the pixels (default numpy.uint8)

            Returns:
                a numpy.ndarray [height x width x channels]
        """
        if level < 0 or level >= self.level_count:
            raise RuntimeError("requested level does not exist")

        # check bounds:
        if x0 >= self.widths[level] or y0 >= self.heights[level] or \
                x0 + width > self.widths[level] or \
                y0 + height > self.heights[level]:
            raise RuntimeError("region out of layer's extent")

        p = self.convert_px(Px(x=x0, y=y0), level, 0)
        x0_0, y0_0 = p.x, p.y
        img = self._slide.read_region((x0_0, y0_0), level, (width, height))
        img = np.array(img)

        if img.shape[2] == 4:  # has alpha channel, usually used for masking
            # fill with background
            mask = img[..., -1].squeeze()
            img[mask == 0, 0:4] = self.info['background']
            img = img[..., :-1]

        return img.astype(as_type)


####
