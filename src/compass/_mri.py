# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import numpy as np
import dask.array as da
import h5py

from pathlib import Path
from os import PathLike

from _pyr import PyramidalImage
from _misc import ImageShape, Px
from _magnif import Magnification

#####
class MRI(PyramidalImage):
    """MultiResolution Image - a simple and convenient interface to access pixels from a
    pyramidal image. The image is supposed to be stored in HDF5 format (see README).

    Args:
        path (str): path to the .h5 file with image data

    Attributes:
        _path (Path)
        _info (dict): contains
            * `max_level`: number of levels in the pyramid
            * `channel_names`: normally R, G, B, but others may be possible
            * `dimension_names`: normally y, x, c
            * `mpp_x` and `mpp_y`: the original resolution (microns-per-pixel) for x and y dimensions
            * `mag_step`: magnification step (usually 2.0): scaling factor between pyramid levels
            * `objective_power`: native objective used to acquire the image (e.g. 20.0 or 40.0)
            * `extent`: a `2 x max_level` array with `extent[0,i]` and `extent[1,i]` indicating
               the width and height of level `i`, respectively
        _pyramid_levels (2 x N array): convenient access to level extents
        _mag (Magnification): magnification converter
    """

    def __init__(self, path: str | Path | PathLike):
        if not Path(path).exists() or Path(path).suffix != '.h5':
            raise ValueError(f"expected a .h5 file, got {path}")

        self.__storage = h5py.File(path, mode='r', rdcc_nbytes=1024 ** 2 * 64, rdcc_nslots=1e6)
        self._info = dict(self.__storage.attrs)

        super().__init__(path,
                         ImageShape(width=int(self._info['extent'][0][0]), height=int(self._info['extent'][1][0])),
                         Magnification(float(self._info["objective_power"]),
                                       0.5 * (float(self._info["mpp_x"]) + float(self._info["mpp_y"])),
                                       level=0,
                                       n_levels=int(self._info["max_level"]),
                                       magnif_step=float(self._info["mag_step"])))

    def __del__(self):
        self.__storage.close()

    @property
    def info(self) -> dict:
        return self._info

    def get_region_px(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a region from the image source. The region is specified in
            pixel coordinates.

            Args:
                x0, y0 (long): top left corner of the region (in pixels, at the specified
                level)
                width, height (long): width and height (in pixels) of the region.
                level (int): the magnification level to read from
                as_type: type of the pixels (default numpy.uint8)

            Returns:
                a numpy.ndarray
        """

        if level < 0 or level >= self.nlevels:
            raise RuntimeError("requested level does not exist")

        # check bounds:
        if x0 >= self.widths[level] or y0 >= self.heights[level] or \
                x0 + width > self.widths[level] or \
                y0 + height > self.heights[level]:
            raise RuntimeError("region out of layer's extent")

        # print(f"reading region from {self.path} at level {level}: {x0}, {y0} x {width}, {height}")
        # img = da.from_zarr(self.path, component=str(level), dtype=as_type)
        # with h5py.File(self.path, mode='r') as z:
        img = self.__storage[f'/{level}/data'][y0:y0 + height, x0:x0 + width, ...]

        return img

    def get_plane(self, level: int = 0, as_type=np.uint8) -> da.Array:
        """Read a whole plane from the image pyramid and return it as a Numpy array.

        Args:
            level (int): pyramid level to read
            as_type: type of the pixels (default numpy.uint8)

        Returns:
            a numpy.ndarray
        """
        if level < 0 or level >= self.nlevels:
            raise RuntimeError("requested level does not exist")

        img = da.array(self.__storage[f'/{level}/data'])[:]

        return img


##
