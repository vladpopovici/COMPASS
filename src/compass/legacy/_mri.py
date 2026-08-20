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

from ._pyr import PyramidalImage
from ._misc import ImageShape, Px
from ._magnif import Magnification

#####
class MRI(PyramidalImage):
    """MultiResolution Image - a simple and convenient interface to access pixels from a
    pyramidal image. The image is supposed to be stored in HDF5 format (see README).

    Args:
        path (str): path to the .h5 file with image data
        open_mode (str): opening mode for h5py; either 'r', 'a' or 'w'

    Attributes:
        _path (Path)
        _info (dict): contains
            * `level_count`: number of levels in the pyramid
            * `channel_names`: normally R, G, B, but others may be possible
            * `dimension_names`: normally y, x, c
            * `base_mpp_x` and `base_mpp_y`: the original resolution (microns-per-pixel) for x and y dimensions
            * `base_mag_step`: magnification step (usually 2.0): scaling factor between pyramid levels
            * `base_objective_power`: native objective used to acquire the image (e.g. 20.0 or 40.0)
        _pyramid_levels (2 x N array): convenient access to level extents
        _mag (Magnification): magnification converter
    """

    def __init__(self, path: str | Path | PathLike, open_mode: str="r"):
        if not Path(path).exists() or Path(path).suffix != '.h5':
            raise ValueError(f"expected a .h5 file, got {path}")

        self._path = Path(path)
        self.__storage = h5py.File(str(path), mode=open_mode, rdcc_nbytes=1024 ** 2 * 64, rdcc_nslots=1e6)
        self._info = dict(self.__storage.attrs)
        base_height, base_width = self.__storage['/scale0/image'].shape[0:2]

        super().__init__(ImageShape(width=base_width, height=base_height),
                         Magnification(float(self._info["base_objective_power"]),
                                       self._info["base_mpp"],
                                       level=0,
                                       n_levels=int(self._info["level_count"]),
                                       magnif_step=float(self._info["base_mag_step"])))

        # Due to rounding errors the actual level images stored may be 1 pixel off from
        # the computed shapes. Update their dims from actual images:
        for l in range(self.nlevels):
            self._pyramid_levels[1, l] = self.__storage[f'/scale{l}/image'].shape[0]  # height
            self._pyramid_levels[0, l] = self.__storage[f'/scale{l}/image'].shape[1]  # width

    def __del__(self):
        self.__storage.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def storage(self) -> h5py.File:
        return self.__storage

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

        img = self.__storage[f'/scale{level}/image'][y0:y0 + height, x0:x0 + width, ...]

        return img

    def get_region_px_dask(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int = 0, as_type=np.uint8) -> da.Array:
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

        img = da.from_array(
            self.__storage[f'/scale{level}/image'],
            chunks=(min(8192, height), min(8192, width), 3),
        )[y0:y0 + height, x0:x0 + width, ...]

        return img

    def get_plane(self, level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a whole plane from the image pyramid and return it as a Numpy array.

        Args:
            level (int): pyramid level to read
            as_type: type of the pixels (default numpy.uint8)

        Returns:
            a numpy.ndarray
        """
        if level < 0 or level >= self.nlevels:
            raise RuntimeError("requested level does not exist")

        img = self.__storage[f'/scale{level}/image'][:]

        return img

    def get_plane_dask(self, level: int = 0, as_type=np.uint8) -> da.Array:
        """Read a whole plane from the image pyramid and return it as a Numpy array.

        Args:
            level (int): pyramid level to read
            as_type: type of the pixels (default numpy.uint8)

        Returns:
            a numpy.ndarray
        """
        if level < 0 or level >= self.nlevels:
            raise RuntimeError("requested level does not exist")

        img = da.from_array(
            self.__storage[f'/scale{level}/image'],
            chunks=(min(8192, self.heights[level]), min(8192, self.widths[level]), 3),
        )

        return img
##
