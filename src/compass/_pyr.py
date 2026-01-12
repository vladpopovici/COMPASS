# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import numpy as np
import pathlib
import shapely
import shapely.geometry as shg
import shapely.affinity as sha

from abc import ABC, abstractmethod
from os import PathLike
from math import floor

from ._misc import ImageShape, Px
from ._magnif import Magnification
from .mask import add_region, apply_mask

#####
class PyramidalImage(ABC):
    """Abstract base class for a pyramidal image. It provides the basic API
    for storing information about the geometry of the image and for
    interacting with the image."""

    def __init__(self,
                 base_dim: ImageShape = ImageShape(width=0, height=0),
                 magnif: Magnification = None):
        """Construct an abstract pyramidal image: no data yet, just
        the image shape, and magnification characteristics that must be
        provided as arguments.

        """
        self._mag: Magnification = magnif
        self._pyramid_levels: np.ndarray | None = None
        self._base_dim: ImageShape = base_dim
        self._nlevels: int = 0
        if self._mag is not None:
            self._nlevels = self._mag.nlevels
            self._pyramid_levels = np.zeros((2, self._mag.nlevels), dtype=int)
            self._pyramid_levels[0, 0] = self._base_dim.width
            self._pyramid_levels[1, 0] = self._base_dim.height
            for lv in range(1, self._mag.nlevels):
                self._pyramid_levels[0, lv] = int(self._pyramid_levels[0, lv - 1] / self._mag.magnif_step)
                self._pyramid_levels[1, lv] = int(self._pyramid_levels[1, lv - 1] / self._mag.magnif_step)

    # end

    @property
    @abstractmethod
    def info(self) -> dict:
        pass

    @property
    def nlevels(self) -> int:
        return self._nlevels

    @property
    def native_magnification(self) -> float:
        return self._mag.base_magnif

    @property
    def native_resolution(self) -> float:
        return self._mag.base_mpp

    def get_level_for_magnification(self, mag: float) -> int:
        return self._mag.get_level_for_magnif(mag)

    def get_level_for_mpp(self, mpp: float) -> int:
        return self._mag.get_level_for_mpp(mpp)

    def get_magnification_for_level(self, level: int) -> float:
        return self._mag.get_magnif_for_level(level)

    def get_magnification_for_mpp(self, mpp: float) -> float:
        return self._mag.get_magnif_for_mpp(mpp)

    def get_mpp_for_level(self, level: int) -> float:
        return self._mag.get_mpp_for_level(level)

    def get_mpp_for_magnification(self, mag: float) -> float:
        return self._mag.get_mpp_for_magnif(mag)

    def shape(self, level: int = 0) -> ImageShape:
        return ImageShape(width=int(self._pyramid_levels[0, level]),
                          height=int(self._pyramid_levels[1, level]))

    @property
    def pyramid_levels(self):
        return self._pyramid_levels

    @property
    def widths(self) -> np.ndarray:
        return self._pyramid_levels[0, :]

    @property
    def heights(self) -> np.ndarray:
        return self._pyramid_levels[1, :]

    def between_level_scaling_factor(self, from_level: int, to_level: int) -> float:
        """Return the scaling factor for converting coordinates (magnification) between
        two levels of the pyramid."""
        return self._mag.magnif_step ** (from_level - to_level)

    def convert_px(self, point: Px, from_level: int, to_level: int) -> Px:
        if from_level == to_level:
            return point
        s = self.between_level_scaling_factor(from_level, to_level)
        return Px(x=int(floor(point.x * s)), y=int(floor(point.y * s)))

    @abstractmethod
    def get_region_px(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a region from the image source. The region is specified in pixel coordinates."""
        pass

    def get_region(self, top_left_corner: Px, shape: ImageShape, level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a region from the image source. The region is specified in pixel coordinates."""
        return self.get_region_px(top_left_corner.x, top_left_corner.y, shape.width, shape.height, level, as_type)

    def get_plane(self, level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a plane from the image source. The plane is specified by its level."""
        return self.get_region_px(0, 0, self.widths[level], self.heights[level], level, as_type)

    def get_polygonal_region_px(self, contour: shg.Polygon, level: int,
                                border: int = 0, as_type=np.uint8) -> np.ndarray:
        """Returns a rectangular view of the image source that minimally covers a closed
        contour (polygon). All pixels outside the contour are set to 0.

        Args:
            contour (shapely.geometry.Polygon): a closed polygonal line given in
                terms of its vertices. The contour's coordinates are supposed to be
                precomputed and to be represented in pixel units at the desired level.
            level (int): image pyramid level
            border (int): if > 0, take this many extra pixels in the rectangular
                region (up to the limits on the image size)
            as_type: pixel type for the returned image (array)

        Returns:
            a numpy.ndarray
        """
        x0, y0, x1, y1 = [int(_z) for _z in contour.bounds]
        x0, y0 = max(0, x0 - border), max(0, y0 - border)
        x1, y1 = min(x1 + border, self.shape(level).width), \
            min(y1 + border, self.shape(level).height)
        # Shift the annotation such that (0,0) will correspond to (x0, y0)
        contour = sha.translate(contour, -x0, -y0)

        # Read the corresponding region
        img = self.get_region_px(x0, y0, x1 - x0, y1 - y0, level, as_type=as_type)

        # Prepare mask
        mask = np.zeros(img.shape[:2], dtype=as_type)
        add_region(mask, shapely.get_coordinates(contour))

        # Apply mask
        img = apply_mask(img, mask)

        return img
    ##
