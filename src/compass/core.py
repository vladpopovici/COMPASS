# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.2

import pathlib
from math import floor, log
from typing import Union, Optional, Tuple, NewType
import dask.array as da
from pathlib import Path
import pyvips
import shapely
import zarr
import numpy as np
import shapely.geometry as shg
import shapely.affinity as sha
from .mask import add_region, apply_mask
from skimage.util import img_as_uint
import simplejson as json
import openslide as osl

ImageShape = NewType("ImageShape", dict[str, int])

"""
COMPASS.CORE: core classes and functions.
"""
#####
class Magnification:
    def __init__(self,
                 magnif: float,
                 mpp: float,
                 level: int = 0,
                 n_levels: int = 10,
                 magnif_step: float = 2.0):
        """Magnification handling/conversion.

        Args:
            magnif: base objective magnification (e.g. 10.0)
            mpp: resolution in microns per pixel at the given magnification
                (e.g. 0.245).
            level: level in the image pyramid corresponding to the
                magnification/mpp. Defaults to 0 - highest magnification.
            n_levels: number of levels in the image pyramid that are relevant/
                feasible
            magnif_step: scaling factor between levels in the image pyramid.
        """
        if level < 0 or level >= n_levels:
            raise RuntimeError("Specified level outside [0, (n_levels-1)]")
        self._base_magnif = magnif
        self._base_mpp = mpp
        self._base_level = level
        self._magnif_step = magnif_step
        self._n_levels = n_levels
        # initialize look-up tables:
        self.__magnif = magnif * self._magnif_step ** (level - np.arange(n_levels))
        self.__mpp = mpp * self._magnif_step ** (np.arange(n_levels) - level)

        return

    @property
    def magnif_step(self) -> float:
        return self._magnif_step

    def get_magnif_for_mpp(self, mpp: float) -> float:
        """
        Returns the objective magnification for a given resolution.
        Args:
            mpp: target resolution (microns per pixel)

        Returns:
            float: magnification corresponding to mpp. If <mpp> is outside the
                normal interval then return the corresponding end of the
                magnification values if still close enough (relative error <=0.1)
                or raise an Error
        """
        if mpp < self.__mpp[0]:
            # mpp outside normal interval, try to see if it's too far:
            if (self.__mpp[0] - mpp) / self.__mpp[0] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return float(self.__magnif[0])
        if mpp > self.__mpp[self._n_levels - 1]:
            # mpp outside normal interval, try to see if it's too far:
            if (mpp - self.__mpp[self._n_levels - 1]) / self.__mpp[self._n_levels - 1] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return float(self.__magnif[self._n_levels - 1])
        k = np.argmin(np.abs(mpp - self.__mpp))

        return float(self.__magnif[k])

    def get_mpp_for_magnif(self, magnif: float) -> float:
        """
        Return the resolution (microns per pixel - mpp) for a given objective
            magnification.
        Args:
            magnif: target magnification

        Returns:
            float: resolution (microns per pixel) corresponding to magnification
        """
        if magnif > self.__magnif[0] or magnif < self.__magnif[self._n_levels - 1]:
            raise RuntimeError('magnif outside supported interval') from None
        k = np.argmin(np.abs(magnif - self.__magnif))

        return float(self.__mpp[k])

    def get_level_for_magnif(self, magnif: float) -> int:
        """
        Return the level for a given objective magnification. Negative values
        correspond to magnification levels higher than the indicated base level.

        Args:
            magnif: target magnification

        Returns:
            int: resolution (mpp) corresponding to magnification
        """
        if magnif > self.__magnif[0] or magnif < self.__magnif[self._n_levels - 1]:
            raise RuntimeError('magnif outside supported interval') from None

        k = np.argmin(np.abs(magnif - self.__magnif))

        return k

    def get_level_for_mpp(self, mpp: float) -> int:
        """
        Return the level for a given resolution. Negative values
        correspond to resolution levels higher than the indicated base level.

        Args:
            mpp: target resolution

        Returns:
            int: resolution (mpp) corresponding to magnification
        """
        if mpp < self.__mpp[0]:
            # mpp outside normal interval, try to see if it's too far:
            if (self.__mpp[0] - mpp) / self.__mpp[0] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return 0
        if mpp > self.__mpp[self._n_levels - 1]:
            # mpp outside normal interval, try to see if it's too far:
            if (mpp - self.__mpp[self._n_levels - 1]) / self.__mpp[self._n_levels - 1] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return self._n_levels - 1

        k = np.argmin(np.abs(mpp - self.__mpp))

        return k

    def get_mpp_for_level(self, level: int) -> float:
        """
        Return the resolution (mpp) for a given level.

        Args:
            level: target level

        Returns:
            float: resolution (mpp)
        """
        if level < 0 or level >= self._n_levels:
            raise RuntimeError('level outside supported interval.') from None

        return float(self.__mpp[level])

    def get_magnification_step(self) -> float:
        """
        Return the magnification step between two consecutive levels.

        Returns:
            float: magnification step
        """
        return self._magnif_step
##

#####
class WSI(object):
    """An extended version of OpenSlide, with more handy methods
    for dealing with microscopy slide images.

    Args:
        path (str): full path to the image file

    Attributes:
        _path (str): full path to WSI file
        info (dict): slide image metadata
    """

    def __init__(self, path: str | pathlib.Path):
        self._path = pathlib.Path(path) if isinstance(path, str) else path
        self._slide = slide_src = osl.OpenSlide(self.path)
        self._original_meta = slide_meta = slide_src.properties
        self.info = {
            'objective_power': float(slide_meta[osl.PROPERTY_NAME_OBJECTIVE_POWER]),
            'width':  slide_src.dimensions[0],
            'height': slide_src.dimensions[1],
            'mpp_x': float(slide_meta[osl.PROPERTY_NAME_MPP_X]), # microns/pixel
            'mpp_y': float(slide_meta[osl.PROPERTY_NAME_MPP_Y]),
            'n_levels': slide_src.level_count,    # no. of levels in pyramid
            'magnification_step': slide_src.level_downsamples[1] / slide_src.level_downsamples[0],
            'roi': None,
            'background': 0xFF
        }

        # optional properties:
        if osl.PROPERTY_NAME_BOUNDS_X in slide_meta:
            self.info['roi'] = {
                'x0': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_X]),
                'y0': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_Y]),
                'width': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_WIDTH]),
                'height': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_HEIGHT]),
            }
        if osl.PROPERTY_NAME_BACKGROUND_COLOR in slide_meta:
            self.info['background'] = 0xFF if slide_meta[osl.PROPERTY_NAME_BACKGROUND_COLOR] == 'FFFFFF' else 0

        # _pyramid_levels: 2 x n_levels: [max_x, max_y] x [0,..., n_levels-1]
        self._pyramid_levels = np.zeros((2, self.info['n_levels']), dtype=int)

        # _pyramid_mpp: 2 x n_levels: [mpp_x, mpp_y] x [0,..., n_levels-1]
        self._pyramid_mpp = np.zeros((2, self.info['n_levels']))
        self._downsample_factors = np.zeros((self.info['n_levels'],), dtype=int)

        self.magnif_converter =  Magnification(
            self.info['objective_power'],
            mpp=0.5*(self.info['mpp_x'] + self.info['mpp_y']),
            level=0,
            magnif_step=float(self.info['magnification_step'])
        )

        for lv in range(self.info['n_levels']):
            s = self.magnif_converter.magnif_step ** lv
            self._pyramid_levels[0, lv] = slide_src.level_dimensions[lv][0]
            self._pyramid_levels[1, lv] = slide_src.level_dimensions[lv][1]
            self._pyramid_mpp[0, lv] = s * self.info['mpp_x']
            self._pyramid_mpp[1, lv] = s * self.info['mpp_y']
            self._downsample_factors[lv] = s

        return

    @property
    def level_count(self) -> int:
        """Return the number of levels in the multi-resolution pyramid."""
        return self.info['n_levels']


    def downsample_factor(self, level:int) -> int:
        """Return the down-sampling factor (relative to level 0) for a given level."""
        if level < 0 or level >= self.level_count:
            return -1

        return int(self._downsample_factors[level])

    @property
    def get_native_magnification(self) -> float:
        """Return the original magnification for the scan."""
        return self.info['objective_power']

    @property
    def get_native_resolution(self) -> float:
        """Return the scan resolution (microns per pixel)."""
        return 0.5 * (self.info['mpp_x'] + self.info['mpp_y'])

    def get_level_for_magnification(self, mag: float, eps=1e-6) -> int:
        """Returns the level in the image pyramid that corresponds the given magnification.

        Args:
            mag (float): magnification
            eps (float): accepted error when approximating the level

        Returns:
            level (int) or -1 if no suitable level was found
        """
        if mag > self.info['objective_power'] or mag < 2.0**(1-self.level_count) * self.info['objective_power']:
            return -1

        lx = log(self.info['objective_power'] / mag, self.info['magnification_step'])
        k = np.where(np.isclose(lx, range(0, self.level_count), atol=eps))[0]
        if len(k) > 0:
            return int(k[0])   # first index matching
        else:
            return -1   # no match close enough

    def get_level_for_mpp(self, mpp: float):
        """Return the level in the image pyramid that corresponds to a given resolution."""
        return self.magnif_converter.get_level_for_mpp(mpp)

    def get_mpp_for_level(self, level: int):
        """Return resolution (mpp) for a given level in pyramid."""
        return self.magnif_converter.get_mpp_for_level(level)

    def get_magnification_for_level(self, level: int) -> float:
        """Returns the magnification (objective power) for a given level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            magnification (float)
            If the level is out of bounds, returns -1.0
        """
        if level < 0 or level >= self.level_count:
            return -1.0
        if level == 0:
            return self.info['objective_power']

        #return 2.0**(-level) * self.info['objective_power']
        return self.info['magnification_step'] ** (-level) * self.info['objective_power']

    def get_extent_at_level(self, level: int) -> Optional[ImageShape]:
        """Returns width and height of the image at a desired level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            (width, height) of the level
        """
        if level < 0 or level >= self.level_count:
            return None
        return ImageShape({
            'width': self._pyramid_levels[0, level],
            'height': self._pyramid_levels[1, level]
        })

    @property
    def pyramid_levels(self):
        return self._pyramid_levels

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def widths(self) -> np.array:
        # All widths for the pyramid levels
        return self._pyramid_levels[0, :]

    @property
    def heights(self) -> np.array:
        # All heights for the pyramid levels
        return self._pyramid_levels[1, :]

    # def extent(self, level: int = 0) -> tuple[Any, ...]:
    #     # width, height for a given level
    #     return tuple(self._pyramid_levels[:, level])

    def level_shape(self, level: int = 0) -> ImageShape:
        return ImageShape(
            {
                'width': int(self._pyramid_levels[0, level]),
                'height': int(self._pyramid_levels[1, level])
            }
        )

    def between_level_scaling_factor(self, from_level: int, to_level: int) -> float:
        """Return the scaling factor for converting coordinates (magnification)
        between two levels in the MRI.

        Args:
            from_level (int): original level
            to_level (int): destination level

        Returns:
            float
        """
        f = self._downsample_factors[from_level] / self._downsample_factors[to_level]

        return float(f)

    def convert_px(self, point, from_level, to_level) -> Tuple[int, int]:
        """Convert pixel coordinates of a point from <from_level> to
        <to_level>

        Args:
            point (tuple): (x,y) coordinates in <from_level> plane
            from_level (int): original image level
            to_level (int): destination level

        Returns:
            x, y (float): new coordinates - no rounding is applied
        """
        if from_level == to_level:
            return point  # no conversion is necessary
        x, y = point
        f = self.between_level_scaling_factor(from_level, to_level)
        x *= f
        y *= f

        return int(x), int(y)

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

        x0_0, y0_0 = self.convert_px((x0, y0), level, 0)
        img = self._slide.read_region((x0_0, y0_0), level, (width, height))
        img = np.array(img)

        if img.shape[2] == 4:  # has alpha channel, usually used for masking
            # fill with background
            mask = img[..., -1].squeeze()
            img[mask == 0, 0:4] = self.info['background']
            img = img[..., :-1]

        return img.astype(as_type)

    def get_plane(self, level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a whole plane from the image pyramid and return it as a Numpy array.

        Args:
            level (int): pyramid level to read
            as_type: type of the pixels (default numpy.uint8)

        Returns:
            a numpy.ndarray
        """
        if level < 0 or level >= self.level_count:
            raise RuntimeError("requested level does not exist")

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
        x1, y1 = min(x1 + border, self.extent(level)[0]), \
            min(y1 + border, self.extent(level)[1])
        # Shift the annotation such that (0,0) will correspond to (x0, y0)
        contour = sha.translate(contour, -x0, -y0)

        # Read the corresponding region
        img = self.get_region_px(x0, y0, x1 - x0, y1 - y0, level, as_type=as_type)

        # mask out the points outside the contour
        for i in np.arange(img.shape[0]):
            # line mask
            lm = np.zeros((img.shape[1], img.shape[2]), dtype=img.dtype)
            j = [_j for _j in np.arange(img.shape[1]) if shg.Point(_j, i).within(contour)]
            lm[j,] = 1
            img[i,] = img[i,] * lm

        return img
##

#####
class MRI(object):
    """MultiResolution Image - a simple and convenient interface to access pixels from a
    pyramidal image. The image is supposed to by stored in ZARR format (see README).

    Args:
        path (str): folder with a ZARR store providing the levels (indexed 0, 1, ...)
        of the pyramid

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
        _downsample_factors (vector): precomputed down-sampling factors
        _mag (Magnification): magnification converter
    """
    _path = None
    _info = None
    _pyramid_levels = None

    def __init__(self, path: Union[str|Path]):
        self._path = Path(path) if path is str else path

        with zarr.open(self._path, mode='r') as z:
            self._info = z.attrs

        self._pyramid_levels = np.array(self._info["extent"], dtype=int)
        self._downsample_factors = np.array([2**i for i in range(self._info['max_level'])], dtype=int)
        self._mag = Magnification(self._info["objective_power"],
                                  0.5 * (self._info["mpp_x"] + self._info["mpp_y"]),
                                  level = 0,
                                  n_levels = self._info["max_level"],
                                  magnif_step = self._info["mag_step"])


    @property
    def path(self) -> Path:
        return self._path

    @property
    def widths(self) -> np.array:
        # All widths for the pyramid levels
        return self._pyramid_levels[0,:]

    @property
    def heights(self) -> np.array:
        # All heights for the pyramid levels
        return self._pyramid_levels[1,:]

    def extent(self, level:int=0) -> (int, int):
        # width, height for a given level
        return tuple(self._pyramid_levels[:, level])

    @property
    def nlevels(self) -> int:
        return self._info["max_level"]

    def between_level_scaling_factor(self, from_level:int, to_level:int) -> float:
        """Return the scaling factor for converting coordinates (magnification)
        between two levels in the MRI.

        Args:
            from_level (int): original level
            to_level (int): destination level

        Returns:
            float
        """
        f = self._downsample_factors[from_level] / self._downsample_factors[to_level]

        return float(f)

    def convert_px(self, point, from_level, to_level):
        """Convert pixel coordinates of a point from <from_level> to
        <to_level>

        Args:
            point (tuple): (x,y) coordinates in <from_level> plane
            from_level (int): original image level
            to_level (int): destination level

        Returns:
            x, y (float): new coordinates - no rounding is applied
        """
        if from_level == to_level:
            return point  # no conversion is necessary
        x, y = point
        f = self.between_level_scaling_factor(from_level, to_level)
        x *= f
        y *= f

        return x, y


    def get_region_px(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int=0, as_type=np.uint8) -> np.array:
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

        img = da.from_zarr(self.path, component=str(level), dtype=as_type)

        return img


    def get_plane(self, level: int = 0, as_type=np.uint8) -> da.array:
        """Read a whole plane from the image pyramid and return it as a Numpy array.

        Args:
            level (int): pyramid level to read
            as_type: type of the pixels (default numpy.uint8)

        Returns:
            a numpy.ndarray
        """
        if level < 0 or level >= self.nlevels:
            raise RuntimeError("requested level does not exist")

        img = da.from_zarr(self.path, component=str(level), dtype=as_type)

        return img


    def get_polygonal_region_px(self, contour: shg.Polygon, level: int,
                                border: int=0, as_type=np.uint8) -> da.array:
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
        x0, y0 = max(0, x0-border), max(0, y0-border)
        x1, y1 = min(x1+border, self.extent(level)[0]), \
                 min(y1+border, self.extent(level)[1])
        # Shift the annotation such that (0,0) will correspond to (x0, y0)
        contour = sha.translate(contour, -x0, -y0)

        # Read the corresponding region
        img = self.get_region_px(x0, y0, x1-x0, y1-y0, level, as_type=as_type)

        # Prepare mask
        mask = np.zeros(img.shape[:2], dtype=as_type)
        add_region(mask, shapely.get_coordinates(contour))

        # Apply mask
        img = apply_mask(img, mask)

        return img
##

#####
class NumpyImage:
    """This is merely a namespace for collecting a number of useful
    functions that are applied to images stored as Numpy arrays.
    Usually, such an image -either single channel or 3(4) channels -
    is stored as a H x W (x C) array, with H (height) rows and W (width)
    columns. C=3 or 4.
    """

    @staticmethod
    def width(img: np.ndarray) -> int:
        return img.shape[1]

    @staticmethod
    def height(img: np.ndarray) -> int:
        return img.shape[0]

    @staticmethod
    def nchannels(img: np.ndarray) -> int:
        if img.ndim > 2:
            return img.shape[2]
        else:
            return 1

    @staticmethod
    def is_empty(img: np.array, empty_level: float=0) -> bool:
        """Is the image empty?

        Args:
            img (numpy.ndarray): image
            empty_level (int/numeric): if the sum of pixels is at most this
                value, the image is considered empty.

        Returns:
            bool
        """

        return img.sum() <= empty_level

    @staticmethod
    def is_almost_white(img: np.array, almost_white_level: float=254, max_stddev: float=1.5) -> bool:
        """Is the image almost white?

        Args:
            img (numpy.ndarray): image
            almost_white_level (int/numeric): if the average intensity per channel
                is above the given level, decide "almost white" image.
            max_stddev (float): max standard deviation for considering the image
                almost constant.

        Returns:
            bool
        """

        return (img.mean() >= almost_white_level) and (img.std() <= max_stddev)
##-

#### Utilities
def R_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 0]


def G_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 1]


def B_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 2]

def rgb2ycbcr(im: np.ndarray) -> np.ndarray:
    """
    RGB2YCBCR: converts an RGB image into YCbCr (YUV) color space.

    :param im: numpy.ndarray
      [m x n x 3] image
    """

    if im.ndim != 3:
        raise ValueError('Input image must be RGB.')
    h, w, c = im.shape
    if c != 3:
        raise ValueError('Input image must be a 3-channel (RGB) image.')

    if im.dtype != np.uint8:
        im = img_as_uint(im)

    ycc = np.array([[0.257, 0.439, -0.148],
                    [0.504, -0.368, -0.291],
                    [0.098, -0.071, 0.439]])

    im = im.reshape((h * w, c))

    r = np.dot(im, ycc).reshape((h, w, c))
    r[:, :, 0] += 16
    r[:, :, 1:3] += 128

    im_res = np.array(np.round(r), dtype=im.dtype)

    return im_res
## end rgb2ycbcr


def ycbcr2rgb(im: np.ndarray) -> np.ndarray:
    """
    YCBCR2RGB: converts an YCbCr (YUV) in RGB color space.

    :param im: numpy.ndarray
      [m x n x 3] image
    """

    if im.ndim != 3:
        raise ValueError('Input image must be YCbCr.')
    h, w, c = im.shape
    if c != 3:
        raise ValueError('Input image must be a 3-channel (YCbCr) image.')

    if im.dtype != np.uint8:
        im = img_as_uint(im)

    iycc = np.array([[1.164, 1.164, 1.164],
                     [0, -0.391, 2.018],
                     [1.596, -0.813, 0]])

    r = im.reshape((h * w, c))

    r[:, 0] -= 16.0
    r[:, 1:3] -= 128.0
    r = np.dot(r, iycc)
    r[r < 0] = 0
    r[r > 255] = 255
    r = np.round(r)
    # x = r[:,2]; r[:,2] = r[:,0]; r[:,0] = x

    im_res = np.array(r.reshape((h, w, c)), dtype=np.uint8)

    return im_res
## end ycbcr2rgb

##-
class NumpyJSONEncoder(json.JSONEncoder):
    """Provides an encoder for Numpy types for serialization."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)
##

##-
def wsi2zarr(
        wsi_path: Union[str|Path],
        dst_path: Union[str|Path],
        crop: Optional[Tuple[int,int,int,int]|bool],
        band_size: Optional[int]=1528,
) -> None:
    """
    Converts a WSI file to pyramidal ZARR format.

    :param wsi_path: source file path.
    :param dst_path: destination file path (normally a .zarr folder).
    :param crop: either bool to control auto-crop or (x0, y0, width, height) for the crop region
    :param band_size: band height for processed regions
    :return: None
    """
    if not isinstance(wsi_path, Path):
        wsi_path = Path(wsi_path)
    if not isinstance(dst_path, Path):
        dst_path = Path(dst_path)
        if not dst_path.exists():
            dst_path.mkdir(parents=True, exist_ok=True)

    wsi = WSI(wsi_path)

    # initially, whole image
    x0, y0, width, height = (0, 0, wsi.info["width"], wsi.info["height"])

    if isinstance(crop, bool):
        if crop and wsi.info['roi'] is not None:
            x0, y0, width, height = (wsi.info['roi']['x0'],
                                     wsi.info['roi']['y0'],
                                     wsi.info['roi']["width"],
                                     wsi.info['roi']["height"])
    else:
        if crop is not None:
            x0, y0, width, height = crop
            x0 = max(0, min(x0, wsi.info["width"]))
            y0 = max(0, min(y0, wsi.info["height"]))
            width = min(width, wsi.info["width"] - x0)
            height = min(height, wsi.info["height"] - y0)

    levels = np.zeros((2, wsi.level_count), dtype=np.int64)

    with (zarr.open_group(str(dst_path), mode='w') as root):
        for i in range(wsi.level_count):
            # copy levels from WSI, band by band...
            # -level i crop region:
            cx0 = int(floor(x0 / wsi.downsample_factor(i)))
            cy0 = int(floor(y0 / wsi.downsample_factor(i)))
            cw = int(floor(width / wsi.downsample_factor(i)))
            ch = int(floor(height / wsi.downsample_factor(i)))
            #print("ZARR writing crop: ", cx0, cy0, cw, ch, i)

            im = pyvips.Image.new_from_file(str(wsi_path), level=i, autocrop=False)
            im = im.crop(cx0, cy0, cw, ch)
            im = im.flatten()

            shape = (ch, cw, 3)  # YXC axes
            levels[:, i] = (cw, ch)

            arr = root.zeros('/'+str(i), shape=shape, chunks=(4096, 4096, None), dtype="uint8")
            n_bands = ch // band_size
            incomplete_band = shape[0] % band_size
            for j in range(n_bands):  # by horizontal bands
                buf = im.crop(0, j * band_size, cw, band_size).numpy()
                arr[j * band_size : (j + 1) * band_size] = buf
                # arr[j * band_size:(j + 1) * band_size, ...] = \
                #     wsi.get_region_px(cx0, cy0+j*band_size, cw, band_size, as_type=np.uint8)

            if incomplete_band > 0:
                buf = im.crop(0, n_bands * band_size, cw, incomplete_band).numpy()
                arr[n_bands * band_size : n_bands * band_size + incomplete_band] = buf
                # arr[n_bands * band_size: n_bands * band_size + incomplete_band, ...] = \
                #     wsi.get_region_px(cx0, n_bands*band_size, cw, incomplete_band, as_type=np.uint8)
        root.attrs["max_level"] = wsi.level_count
        root.attrs["channel_names"] = ["R", "G", "B"]
        root.attrs["dimension_names"] = ["y", "x", "c"]
        root.attrs["mpp_x"] = wsi.info['mpp_x']
        root.attrs["mpp_y"] = wsi.info["mpp_y"]
        root.attrs["mag_step"] = int(wsi.info['magnification_step'])
        root.attrs["objective_power"] = wsi.info['objective_power']
        root.attrs["extent"] = levels.tolist()

    return
##