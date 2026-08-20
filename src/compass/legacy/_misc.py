# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import numpy as np
import simplejson as json

from skimage.util import img_as_uint
from pydantic import BaseModel

class ImageShape(BaseModel):  # = NewType("ImageShape", dict[str, int])
    width: int
    height: int


class Px(BaseModel):  # Pixel: int coords
    x: int
    y: int


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
    def is_empty(img: np.ndarray, empty_level: float = 0) -> bool:
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
    def is_almost_white(img: np.ndarray, almost_white_level: float = 254, max_stddev: float = 1.5) -> bool:
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

    r = im.reshape((h * w, c)).astype(np.float32)

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
