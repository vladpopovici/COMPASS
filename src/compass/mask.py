# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

#
# COMPASS.MASK - various functions for creating and manipulating image
# masks (i.e. binary images of 0s and 1s).
#

__all__ = ['binary_mask', 'mask_to_external_contours', 'add_region', 'masked_points', 'apply_mask']

import dask.array as da
import numpy as np
import skimage.draw
import cv2
import shapely.geometry as shg


##-
def binary_mask(image: da.array, level: float, mode: str = 'exact') -> da.array:
    """Convert a single channel image into a binary mask by simple
    thresholding. This is a convenience function, smarter ways for
    binarizing images exist.

    Args:
        image (dask.array): a 2-dimensional array
        level (float): level for binarization
        mode (str): binarization strategy:
            'exact': pixels having value 'level' are set to 1, all others to 0
            'above': pixels having value > 'level' are set 1, all others to 0
            'below': pixels having value < 'level' are set 1, all others to 0

    Returns:
        a numpy.array of type 'uint8' and same shape as <image>
    """
    mode = mode.lower()
    if mode not in ['exact', 'above', 'below']:
        raise RuntimeError('unknown mode: ' + mode)

    if image.ndim != 2:
        raise RuntimeError('<image> must be single channel!')

    level = np.asarray([level], dtype=image.dtype)[0]  # need to convert to image dtype for == to work well

    mask = da.zeros_like(image, dtype=np.uint8)
    if mode == 'exact':
        mask[image == level] = 1
    elif mode == 'above':
        mask[image > level] = 1
    else:
        mask[image < level] = 1

    return mask
##-


##-
def mask_to_external_contours(mask: da.array, approx_factor: float = None, min_area: int = None) -> list:
    """Extract contours from a mask.

    Args:
        mask (dask.array): a binary image
        approx_factor (float): if provided, the contours are simplified by the
            given factor (see cv2.approxPolyDP() function)
        min_area (float): if provided, filters out contours (polygons) with an area
            less than this value

    Returns:
        a list of contours (shapely.Polygon)
    """
    mask = mask.compute()  # need to work with NumPy arrays
    m = np.pad(mask.astype(np.uint8), pad_width=2, mode="constant", constant_values=0)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if approx_factor is not None:
        cnt = [cv2.approxPolyDP(c, approx_factor * cv2.arcLength(c, True), True) for c in cnt]

    # remove eventual singleton dimensions and convert to Polygon (removing the padding of 2)
    cnt = [c.squeeze() - 2 for c in cnt]
    for c in cnt:
        c[c < 0] = 0

    cnt = [shg.Polygon(c) for c in cnt if len(c) >= 3]

    if min_area is not None:
        res = [p for p in cnt if p.geom.area >= min_area]
        return res

    return cnt
##-


##-
def add_region(mask: da.array, poly_line: np.ndarray) -> da.array:
    """Add a new masking region by setting to 1 all the
    pixels within the boundaries of a polygon. The changes are
    operated directly in the array.

    Args:
        mask (dask.array): an array possibly already containing
            some masked regions, to be updated
        poly_line (numpy.array): an N x 2 array with the (x,y)
            coordinates of the polygon vertices as rows

    Returns:
        a numpy.array - the updated mask
    """

    c, r = masked_points(poly_line, mask.shape)
    mask[r, c] = 1

    return mask
##-


##-
def masked_points(poly_line: np.ndarray, shape: tuple) -> tuple:
    """Compute the coordinates of the points that are inside the polygonal
    region defined by the vertices of the polygon.

    Args:
        poly_line (numpy.array): an N x 2 array with the (x,y)
            coordinates of the polygon vertices as rows
        shape (pair): the extend (width, height) of the rectangular region
            within which the polygon lies (typically image.shape[:2])
    Returns:
        a pair of lists (X, Y) where (X[i], Y[i]) are the coordinates of a
        point within the mask (polygonal region)
    """

    # check the last point to match the first one
    if not np.all(poly_line[0,] == poly_line[-1,]):
        poly_line = np.concatenate((poly_line, [poly_line[0,]]))

    # remember: row, col in polygon()
    r, c = skimage.draw.polygon(poly_line[:, 1], poly_line[:, 0], shape)

    return c, r
##-


##-
def apply_mask(img: da.array, mask: da.array) -> da.array:
    """Apply a mask to each channel of an image. Pixels corresponding to 0s in
    the mask will be set to 0. Changes are made in situ.

    Args:
        img (dask.array): an image array (height x width x no_of_channels)
        mask (dask.array): a mask as an array (height x width)

    Return:
        dask.array: the modified image
    """
    if mask.dtype is np.bool:
        mask = mask.astype(img.dtype)
    mask[mask > 0] = 1

    if img.ndim == 2:
        img *= mask
    else:
        img = img * mask[..., np.newaxis]

    return img
##-