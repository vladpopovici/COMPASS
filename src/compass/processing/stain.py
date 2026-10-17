# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Stain separation for H&E images.

Only :func:`rgb2he` is ported so far (needed by the ``simple-he`` tissue
detector); the rest of ``compass.legacy.stain`` (Macenko separation,
normalization) is still to be moved here.
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

__all__ = ["rgb2he"]

import numpy as np
from skimage.exposure import rescale_intensity

# Colour-deconvolution matrix, following
# http://web.hku.hk/~ccsigma/color-deconv/color-deconv.html
_HE_DECONV = np.array([[ 1.92129515,  1.00941672, -2.34107612],
                       [-2.34500192,  0.47155124,  2.65616872],
                       [ 1.21495282, -0.99544467,  0.2459345 ]])


def rgb2he(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the Haematoxylin and Eosin signal intensities and the
    residual of an RGB H&E image by colour deconvolution.

    Args:
        img: H x W x 3 RGB image.

    Returns:
        three H x W float arrays, each rescaled to [0, 1]: H, E and the
        background/residual component.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 RGB image, got {img.shape}")

    od = -np.log((img + 1.0) / img.max())          # optical density
    heb = od.reshape(-1, 3) @ _HE_DECONV.T          # (H*W, 3)
    heb = heb.reshape(img.shape)

    return (rescale_intensity(heb[..., 0], out_range=(0, 1)),
            rescale_intensity(heb[..., 1], out_range=(0, 1)),
            rescale_intensity(heb[..., 2], out_range=(0, 1)))
