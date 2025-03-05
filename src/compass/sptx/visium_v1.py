# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

##
# VISIUM_V1: Visium v1-specific image processing.
###

#
# Given the "high resolution" images associated with tissue sampling spots (where
# the RNA-Seq was performed) and a whole slide image (WSI) of the microscope slide
# (with the 4 tissue pieces), the goal is to register the "high resolution"
# images onto the WSI so that the tissue regions corresponding to each spot can
# be extracted at the highest magnification.
##`

from ..core import NumpyImage as npi, WSI
from ..tissue import detect_foreground

from typing import Union
from math import ceil, log2

from skimage.morphology import disk, binary_dilation
from skimage.filters import threshold_sauvola, median, sobel_h, sobel_v
from skimage.transform import rescale, probabilistic_hough_line
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity
from skimage.io import imread, imsave

import cv2 as cv
import numpy as np

VISIUM1 = {
    'vendor': '10x Genomics',
    'name': 'Visium',
    'version': 1.0,
    'layout': '1x4',  # tissue parts: 1 row, 4 columns
    # Some estimated dimensions for each of the 4 regions:
    # external size (covers also the control spots) and internal
    # extent.
    'roi_width': 7650,           # estimated, microns
    'roi_height': 7570,          # estimated, microns
    'roi_internal_width': 6960,  # estimated, microns
    'roi_internal_height': 6610, # estimated, microns
}


def register_patch(img1: np.array, img2: np.array,
                   resize_factor:float=1.0/8.0,
                   min_matching_points: int = 10) -> Union[np.array, None]:
    # Register an image onto the reference image and return the
    # affine transformation as a Numpy array.
    # img1: reference image
    # img2: sensed image (patch)
    # resize_factor: work resolution for registration

    img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    img1_rs = cv.resize(
        img1,
        (0,0), fx=resize_factor, fy=resize_factor
    )
    img2_rs = cv.resize(
        img2,
        (0,0), fx=resize_factor, fy=resize_factor
    )

    # Initiate SIFT detector
    sift_detector = cv.SIFT_create()

    # Find the key points and descriptors with SIFT on the lower resolution images
    kp1, des1 = sift_detector.detectAndCompute(img1_rs, None)
    kp2, des2 = sift_detector.detectAndCompute(img2_rs, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter out poor matches
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    matches = good_matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    if points1.shape[0] <= min_matching_points:
        return None
    H, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # Get low-res and high-res sizes
    low_height, low_width = img1_rs.shape
    height, width = img1.shape
    low_size = np.array(
        [
            [0, 0],
            [0, low_height],
            [low_width, low_height],
            [low_width, 0]
        ], dtype=np.float32
    )
    high_size = np.array(
        [
            [0, 0],
            [0, height],
            [width, height],
            [width, 0]
        ], dtype=np.float32
    )

    # Compute scaling transformations
    scale_up = cv.getPerspectiveTransform(low_size, high_size)
    scale_down = cv.getPerspectiveTransform(high_size, low_size)

    # Combine the transformations. Remember that the order of the transformation
    # is reversed when doing matrix multiplication
    # so this is actually scale_down -> H -> scale_up
    h_and_scale_up = np.matmul(scale_up, H)
    scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

    # Warp image 1 to align with image 2
    #img1Reg = cv.warpPerspective(
    #            img1,
    #            scale_down_h_scale_up,
    #            (img2.shape[1], img2.shape[0])
    #          )
    return scale_down_h_scale_up

def get_affine_transformation_per_region(
        slide: WSI,
        high_res_images: list,
        min_matching_quality: float = 0.5
) -> dict:
    """
    Registers high resolution images from Visium data set to a whole slide image of
    the pathology slide used in transcriptomics.

    The 4 tissue parts are supposed to be arranged in a single row and the WSI to be
    in the "landscape" orientation:

    +-----------+-----------+-----------+-----------+
    |     A     |     B     |     C     |     D     |
    +-----------+-----------+-----------+-----------+

    The Visium (v.1.0) data set is supposed to contain 4 "high resolution" images of
    each of the regions A,...,D, which define the coordinate system for transcriptomics
    spots. This function will compute the transformation mapping/registering each "high
    resolution" image onto the slide regions A,...,D, at the slide native magnification
    (typically 20x or 40x). The transformations will be returned as a dict() with
    keys "A", ..., "D".

    Args:
        slide (WSI): Slide access object.
        high_res_images (list): Names (paths) of the high resolution images.
        min_matching_quality (float): A scalar [0, 1] indicating the minimum
            requited matching quality between a high resolution image and a
            slide region to call a match.
    Returns:
        dict: A dictionary with keys A,..., D with matching information:
            dict['A',...,'D'] = {
            'image': path to high resolution image,
            'wsi_region': region of interest from WSI,
            'transform': 3x3 transformation matrix
            }
    """
    # from micrometers to pixels:
    visium_roi_width = int(7650 / float(slide.info['mpp_x'])) # [um] / [um/px]
    visium_roi_height = int(7570 / float(slide.info['mpp_y']))
    # visium_roi_internal_width = int(6960 / float(slide.info['mpp_x']))
    # visium_roi_internal_height = int(6610 / float(slide.info['mpp_y']))

    # visium_roi_horiz_band = int(ceil((visium_roi_height - visium_roi_internal_height)/2))
    # visium_roi_vert_band = int(ceil((visium_roi_width - visium_roi_internal_width)/2))

    min_obj_size = {'60.0': 500, '15.0': 1500, '3.75': 50000, '1.8': 100000, '0.9': 500000}
    WORK_MPP_1: float = 15.0  # mpp
    WORK_MPP_2: float = 60.0  # mpp

    level_1 = slide.get_level_for_mpp(WORK_MPP_1)
    level_2 = slide.get_level_for_mpp(WORK_MPP_2)

    #img_src = MRI(slide)

    img = slide.get_plane(level=level_1)
    mask, _ = detect_foreground(img, method='simple-he', min_area=min_obj_size[str(WORK_MPP_2)])

    mask2 = np.zeros_like(mask)
    _ = median(mask.astype(np.uint8), footprint=disk(5), out=mask2, behavior='rank')
    _ = binary_dilation(mask2, footprint=disk(15), out=mask)
    mask2 = rescale(mask, WORK_MPP_1/WORK_MPP_2)

    img = slide.get_plane(level=level_2)
    img[mask2] = 0

    img = rgb2lab(img, channel_axis=2)[...,0].squeeze()
    thr = threshold_sauvola(img, window_size=3)
    mask = np.logical_not((img > thr).squeeze().astype(bool))

    mask[mask2] = 0

    _ = median(mask.astype(np.uint8), footprint=disk(7), out=mask2, behavior='rank')
    mask = np.not_equal( sobel_h(mask2) + sobel_v(mask2), 0)

    lines = probabilistic_hough_line(
        mask, line_length=80, line_gap=80,
        theta=np.array([0, np.pi/2])
    )

    # local dimensions
    roi_width = int(ceil(visium_roi_width / 2**level_2))
    roi_height = int(ceil(visium_roi_height / 2**level_2))

    lines = np.array(lines).reshape((len(lines),4))

    # find the lines defining the top, bottom, left, and right margins
    top = lines[:,1].max()
    bottom = lines[:,1].min()
    left = lines[:,0].max()
    right = lines[:,0].min()

    for ln in lines:
        x0, y0, x1, y1 = ln
        if y0 == y1:
            # horizontal line
            if abs(x1 - x0) < 2*roi_width: # line too short
                continue
            top = min(top, y0)
            bottom = max(bottom, y1)
        else: # there are only horizontal and vertical lines
            # if x0 == x1:
            # vertical line
            if abs(y1 - y0) < roi_height/2: # line too short
                continue
            left = min(left, x0)
            right = max(right, x1)

    # allow some space around current estimated ROI:
    top = max(0, top - 1)
    bottom = min(bottom + 1, npi.height(mask))
    left = max(0, left - 1)
    right = min(right + 1, npi.width(mask))

    # construct the ROIs for the 4 tissue patches
    patches = list()
    patch_width = int((right - left)/4)
    for k in range(4):
        patches.append({
            'x0': left + k*patch_width,
            'y0': top,
            'x1': left + (k+1)*patch_width,
            'y1': bottom
        })

    match_quality = np.zeros((4, len(high_res_images))) # cols: A, B, C, D; rows: high_res_images
    transforms = list()

    pw = patches[0]['x1'] - patches[0]['x0']
    ph = patches[0]['y1'] - patches[0]['y0']
    i = 0
    for hr_img_name in high_res_images:
        hr_img = imread(hr_img_name)[...,0:3] # drop alpha

        # estimate the scale from the WSI that is closest to the high resolution image:
        # img was at level_2 from the WSI
        f = (hr_img.shape[0] / ph + hr_img.shape[1] / pw) / 2  # <- average scale factor b/w hr and level_2 image
        level_3 = level_2 - int(ceil(log2(f)/log2(slide.info['magnification_step'])))
        # print(f"level_2 = {level_2}, level_3 = {level_3}, f = {f}")

        # Find the match and its parameters:
        # For all patches P:
        #   - register(P, hi_res_image) -> H affine transf. matrix
        #   - transform P onto hi_res_image -> tP
        #   - compute structural similarity index SSI(tP, hi_res_image)
        # return P* such that P* = arg max_P (SSI(tP, hi_res_image)
        j = 0
        for patch in patches: # regions of the WSI
            # get coords at high resolution
            ff = 2 ** level_3
            x0, y0, x1, y1 = ff*patch['x0'], ff*patch['y0'], ff*patch['x1'], ff*patch['y1']

            P = slide.get_region_px(x0, y0, x1 - x0, y1 - y0, level=level_3)  # from WSI
            H = register_patch(hr_img, P)

            if H is not None:
                tP = cv.warpPerspective(hr_img, H, (P.shape[1], P.shape[0]))
                match_quality[i,j] = structural_similarity(tP, P, win_size=35, channel_axis=2)
                imsave(f"p_{i}_{j}.jpg", P)

            transforms.append({
                'patch': dict(x0=x0, y0= y0, x1=x1, y1=y1),  # coords in level_3
                'H': H,
                'level': level_3
            }) # index: 4*i+j
            j += 1
        i += 1

    # Find best matches - if they exist:
    matches = dict(A=None, B=None, C=None, D=None)
    j = 0
    for k in matches:
        matches[k] = {
            'image': '',
            'wsi_region': transforms[j]['patch'],
            'wsi_level': -1,
            'transform': None
        }
        i = np.argmax(match_quality[:,j])
        if match_quality[i, j] >= min_matching_quality:
            matches[k]['image'] = high_res_images[i]
            matches[k]['transform'] = transforms[4*i+j]['H']
            matches[k]['wsi_level'] = transforms[4*i+j]['level']
        j += 1

    return matches
