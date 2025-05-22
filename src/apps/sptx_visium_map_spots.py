# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

# SPTX_VISIUM_MAP_SPOTS: after registering the "high resolution" images to
# a WSI (see sptx_visium_match_wsi.py), we use the computed transformations
# to:
# - extract for each region the corresponding area from WSI and save it as
#   an individual whole slide image (a pyramidal image in ZARR format);
# - transform the coordinates of all spots from the VISIUM data to match
#   the extracted WSI

import simplejson as json
import numpy as np
import cv2
import configargparse as opt
from datetime import datetime
import hashlib
from pathlib import Path
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.exposure import match_histograms
import skimage.draw as skid
import pandas as pd
import wsitk_annot
import joblib

from compass.core import WSI, wsi2zarr, MRI, Px
from compass.sample import SampleManager

_time = datetime.now()
__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "0.1"
__description__ = {
    'name': 'st_match_tissue_parts',
    'unique_id' : hashlib.md5(str.encode("sptx_visium_map_spots" + __version__)).hexdigest(),
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': ['None'],
    'output': ['None'],
    'params': dict()
}


def find_patch(tmpl: np.ndarray, img: np.ndarray,
               angles: np.ndarray = np.linspace(0, 360, 37),
               scales: np.ndarray = np.linspace(0.8, 1.2, 9)) -> dict:
    """Find a template image within a larger image using template matching with rotation and scaling.

    Args:
        tmpl: Template image (grayscale) to search for
        img: Larger image (grayscale) to search within
        angles: Array of rotation angles in degrees to test (default: 37 angles from 0 to 360)
        scales: Array of scaling factors to test (default: 9 values from 0.8 to 1.2)

    Returns:
        dict: Dictionary containing best match information with keys:
            - 'score': Correlation coefficient (-1 to 1)
            - 'angle': Best rotation angle in degrees
            - 'scale': Best scaling factor
            - 'loc': (x,y) tuple of top-left corner position in img
    """

    best = {'score': -1.0, 'angle': None, 'scale': None, 'loc': None}

    for scale in scales:
        # resize template
        sz = (int(tmpl.shape[1] * scale), int(tmpl.shape[0] * scale))
        if sz[0] < 4 or sz[1] < 4:  # too small to match
            continue
        scaled = cv2.resize(tmpl, sz, interpolation=cv2.INTER_LINEAR)

        for angle in angles:
            # rotate about its center
            M = cv2.getRotationMatrix2D((sz[0] / 2, sz[1] / 2), angle, 1.0)
            rot = cv2.warpAffine(scaled, M, sz,
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

            # template match
            res = cv2.matchTemplate(img, rot, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)

            if score > best['score']:
                best.update({
                    'score': score,
                    'angle': angle,
                    'scale': scale,
                    'loc': loc  # top-left corner in img
                })
    return best  # best transformation


def refine_spot_positions(hr_spots, radius_spots, hr_img, wsi_spots, wsi_img, initial_scale_factor, hr_expand_factor=2.0,
                          wsi_expand_factor=3.0, min_score=0.5):
    """Refine the positions of spots in the WSI space by template matching. Work at the resolution of the "high
    resolution" image from the Visium dataset.

    Args:
        hr_spots: numpy.ndarray, shape (n,2)
            Coordinates (x,y) for each of n spots in high resolution image
        radius_spots: numpy.ndarray, shape (n,)
            Radius values for each of n spots
        hr_img: numpy.ndarray
            High resolution image as it comes from Visium
        wsi_spots: numpy.ndarray, shape (n,2)
            Coordinates (x,y) for each spot in the WSI, at the level closest to hr_img resolution,
            already transformed by the affine matrix H
        wsi_img: numpy.ndarray
            Image from WSI closest in resolution to the hr_image
        initial_scale_factor: float
            Scaling factor estimated from H (the mapping of hr_img onto wsi_img)
        hr_expand_factor: float, optional (default=2)
            Multiplier to enlarge the region around the spot in hr_img; defines the spot patch size
        wsi_expand_factor: float, optional (default=3)
            Multiplier to enlarge the region around the initial predicted spot position
            in wsi_img; defines the search region for the final spot position
        min_score: float, optional (default=0.5)
            Minimum score for a spot position to be updated (0...1)

    Returns:
        tuple:
            - numpy.ndarray: Updated spot coordinates in WSI space
            - numpy.ndarray: Updated spot radii in WSI space
    """
    assert hr_spots.shape[0] == wsi_spots.shape[0]

    updated_spots = wsi_spots.copy()  # overwrite updated positions # np.zeros_like(hr_spots)
    updated_radius = initial_scale_factor * radius_spots.copy() #np.zeros_like(radius_spots)

    patches = [
        (
            (int(hr_spots[k, 1] - hr_expand_factor * radius_spots[k]),
             int(hr_spots[k, 1] + hr_expand_factor * radius_spots[k]),
             int(hr_spots[k, 0] - hr_expand_factor * radius_spots[k]),
             int(hr_spots[k, 0] + hr_expand_factor * radius_spots[k])),
            (int(wsi_spots[k, 1] - wsi_expand_factor * updated_radius[k]),
             int(wsi_spots[k, 1] + wsi_expand_factor * updated_radius[k]),
             int(wsi_spots[k, 0] - wsi_expand_factor * updated_radius[k]),
             int(wsi_spots[k, 0] + wsi_expand_factor * updated_radius[k]))
        )
        for k in range(hr_spots.shape[0])
    ]
    updates = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(find_patch)(
            hr_img[s[0]:s[1], s[2]:s[3]],
            wsi_img[w[0]:w[1], w[2]:w[3]],
            scales = [initial_scale_factor]
            #scales=np.linspace(initial_scale_factor - 0.1, initial_scale_factor + 0.1, 5)
        )
        for s, w in patches
    )

    u = 0
    for k, res in enumerate(updates):
        if res['score'] > min_score:
            x, y = res['loc']  # top-left of the rectangle matches that the spot region
            sp, wp = patches[k]  # coords of patches (source, wsi)
            x += wp[2] # in wsi_img coords
            y += wp[0] # in wsi_img coords
            h, w = int(wp[1]-wp[0]), int(wp[3]-wp[2])  # matched region shape (wsi_coords)
            # the new center for the spot is the matched region's center
            updated_spots[k, 0] = int(x + w / 2.0)  # new x-coord
            updated_spots[k, 1] = int(y + h / 2.0)  # new y-coord
            updated_radius[k] = radius_spots[k] * res['scale']  # new radius in wsi_img space
            u += 1

    print(f"Updated {u} spots")

    return updated_spots, updated_radius


def main() -> int:
    p = opt.ArgumentParser(description="Extract original resolution images and re-maps the spots.")
    p.add_argument(
        "--input", action="store", help="JSON file with the results from SPTX_VISIUM_MATCH_WSI", required=True,
    )
    p.add_argument(
        "--part", action="store", help="which part to process (A,...,D)", type=str,   required=True,
    )
    p.add_argument(
        "--spots", action="store", help=".feather file with spots positions (DataFrame with columns x, y, radius)",
        type=str, required=True
    )
    p.add_argument(
        "--expression", action="store", help=".feather file with expression data frame",
        type=str, required=True
    )
    p.add_argument(
        "--out", action="store",
        help="path where to store the results. (<path>.cp/pyramid_0.zarr and <path>/annot_0). Last part of path is taken as sample name",
        required=True
    )
    p.add_argument(
        "--transform_hires", action="store_true", help="transform HIRES images from VISIUM?", required=False,
    )
    p.add_argument(
        "--verify", action="store_true",
        help="save a low resolution images to help verify the results?", required=False,
    )

    args = p.parse_args()
    __description__['params'] = vars(args)
    __description__['input'] = [str(args.input)]
    __description__['output'] = [str(args.out)]

    with open(args.input, "r") as f:
        # read registration results
        registration = json.load(f)

    if args.part not in registration:
        raise Exception(f"Part {args.part} not in registration results")

    wsi_file = registration["__description__"]["input"][0]
    hr_file = registration[args.part]["image"]
    crop = registration[args.part]["wsi_region"]
    H = np.array(registration[args.part]["transform"]).reshape((3,3))

    out_path = Path(args.out)
    sample_name = out_path.stem
    out_path = out_path.parent
    mgr = SampleManager(out_path, sample_name, mode="w", overwrite_if_exists=True)

    wsi = WSI(wsi_file)
    pp = wsi.convert_px(
        Px(x=crop["x0"], y=crop["y0"]),
        from_level=registration[args.part]["wsi_level"],
        to_level=0
    )
    x0, y0 = pp.x, pp.y
    pp = wsi.convert_px(
        Px(x=crop["x1"], y=crop["y1"]),
        from_level=registration[args.part]["wsi_level"],
        to_level=0
    )
    x1, y1 = pp.x, pp.y

    # write the ROI from WSI into a pyramidal ZARR:
    wsi2zarr(wsi_file, mgr.get_pyramid_path(0), crop=(x0, y0, x1-x0, y1-y0))

    # process annotations (spots and gene expression)
    spots = pd.read_feather(args.spots)
    expr_data = pd.read_feather(args.expression)

    s_xy = spots[["x", "y"]].values        # spot centres
    s_rd = spots[["radius"]].values[:,-1]  # just a vector
    s_xy_t = cv2.transform(s_xy[None, ...], H)[0]  # new centres

    # Compute the scaling factor for the radius as the mean of scaling on x and
    # y, respectively. Consider the transformation of a triangle (0,0), (0,1), (1,0)
    # and compute the length of the new sides corresponding to original unit sides.
    v1 = np.array([[0,0], [0,1], [1,0]], dtype=np.float64)
    v1t = cv2.transform(v1[None,...], H)[0]
    rsf = np.linalg.norm(
        [v1t[1] - v1t[0], v1t[2] - v1t[0]], axis=1
    ).mean()  # radius scaling factor
    rd = s_rd * rsf

    # Refine estimated spot positions by comparing the high-resolution image to the
    # corresponding level of the whole slide image.
    hr_img = imread(hr_file)[..., 0:3]
    hr_img = img_as_ubyte(rgb2gray(hr_img, channel_axis=2))

    mri = MRI(mgr.get_pyramid_path(0))
    wsi_img = np.array(
        mri.get_plane(registration[args.part]["wsi_level"], as_type=np.uint8)
    )
    wsi_img = img_as_ubyte(rgb2gray(wsi_img, channel_axis=2))

    # match histograms:
    hr_img = np.floor(match_histograms(hr_img, wsi_img)).astype(np.uint8)
    new_spot_xy, new_spot_rd = refine_spot_positions(
        s_xy, s_rd, hr_img,
        s_xy_t, wsi_img,
        rsf, hr_expand_factor=2.5, wsi_expand_factor=2.75, min_score=0.8
    )
    s_xy_t = new_spot_xy
    rd = new_spot_rd

    #print('Scaling factor for radius: ', sf, ' (radius in pixels: ', s_rd[0], ' -> ', rd[0], ' )')
    # Initial annotation is at the resolution/level at which the high-resolution image was matched
    # with the WSI
    w, h = mri.extent(registration[args.part]["wsi_level"])
    annot = wsitk_annot.Annotation(
        name="transcriptomics",
        image_shape={'width': w, 'height': h},
        # the initial mpp is at the registration[args.part]["wsi_level"] level:
        mpp = mri.get_mpp_for_level(registration[args.part]["wsi_level"])
        # mpp= wsi.get_native_resolution
    )
    # Create the corresponding annotations
    genes = expr_data.apply(lambda row: row[row > 0].index.tolist(), axis=1) # detected genes, for each spot (list of lists)
    for k in range(len(spots)):
        annot.add_annotation_object(
            wsitk_annot.Circle((s_xy_t[k,0], s_xy_t[k,1]), # (x,y)
                               rd[k],
                               name=f"spot_{k}",
                               data=expr_data.loc[k, genes[k]].tolist())
        )

    # bring annotation to the highest mpp
    annot.set_resolution(mri.get_mpp_for_level(0))

    # save the annotation
    mgr.add_annotation(annot, 0, "Visium spots")

    if args.transform_hires:
        p = wsi.get_region_px(
            crop["x0"], crop["y0"],
            crop["x1"] - crop["x0"], crop["y1"] - crop["y0"],
            level=registration[args.part]["wsi_level"]
        )  # from WSI
        hrimg = imread(hr_file)[...,0:3]
        hrimg = cv2.warpPerspective(hrimg, H, (p.shape[1], p.shape[0]), flags=cv2.INTER_CUBIC)

        res_img = cv2.addWeighted(p, 0.5, hrimg, 0.5, 0)

        imsave(mgr.full_path / "hires.jpeg", hrimg)
        imsave(mgr.full_path / "combined_images.jpeg", res_img)

    if args.verify:
        # use the newly generated data to visualize the spots
        # mri = MRI(mgr.get_pyramid_path(0))  # already open
        # use the same magnification as for hires:
        img = np.array(
            mri.get_plane(registration[args.part]["wsi_level"] , as_type=np.uint8)
        )
        annot.set_resolution(
            mri.get_mpp_for_level(registration[args.part]["wsi_level"])
        )
        for a in annot._annots['base']:
            if a._annotation_type != "CIRCLE":
                continue
            #print(f"center = {a.center}, radius = {a.radius}")
            r, c = skid.circle_perimeter(int(a.center[1]), int(a.center[0]), int(a.radius), shape=img.shape)
            img[r, c] = (10, 255, 10)

        imsave(mgr.full_path / "verify.jpeg", img)


    return 0


if __name__ == '__main__':
    main()

