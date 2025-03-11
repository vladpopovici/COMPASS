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
import pandas as pd
import wsitk_annot

from compass.core import WSI, wsi2zarr
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
    x0, y0 = wsi.convert_px(
        (crop["x0"], crop["y0"]),
        from_level=registration[args.part]["wsi_level"],
        to_level=0
    )
    x1, y1 = wsi.convert_px(
        (crop["x1"], crop["y1"]),
        from_level=registration[args.part]["wsi_level"],
        to_level=0
    )

    # write the ROI from WSI into a pyramidal ZARR:
    wsi2zarr(wsi_file, mgr.get_pyramid_path(0), crop=(x0, y0, x1-x0, y1-y0))

    # process annotations (spots and gene expression)
    spots = pd.read_feather(args.spots)
    expr_data = pd.read_feather(args.expression)

    s_xy = spots[["x", "y"]].values        # spot centres
    s_rd = spots[["radius"]].values[:,-1]  # just a vector
    s_xy_t = cv2.transform(s_xy[None, ...], H)[0]  # new centres

    # for radii: for each value transform the triangle [(0,0), (0,radius), (radius,0)]
    # and compute the new radius as the average of the transformed edges
    # ((0,0) -> (0, radius)), ((0,0) -> (radius, 0)). Thus, we construct the 3 vertices,
    # transform them and compute the distances.
    rd_xy = np.zeros((3 * s_rd.size, 2))
    rd_xy[np.arange(1, 3 * s_rd.size, 3), 1] = s_rd
    rd_xy[np.arange(2, 3 * s_rd.size, 3), 0] = s_rd
    rd_xy = cv2.transform(rd_xy[None,...], H)[0]
    d1 = np.linalg.norm(
        rd_xy[np.arange(0, 3 * s_rd.size, 3)] - rd_xy[np.arange(1, 3 * s_rd.size, 3)],
        axis=1
    )
    d2 = np.linalg.norm(
        rd_xy[np.arange(0, 3 * s_rd.size, 3)] - rd_xy[np.arange(2, 3 * s_rd.size, 3)],
        axis=1
    )
    rd = 0.5 * (d1 + d2)  # new radius for each spot

    annot = wsitk_annot.Annotation(
        name="transcriptomics",
        image_shape={'width': x1-x0, 'height': y1-y0},
        mpp= wsi.get_native_resolution
    )
    # Create the corresponding annotations
    for k in range(len(spots)):
        # find the genes that have been measured (>0) for this spot:
        genes = [g for g in expr_data.columns if expr_data[g][k] > 0]
        annot.add_annotation_object(
            wsitk_annot.Circle((s_xy_t[k,0], s_xy_t[k,1]),
                               rd[k], name=f"spot_{k}", data=expr_data[genes].iloc[k])
        )

    if args.transform_hires:
        p = wsi.get_region_px(
            crop["x0"], crop["y0"],
            crop["x1"] - crop["x0"], crop["y1"] - crop["y0"],
            level=registration[args.part]["wsi_level"]
        )  # from WSI
        hrimg = imread(hr_file)[...,0:3]
        hrimg = cv2.warpPerspective(hrimg, H, (p.shape[1], p.shape[0]), flags=cv2.INTER_CUBIC)
        imsave(mgr.full_path / "hires.jpeg", hrimg)

    annot.save(mgr.get_annotation_path(0))
    mgr.register_annotation(0, 0, "Visium spots")

    return 0


if __name__ == '__main__':
    main()

