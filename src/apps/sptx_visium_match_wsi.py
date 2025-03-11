# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

# SPTX_VISIUM_MATCH_WSI: given a Visium set of 4 regions and the corresponding
# whole slide image, match the regions to tissue parts in the slide.

from datetime import datetime
import hashlib
import simplejson as json
import configargparse as opt
from pathlib import Path

from compass.sptx.visium_v1 import get_affine_transformation_per_region
from compass.core import NumpyJSONEncoder, WSI

_time = datetime.now()
__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "0.1"
__description__ = {
    'name': 'st_match_tissue_parts',
    'unique_id' : hashlib.md5(str.encode("sptx_visium_match_wsi" + __version__)).hexdigest(),
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': ['None'],
    'output': ['None'],
    'params': dict()
}


def main() -> int:
    p = opt.ArgumentParser(description="Match tissue parts to high resolution images.")
    p.add_argument("--wsi", action="store", help="name of the whole slide image file",
                   required=True)
    p.add_argument("--parts", action="store", help="names of the high resolution image files (from Visium)",
                   nargs=4, type=str, required=True)
    p.add_argument("--out", action="store", help="JSON file for storing the results",
                   required=True)

    args = p.parse_args()

    __description__['params'] = vars(args)
    __description__['input'] = [str(args.wsi)]
    __description__['output'] = [str(args.out)]

    res = get_affine_transformation_per_region(
        WSI(args.wsi),
        args.parts,
        min_matching_quality=0.25
    )

    if Path(args.out).suffix.lower() != ".json":
        fout = args.out + ".json"
    else:
        fout = args.out

    with open(fout, "w") as f:
        res["__description__"] = __description__
        json.dump(res, f, cls=NumpyJSONEncoder)

    return 0

if __name__ == "__main__":
    main()
