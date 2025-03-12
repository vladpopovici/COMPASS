# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

# SAMPLE.PY: Sample manager trying to orchestrate all files related to a single sample.
#
# General structure of the folder holding the data for a sample:
#
#  <SAMPLE>.CP
#       |---- pyramid_0.zarr
#                |---- 0
#                |...
#       |---- pyramid_1.zarr
#       .....
#       |---- annot_0.bin
#       |---- annot_1.bin
#       .....
#       |---- annot.idx    <- JSON file with info on annotation files
#

from pathlib import Path
from typing import Union
import simplejson as json
from shutil import rmtree

class SampleManager(object):
    def __init__(self, sample_path: Union[str, Path], sample_name: str,
                 mode: str = 'r', overwrite_if_exists: bool = False) -> None:
        """
        SampleManager facilitates working with the files and folders associated with
        a single sample. Data associated with a sample is stored in a folder
        <sample_path>/<sample_name>.cp (.cp stands for "computational pathology"
        or "compass project")

        Args:
            sample_path: parent folder of the sample-associated data files
            sample_name: name of the sample
            mode: r/rw/w opening mode: if w, all previous files are discarded
            overwrite_if_exists: safety flag: if False, refuses to overwrite existing files
        """
        self.sample_path = Path(sample_path).expanduser().absolute()
        self.sample_name = sample_name
        self.full_path = (self.sample_path / self.sample_name).with_suffix('.cp')
        self.ann_list = None

        mode = mode.lower()
        if mode not in ['r', 'w', 'rw']:
            raise RuntimeError('unknown mode specified')
        if mode == 'r' or mode == 'rw':
            if not self.full_path.exists():
                raise RuntimeError('sample path does not exist')
            if (self.full_path / 'annot.idx').exists():
                with open(str(self.full_path / 'annot.idx'), 'r') as f:
                    self.ann_list = json.load(f)
        else:
            # mode == 'w':
            if self.full_path.exists():
                if self.full_path == Path('/'):
                    # refuse to delete it
                    raise RuntimeError(f'cannot remove/overwrite {self.full_path}')
                if overwrite_if_exists:
                    rmtree(self.full_path, ignore_errors=True)
                else:
                    raise RuntimeError(f'cannot overwrite {self.full_path}')
            self.full_path.mkdir(parents=True, exist_ok=True)

    def get_pyramid_path(self, pyramid_idx: int = 0) -> Path:
        return self.full_path / f'pyramid_{pyramid_idx}.zarr'

    def get_annotation_path(self, annot_idx: int = 0) -> Path:
        return self.full_path / f'annot_{annot_idx}.bin'

    def register_annotation(self, annot_idx: int, pyramid_idx: int, descr: str = "") -> None:
        if self.ann_list is None:
            self.ann_list = {}
        self.ann_list[annot_idx] = {'pyramid_idx': pyramid_idx, 'description': descr}
        with open(str(self.full_path / 'annot.idx'), 'w') as f:
            json.dump(self.ann_list, f)

    def get_new_annotation_idx(self) -> int:
        if len(self.ann_list) == 0:
            return 0
        idx = [k for k in self.ann_list]
        idx.sort()
        return idx[-1] + 1

    def get_registered_annotation(self, pyramid_idx: int) -> Union[list,None]:
        if self.ann_list is None:
            return None
        idx = [i for i in self.ann_list if self.ann_list[i]['pyramid_idx'] == pyramid_idx]
        idx.sort()

        return idx