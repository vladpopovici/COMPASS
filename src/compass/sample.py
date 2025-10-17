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
#       |---- pyramid_0.h5
#       |---- pyramid_1.h5
#       .....
#       |---- annot_0.json.gz      <- first annotation (compressed)
#       |---- annot_1.json.gz
#         .....
#       |---- annot_idx.json    <- JSON file with info on annotation files
#

from pathlib import Path
from typing import Union
import orjson as json
from shutil import rmtree
import gzip
from wsitk_annot import Annotation
from functools import cached_property

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
        if mode not in ['r', 'w', 'a']: # read/write/append
            raise RuntimeError('unknown mode specified')
        if mode == 'r' or mode == 'a':
            if not self.full_path.exists() or not self.get_annot_registry_path.exists():
                raise RuntimeError('sample or annotation path does not exist')
            self._load_annot_registry()
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
            self._init_annotations()

    @cached_property
    def get_annot_registry_path(self) -> Path:
        return self.full_path / 'annot_idx.json'

    def _load_annot_registry(self):
        if not self.get_annot_registry_path.exists():
            self.ann_list = None
            return
        with open(self.get_annot_registry_path, 'rb') as f:
            self.ann_list = json.loads(f.read())

    def _init_annotations(self):
        if not self.get_annot_registry_path.exists():
            with open(self.get_annot_registry_path, 'wb') as f:
                f.write(
                    json.dumps(self.ann_list, option=json.OPT_NON_STR_KEYS)
                )

    def get_pyramid_path(self, pyramid_idx: int = 0) -> Path:
        return self.full_path / f'pyramid_{pyramid_idx}.h5'

    def get_annotation_path(self, ann_idx: int) -> Path:
        return self.full_path / f'annot_{ann_idx}.json.gz'

    def add_annotation(self, annot: Annotation, pyramid_idx: int, descr: str = "") -> None:
        if self.ann_list is None:
            self.ann_list = {}

        annot_idx = self._get_new_annotation_idx()
        self.ann_list[annot_idx] = {'pyramid_idx': pyramid_idx, 'description': descr}

        with gzip.open(self.get_annotation_path(annot_idx), 'wb') as zf:
                annot.save(zf)

        with open(self.get_annot_registry_path, 'wb') as f:
            f.write(
                json.dumps(self.ann_list, option=json.OPT_NON_STR_KEYS)
            )

    def update_annotation(self, annot:Annotation, ann_idx:int) -> int:
        if self.ann_list is None or len(self.ann_list) == 0:
            raise RuntimeError("no annotations in the registry")
        if ann_idx not in self.ann_list:
            raise RuntimeError(f"annotation with index {ann_idx} not found")

        with gzip.open(self.get_annotation_path(ann_idx), 'wb') as f:
            annot.save(f)

    def _get_new_annotation_idx(self) -> int:
        if len(self.ann_list) == 0:
            return 0
        idx = [k for k in self.ann_list]
        idx.sort()
        return idx[-1] + 1

    def get_annotations_for_pyramid(self, pyramid_idx: int) -> Union[list,None]:
        if self.ann_list is None:
            return None
        idx = [i for i in self.ann_list if self.ann_list[i]['pyramid_idx'] == pyramid_idx]
        idx.sort()

        return idx
