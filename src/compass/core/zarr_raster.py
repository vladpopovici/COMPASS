# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from os import PathLike
from pathlib import Path
from typing import Sequence

import dask.array as da
import numpy as np
import zarr
from zarr.codecs import BloscCodec

from .magnif import Magnification
from .pyramid import PyramidalRaster
from .types import ImageShape

FORMAT_VERSION = 2  # version 1 was the HDF5 layout (legacy compass.legacy._mri)


#####
class ZarrRaster(PyramidalRaster):
    """MultiResolution raster backed by a Zarr v3 (sharded) store.

    Storage layout (written by :func:`write_pyramid`):
      - one Zarr group per image, one array per pyramid level, named
        ``"0"``, ``"1"``, ... (level 0 = highest resolution);
      - arrays are (height, width[, channels]), chunked at the viewer's
        tile-fetch granularity and gathered into shards;
      - group attrs: ``compass_version``, ``level_count``, ``base_mpp``,
        ``base_mag_step``, ``base_objective_power``, ``channel_names``,
        ``dimension_names``;
      - per-level attrs: ``mpp_x``, ``mpp_y``, ``objective_power``,
        ``scale_factor``.

    Args:
        path: path to the Zarr group (directory).

    Note: `as_type` arguments are accepted for interface compatibility but,
    as in the legacy HDF5 implementation, pixels are returned in their
    stored dtype.
    """

    def __init__(self, path: str | Path | PathLike):
        self._path = Path(path)
        if not self._path.exists():
            raise ValueError(f"no such Zarr store: {path}")

        self.__storage = zarr.open_group(str(self._path), mode="r")
        self._info = dict(self.__storage.attrs)

        nlevels = int(self._info["level_count"])
        self._levels = [self.__storage[str(lv)] for lv in range(nlevels)]
        base_height, base_width = self._levels[0].shape[0:2]

        super().__init__(ImageShape(width=base_width, height=base_height),
                         Magnification(float(self._info["base_objective_power"]),
                                       float(self._info["base_mpp"]),
                                       level=0,
                                       n_levels=nlevels,
                                       magnif_step=float(self._info["base_mag_step"])))

        # Due to rounding errors the actual level images stored may be 1 pixel
        # off from the computed shapes. Update their dims from actual arrays:
        for lv in range(self.nlevels):
            self._pyramid_levels[1, lv] = self._levels[lv].shape[0]  # height
            self._pyramid_levels[0, lv] = self._levels[lv].shape[1]  # width

    @property
    def path(self) -> Path:
        return self._path

    @property
    def storage(self) -> zarr.Group:
        return self.__storage

    @property
    def info(self) -> dict:
        return self._info

    @property
    def channel_names(self) -> list[str]:
        return list(self._info.get("channel_names", []))

    @property
    def nchannels(self) -> int:
        shape = self._levels[0].shape
        return int(shape[2]) if len(shape) > 2 else 1

    def level_info(self, level: int) -> dict:
        """Per-level attributes (mpp_x, mpp_y, objective_power, scale_factor)."""
        self._check_level(level)
        return dict(self._levels[level].attrs)

    def _check_level(self, level: int) -> None:
        if level < 0 or level >= self.nlevels:
            raise RuntimeError("requested level does not exist")

    def _check_bounds(self, x0: int, y0: int, width: int, height: int, level: int) -> None:
        if x0 < 0 or y0 < 0 or width < 0 or height < 0 or \
                x0 >= self.widths[level] or y0 >= self.heights[level] or \
                x0 + width > self.widths[level] or \
                y0 + height > self.heights[level]:
            raise RuntimeError("region out of layer's extent")

    def get_region_px(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a region from the image source. The region is specified in
        pixel coordinates.

        Args:
            x0, y0 (long): top left corner of the region (in pixels, at the
                specified level)
            width, height (long): width and height (in pixels) of the region.
            level (int): the magnification level to read from
            as_type: unused; pixels are returned in their stored dtype

        Returns:
            a numpy.ndarray
        """
        self._check_level(level)
        self._check_bounds(x0, y0, width, height, level)

        return self._levels[level][y0:y0 + height, x0:x0 + width, ...]

    def get_region_px_dask(self, x0: int, y0: int,
                           width: int, height: int,
                           level: int = 0, as_type=np.uint8) -> da.Array:
        """Like :meth:`get_region_px`, but returns a (lazy) dask array chunked
        at the Zarr chunk granularity."""
        self._check_level(level)
        self._check_bounds(x0, y0, width, height, level)

        arr = self._levels[level]
        img = da.from_array(arr, chunks=arr.chunks)[y0:y0 + height, x0:x0 + width, ...]

        return img

    def get_plane(self, level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a whole plane from the image pyramid and return it as a numpy
        array.

        Args:
            level (int): pyramid level to read
            as_type: unused; pixels are returned in their stored dtype

        Returns:
            a numpy.ndarray
        """
        self._check_level(level)

        return self._levels[level][:]

    def get_plane_dask(self, level: int = 0, as_type=np.uint8) -> da.Array:
        """Like :meth:`get_plane`, but returns a (lazy) dask array chunked at
        the Zarr chunk granularity."""
        self._check_level(level)

        arr = self._levels[level]
        return da.from_array(arr, chunks=arr.chunks)
##


def write_pyramid(path: str | Path | PathLike,
                  levels: Sequence[np.ndarray],
                  *,
                  base_mpp: float,
                  base_objective_power: float,
                  mag_step: float = 2.0,
                  channel_names: Sequence[str] | None = None,
                  chunk_size: int = 512,
                  shard_factor: int = 8,
                  overwrite: bool = False) -> None:
    """Write a pyramidal raster as a Zarr v3 (sharded) group, in the layout
    read by :class:`ZarrRaster`.

    This writer takes fully materialized per-level arrays; it is meant for
    maps, test data and small images. Ingestion of whole slides (streaming
    via pyvips) belongs to ``compass.connector``, which targets this same
    layout.

    Args:
        path: destination directory of the Zarr group.
        levels: per-level arrays, level 0 first (highest resolution), each
            (height, width[, channels]); all levels must share dtype and
            number of channels.
        base_mpp: microns-per-pixel at level 0.
        base_objective_power: objective power at level 0 (e.g. 20.0, 40.0).
        mag_step: scale factor between consecutive levels.
        channel_names: names of the channels; defaults to ("R", "G", "B")
            for 3-channel images, ("I",) for single-channel ones.
        chunk_size: chunk edge (pixels); should match the viewer's
            tile-fetch granularity.
        shard_factor: shard edge, in chunks (shards of
            ``shard_factor x shard_factor`` chunks).
        overwrite: overwrite an existing store at ``path``.
    """
    if len(levels) == 0:
        raise ValueError("at least one pyramid level is required")

    ndim = levels[0].ndim
    dtype = levels[0].dtype
    nchannels = levels[0].shape[2] if ndim == 3 else 1
    for lv, img in enumerate(levels):
        if img.ndim != ndim or img.dtype != dtype:
            raise ValueError(f"level {lv}: inconsistent shape/dtype across levels")
        if img.ndim == 3 and img.shape[2] != nchannels:
            raise ValueError(f"level {lv}: inconsistent channel count")

    if channel_names is None:
        channel_names = ["R", "G", "B"] if nchannels == 3 else ["I"] * nchannels
    if len(channel_names) != nchannels:
        raise ValueError("channel_names length does not match channel count")

    root = zarr.create_group(str(path), overwrite=overwrite)
    codec = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    for lv, img in enumerate(levels):
        h, w = img.shape[0:2]
        chunks = (min(chunk_size, h), min(chunk_size, w)) + ((nchannels,) if ndim == 3 else ())
        shards = (chunks[0] * shard_factor, chunks[1] * shard_factor) + ((nchannels,) if ndim == 3 else ())
        arr = root.create_array(name=str(lv),
                                shape=img.shape,
                                dtype=dtype,
                                chunks=chunks,
                                shards=shards,
                                compressors=codec)
        arr[...] = img

        sf = mag_step ** lv
        arr.attrs["mpp_x"] = base_mpp * sf
        arr.attrs["mpp_y"] = base_mpp * sf
        arr.attrs["objective_power"] = base_objective_power / sf
        arr.attrs["scale_factor"] = sf

    root.attrs["compass_version"] = FORMAT_VERSION
    root.attrs["level_count"] = len(levels)
    root.attrs["base_mpp"] = float(base_mpp)
    root.attrs["base_mag_step"] = float(mag_step)
    root.attrs["base_objective_power"] = float(base_objective_power)
    root.attrs["channel_names"] = list(channel_names)
    root.attrs["dimension_names"] = ["y", "x", "c"] if ndim == 3 else ["y", "x"]
##
