# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""WSI -> Zarr ingestion: bake a vendor slide file into the canonical
Zarr v3 (sharded) pyramid read by :class:`compass.core.ZarrRaster`.

This is the Zarr retargeting of the legacy ``wsi2hdf5``
(``compass.legacy.core``): same pyramid-generation algorithm — crop to ROI,
iterative downsample via pyvips through temporary on-disk .v images,
shard-by-shard copy into the destination arrays — with the write step
producing the layout of :func:`compass.core.zarr_raster.write_pyramid`
instead of HDF5 groups.

OME-TIFF ingestion (:func:`ometiff2zarr`) reads through tifffile instead
(multi-channel, uint16, OME-XML calibration) and builds the lower levels
directly from the Zarr arrays already written, without temporary files.

pyvips and tifffile are imported lazily (inside the functions that need
them) so the pure planning helpers remain importable without them.
"""

import logging
import tempfile
from math import floor
from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
import zarr

from ..core.zarr_raster import create_level_array, write_pyramid_attrs

logger = logging.getLogger(__name__)


def resolve_crop(width: int, height: int,
                 roi: dict,
                 crop: tuple[int, int, int, int] | bool) -> tuple[int, int, int, int]:
    """Resolve the crop specification of :func:`wsi2zarr` to a concrete
    (x0, y0, width, height) region at level 0.

    Args:
        width, height: full extent of the slide at level 0.
        roi: the scanner's scan-region metadata ({'x0','y0','width','height'}
            or empty) as reported by :class:`compass.connector.wsi.WSI`.
        crop: False for the whole image; True to auto-crop to ``roi`` (if
            present); or an explicit (x0, y0, width, height), clamped to the
            image extent.
    """
    x0, y0, w, h = 0, 0, width, height
    if isinstance(crop, bool):
        if crop and len(roi) > 0:
            x0 = roi.get('x0', 0)
            y0 = roi.get('y0', 0)
            w = roi.get('width', width)
            h = roi.get('height', height)
    else:
        x0, y0, w, h = crop
        x0 = max(0, min(x0, width - 1))   # >= 0, < width
        y0 = max(0, min(y0, height - 1))  # >= 0, < height
        w = min(w, width - x0)
        h = min(h, height - y0)
    return x0, y0, w, h


def plan_levels(width: int, height: int,
                x0: int = 0, y0: int = 0,
                downscale_factor: int = 2,
                min_size: int = 256) -> list[dict]:
    """Plan the pyramid: per-level extents (and the level-0 crop origin
    scaled down alongside), largest first, stopping before either dimension
    would drop below ``min_size``.

    Returns a list of dicts with keys ``w``, ``h``, ``x0``, ``y0``.
    """
    levels = [{'w': width, 'h': height, 'x0': x0, 'y0': y0}]
    while True:
        w = int(floor(levels[-1]['w'] / downscale_factor))
        h = int(floor(levels[-1]['h'] / downscale_factor))
        cx0 = int(floor(levels[-1]['x0'] / downscale_factor))
        cy0 = int(floor(levels[-1]['y0'] / downscale_factor))
        if w < min_size or h < min_size:
            break
        levels.append({'w': w, 'h': h, 'x0': cx0, 'y0': cy0})
    return levels


def _fill_by_shards(arr: zarr.Array,
                    read_block: Callable[[int, int, int, int], np.ndarray]) -> None:
    """Fill ``arr`` one whole shard per write: ``read_block(x, y, w, h)``
    returns the (h, w[, c]) source block for that shard. No
    read-modify-write of partial shards, and memory bounded by the shard
    (not the slide) size."""
    ch, cw = arr.shape[0:2]
    shy, shx = arr.shards[0:2]
    for j in range(0, ch, shy):
        bh = min(shy, ch - j)
        for i in range(0, cw, shx):
            bw = min(shx, cw - i)
            arr[j: j + bh, i: i + bw] = read_block(i, j, bw, bh)


def box_downsample(block: np.ndarray, factor: int) -> np.ndarray:
    """Downsample (h, w[, c]) by an integer ``factor`` with box averaging
    (mean over ``factor x factor`` blocks), keeping the dtype (integers are
    rounded). Trailing rows/columns that do not fill a whole box are
    dropped, consistent with :func:`plan_levels`' floor rounding."""
    h, w = block.shape[0] // factor, block.shape[1] // factor
    b = block[: h * factor, : w * factor]
    b = b.reshape((h, factor, w, factor) + block.shape[2:])
    acc = np.float64 if block.dtype.itemsize > 2 else np.float32
    out = b.mean(axis=(1, 3), dtype=acc)
    if np.issubdtype(block.dtype, np.integer):
        out = np.rint(out)
    return out.astype(block.dtype)


##-
def wsi2zarr(
        wsi_path: str | Path | PathLike,
        dst_path: str | Path | PathLike,
        crop: tuple[int, int, int, int] | bool = False,
        downscale_factor: int = 2,
        min_size: int = 256,
        chunk_size: int = 512,
        shard_factor: int = 8,
        tmp_dir: str | Path | PathLike | None = None,
        overwrite: bool = False,
) -> None:
    """Convert a WSI file to the canonical pyramidal Zarr v3 (sharded)
    format. Highest resolution is level ``"0"``.

    The pyramid is rebuilt rather than copied from the WSI: level 0 is the
    (cropped) base image; every further level is generated by downscaling
    the previous one with pyvips. This gives control over the pyramid
    structure independently of what the scanner stored. Temporary .v images
    (uncompressed, together several times the size of level 0) keep RAM
    pressure low; data moves into Zarr one shard at a time, so peak memory
    of the copy step is a single shard regardless of slide size.

    :param wsi_path: source file path.
    :param dst_path: destination Zarr group (directory).
    :param crop: either bool to control auto-crop to the scan ROI, or
        (x0, y0, width, height) for an explicit crop region.
    :param downscale_factor: downsampling factor between pyramid levels.
    :param min_size: minimum size (either width or height) of the last level.
    :param chunk_size: chunk edge (pixels); match the viewer's tile size.
    :param shard_factor: shard edge, in chunks.
    :param tmp_dir: directory for the temporary .v images (default: next to
        ``dst_path``). Point it at a fast local disk when ``dst_path`` is on
        slow storage (USB/network drive): the intermediates are written and
        re-read several times, and a slow disk backs up the OS write cache
        until RAM runs out. Avoid a RAM-backed ``/tmp`` (tmpfs) — the
        intermediates are slide-sized.
    :param overwrite: overwrite an existing store at ``dst_path``.
    """
    # Order matters: openslide (wheel-bundled libopenslide.so.1, 4.x) must be
    # loaded before pyvips. libvips' openslide module drags in the system
    # libopenslide.so.0 (3.x), whose global openslide_* symbols then shadow
    # the 4.x ones and OpenSlide() fails in read_icc_profile (size -1).
    from .wsi import WSI

    import pyvips

    wsi_path = Path(wsi_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = dst_path.parent if tmp_dir is None else Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    wsi = WSI(wsi_path)
    x0, y0, width, height = resolve_crop(
        wsi.info["width"], wsi.info["height"], wsi.info["roi"], crop)
    levels = plan_levels(width, height, x0, y0, downscale_factor, min_size)
    nlevels = len(levels)

    logger.info("wsi2zarr: %d levels to generate (level 0 is just a copy)", nlevels)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_path:
        tmp_path = Path(tmp_path)

        logger.info("wsi2zarr: copy base level, crop (%d, %d, %d, %d)",
                    levels[0]['x0'], levels[0]['y0'], levels[0]['w'], levels[0]['h'])
        im = pyvips.Image.new_from_file(str(wsi_path), level=0, autocrop=False)
        im = im.crop(levels[0]['x0'], levels[0]['y0'], levels[0]['w'], levels[0]['h'])
        im = im.flatten(background=wsi.info['background'])
        im.write_to_file(str(tmp_path / "level0.v"))

        for level in range(1, nlevels):
            logger.info("wsi2zarr: generating level %d...", level)
            im = pyvips.Image.new_from_file(str(tmp_path / f"level{level - 1}.v"))
            im_scaled = im.resize(1.0 / downscale_factor, gap=2.0)
            im_scaled.write_to_file(str(tmp_path / f"level{level}.v"))

        logger.info("wsi2zarr: from temporary to destination")
        root = zarr.create_group(str(dst_path), overwrite=overwrite)

        for level in range(nlevels):
            logger.info("wsi2zarr: copying level %d...", level)
            im = pyvips.Image.new_from_file(str(tmp_path / f"level{level}.v"))
            arr = create_level_array(root, level, (im.height, im.width, 3), "uint8",
                                     chunk_size=chunk_size, shard_factor=shard_factor)
            _fill_by_shards(arr, lambda x, y, w, h: im.crop(x, y, w, h).numpy())

        write_pyramid_attrs(root, nlevels,
                            mpp_x=wsi.info['mpp_x'], mpp_y=wsi.info['mpp_y'],
                            objective_power=wsi.info['objective_power'],
                            mag_step=downscale_factor,
                            channel_names=["R", "G", "B"])

    logger.info("wsi2zarr: done")
##


##-
def ometiff2zarr(
        src_path: str | Path | PathLike,
        dst_path: str | Path | PathLike,
        series: int | None = None,
        z: int | None = None,
        t: int | None = None,
        crop: tuple[int, int, int, int] | bool = False,
        mpp: float | tuple[float, float] | None = None,
        objective_power: float | None = None,
        channel_names: list[str] | None = None,
        downscale_factor: int = 2,
        min_size: int = 256,
        chunk_size: int = 512,
        shard_factor: int = 8,
        overwrite: bool = False,
) -> None:
    """Convert an OME-TIFF file to the canonical pyramidal Zarr v3 (sharded)
    format. Highest resolution is level ``"0"``.

    As in :func:`wsi2zarr`, the pyramid is rebuilt rather than copied: level
    0 is the (cropped) full-resolution plane of the chosen series, copied
    one shard at a time through tifffile's lazy Zarr view of the TIFF tiles;
    every further level is box-averaged from the previous Zarr level, shard
    by shard (an output shard maps onto a ``downscale_factor``-times larger,
    shard-aligned block of the level above). No temporary files; peak
    memory ~ ``downscale_factor**2`` shards.

    Any axis layout tifffile reports (interleaved ``YXS`` RGB, planar
    ``CYX`` multi-channel, ``TCZYX``, ...) is normalized to YXC; the dtype
    (uint8, uint16, ...) is preserved. Single-channel images are stored as
    (h, w, 1).

    :param src_path: source OME-TIFF file.
    :param dst_path: destination Zarr group (directory).
    :param series: index of the image series to convert (default: the one
        with the largest full-resolution plane — skips label/macro images).
    :param z, t: plane to take along Z/T. Needed only when that axis has
        more than one plane — a stack is never reduced silently.
    :param crop: False for the whole image, or (x0, y0, width, height) at
        full resolution (clamped to the image). OME-TIFF has no scan ROI,
        so True behaves as False.
    :param mpp: microns-per-pixel at full resolution, either one value or
        (mpp_x, mpp_y). Overrides ``PhysicalSizeX/Y``; required when the
        file carries no calibration.
    :param objective_power: overrides the OME ``NominalMagnification``;
        required when the file has none.
    :param channel_names: overrides the OME channel names (default when
        neither: R,G,B for 3-sample RGB, C0..Cn otherwise).
    :param downscale_factor: integer downsampling factor between levels.
    :param min_size: minimum size (either width or height) of the last level.
    :param chunk_size: chunk edge (pixels); match the viewer's tile size.
    :param shard_factor: shard edge, in chunks.
    :param overwrite: overwrite an existing store at ``dst_path``.
    """
    import tifffile

    from .ometiff import read_ome_metadata, yxc_selection

    if int(downscale_factor) != downscale_factor or downscale_factor < 2:
        raise ValueError("downscale_factor must be an integer >= 2")
    downscale_factor = int(downscale_factor)

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(src_path) as tif:
        if series is None:
            series = max(range(len(tif.series)),
                         key=lambda k: int(np.prod(tif.series[k].shape)))
        sr = tif.series[series]
        meta = (read_ome_metadata(tif.ome_metadata, series) if tif.is_ome
                else {"mpp_x": None, "mpp_y": None,
                      "objective_power": None, "channel_names": []})

        # --- calibration: explicit arguments win, then OME-XML, else fail
        if mpp is not None:
            mpp_x, mpp_y = (mpp, mpp) if np.isscalar(mpp) else mpp
        else:
            mpp_x, mpp_y = meta["mpp_x"], meta["mpp_y"]
            if mpp_x is None and mpp_y is None:
                raise ValueError(f"{src_path}: no PhysicalSizeX/Y; pass mpp=")
            mpp_x = mpp_y if mpp_x is None else mpp_x
            mpp_y = mpp_x if mpp_y is None else mpp_y
        if objective_power is None:
            objective_power = meta["objective_power"]
            if objective_power is None:
                raise ValueError(f"{src_path}: no NominalMagnification; pass objective_power=")

        # --- axes -> YXC
        index, perm, nchannels = yxc_selection(
            sr.axes, sr.shape, {ax: v for ax, v in (("Z", z), ("T", t)) if v is not None})
        iy, ix = sr.axes.index("Y"), sr.axes.index("X")
        height, width = sr.shape[iy], sr.shape[ix]

        if channel_names is None:
            names = meta["channel_names"]
            if len(names) == nchannels and all(names):
                channel_names = list(names)
            elif nchannels == 3 and "S" in sr.axes:
                channel_names = ["R", "G", "B"]
            else:
                channel_names = [f"C{k}" for k in range(nchannels)]
        if len(channel_names) != nchannels:
            raise ValueError(f"{len(channel_names)} channel names for {nchannels} channels")

        x0, y0, width, height = resolve_crop(width, height, {}, crop)
        levels = plan_levels(width, height, x0, y0, downscale_factor, min_size)
        nlevels = len(levels)
        logger.info("ometiff2zarr: series %d (%s %s, %s), %d levels to generate",
                    series, sr.axes, sr.shape, sr.dtype, nlevels)

        src = zarr.open(sr.aszarr(level=0), mode="r")

        def read_base(x, y, w, h):
            idx = list(index)
            idx[iy] = slice(y0 + y, y0 + y + h)
            idx[ix] = slice(x0 + x, x0 + x + w)
            block = np.transpose(src[tuple(idx)], perm)
            return block if block.ndim == 3 else block[..., np.newaxis]

        root = zarr.create_group(str(dst_path), overwrite=overwrite)

        logger.info("ometiff2zarr: copy base level, crop (%d, %d, %d, %d)",
                    x0, y0, width, height)
        prev = create_level_array(root, 0, (height, width, nchannels), sr.dtype,
                                  chunk_size=chunk_size, shard_factor=shard_factor)
        _fill_by_shards(prev, read_base)

    f = downscale_factor
    for level in range(1, nlevels):
        logger.info("ometiff2zarr: generating level %d...", level)
        arr = create_level_array(root, level,
                                 (levels[level]['h'], levels[level]['w'], nchannels),
                                 prev.dtype,
                                 chunk_size=chunk_size, shard_factor=shard_factor)
        _fill_by_shards(arr, lambda x, y, w, h, src=prev:
                        box_downsample(src[f * y: f * (y + h), f * x: f * (x + w)], f))
        prev = arr

    write_pyramid_attrs(root, nlevels,
                        mpp_x=mpp_x, mpp_y=mpp_y,
                        objective_power=objective_power,
                        mag_step=downscale_factor,
                        channel_names=channel_names)

    logger.info("ometiff2zarr: done")
##
