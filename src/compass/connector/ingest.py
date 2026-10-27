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
from copy import deepcopy
from datetime import datetime
from math import floor
from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
import zarr

from ..core.zarr_raster import FORMAT_VERSION, create_level_array, write_pyramid_attrs

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


#: controlled vocabulary for ``import_record(...)["transform"]["type"]``
TRANSFORM_TYPES = ("shift", "rotate", "scale", "affine")
#: version of the import record layout (bump on incompatible changes)
IMPORT_RECORD_VERSION = 1
#: OME-TIFF start-level reads: source pixels per side of one sub-read
_BASE_READ_EDGE = 2048


def plan_import(width: int, height: int,
                crop: tuple[int, int, int, int],
                start_level: int = 0,
                downscale_factor: int = 2,
                min_size: int = 256) -> dict:
    """Plan an import that starts at level ``start_level`` of the pyramid
    :func:`plan_levels` would build for ``crop``.

    Level ``start_level`` of that plan becomes the new level 0, so the
    result has ``start_level`` fewer levels. The crop is snapped to the
    start level's pixel grid, which keeps the mapping from source to output
    coordinates exact: ``x_out = (x_src - crop_x0) / downsample``.

    Args:
        width, height: source extent at level 0 (for validation only).
        crop: (x0, y0, w, h) at source level 0, already clamped
            (see :func:`resolve_crop`).
        start_level: index into the planned pyramid.
        downscale_factor, min_size: as in :func:`plan_levels`.

    Returns:
        dict with ``start_level``, ``downsample`` (source level-0 pixels
        per output level-0 pixel), ``crop`` (effective x0, y0, w, h at
        source level 0), ``requested_crop``, ``levels`` (output pyramid,
        list of ``{'w', 'h'}``), ``planned`` (the full plan from source
        level 0, list of ``{'w', 'h'}``).

    Raises:
        ValueError: ``start_level`` outside the planned pyramid.
    """
    x0, y0, w, h = (int(v) for v in crop)
    if not (0 <= x0 and 0 <= y0 and x0 + w <= width and y0 + h <= height):
        raise ValueError(f"crop {crop} outside the {width} x {height} image")
    full = plan_levels(w, h, x0, y0, downscale_factor, min_size)
    if not 0 <= start_level < len(full):
        raise ValueError(f"start level {start_level} not in the planned pyramid "
                         f"(levels 0..{len(full) - 1})")
    d = int(downscale_factor) ** int(start_level)
    base = full[start_level]
    return {
        "start_level": int(start_level),
        "downsample": d,
        "crop": (base["x0"] * d, base["y0"] * d, base["w"] * d, base["h"] * d),
        "requested_crop": (x0, y0, w, h),
        "levels": [{"w": lv["w"], "h": lv["h"]} for lv in full[start_level:]],
        "planned": [{"w": lv["w"], "h": lv["h"]} for lv in full],
    }


def closest_level_for_mpp(mpp0: float, target_mpp: float,
                          nlevels: int, downscale_factor: int = 2) -> int:
    """Level of a planned pyramid (level k has ``mpp0 * factor**k``)
    closest to ``target_mpp``, compared by ratio; ties go to the finer
    level."""
    if mpp0 <= 0 or target_mpp <= 0:
        raise ValueError("mpp values must be positive")
    errs = [abs(np.log(mpp0 * downscale_factor ** k / target_mpp))
            for k in range(nlevels)]
    best = min(errs)
    return next(k for k, e in enumerate(errs) if e <= best + 1e-12)


def crop_scale_transform(x0: float, y0: float, downsample: float) -> dict:
    """Transform from source level-0 to output level-0 pixel coordinates
    for a crop at (x0, y0) followed by a ``downsample`` reduction:
    ``x_out = (x_src - x0) / downsample`` (same for y).

    ``type`` is the most specific of :data:`TRANSFORM_TYPES` that
    describes it (``shift``: dx, dy; ``scale``: sx, sy; ``affine``:
    matrix only). ``matrix`` (2 x 3, row-major, applied to (x, y, 1)) is
    always present, so consumers may ignore ``type``.
    """
    s = 1.0 / downsample
    matrix = [[s, 0.0, -x0 * s], [0.0, s, -y0 * s]]
    if downsample == 1:
        params = {"type": "shift", "dx": -float(x0), "dy": -float(y0)}
    elif x0 == 0 and y0 == 0:
        params = {"type": "scale", "sx": s, "sy": s}
    else:
        params = {"type": "affine"}
    return {**params, "from": "source_level0", "to": "output_level0",
            "matrix": matrix}


def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def import_record(*,
                  src_path: str | Path | PathLike,
                  src_format: str,
                  source: dict,
                  crop_mode: str,
                  plan: dict,
                  dst_path: str | Path | PathLike,
                  downscale_factor: int,
                  chunk_size: int,
                  shard_factor: int,
                  provenance: dict | None = None) -> dict:
    """Describe an import (JSON-serializable): source, crop at source level
    0, start level, output pyramid and the source -> output coordinate
    transform. Written to the store as the ``import`` group attribute.

    Args:
        source: source facts: ``width``, ``height``, ``mpp_x``, ``mpp_y``,
            ``objective_power`` (calibration may be None), plus any extras
            (e.g. OME ``series``, ``z``, ``t``).
        crop_mode: how the crop was chosen ("none", "roi", "explicit", or a
            caller's own, e.g. "detect").
        plan: result of :func:`plan_import`.
        provenance: merged (recursively) into the record last, e.g.
            ``{"tool": "wsi_import", "crop": {"mode": "detect"}}``.
    """
    d = plan["downsample"]
    x0, y0, w, h = plan["crop"]
    rx0, ry0, rw, rh = plan["requested_crop"]
    scale = lambda v, f: None if v is None else float(v) * f
    src_path = Path(src_path).resolve()
    record = {
        "record_version": IMPORT_RECORD_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {"path": str(src_path), "name": src_path.name,
                   "format": src_format, **source},
        "crop": {"mode": crop_mode, "x0": x0, "y0": y0, "width": w, "height": h,
                 "requested": {"x0": rx0, "y0": ry0, "width": rw, "height": rh}},
        "start": {"level": plan["start_level"], "downsample": d,
                  "mpp_x": scale(source.get("mpp_x"), d),
                  "mpp_y": scale(source.get("mpp_y"), d)},
        "output": {"path": str(Path(dst_path).resolve()),
                   "format_version": FORMAT_VERSION,
                   "levels": len(plan["levels"]),
                   "width": plan["levels"][0]["w"], "height": plan["levels"][0]["h"],
                   "mpp_x": scale(source.get("mpp_x"), d),
                   "mpp_y": scale(source.get("mpp_y"), d),
                   "objective_power": scale(source.get("objective_power"), 1.0 / d),
                   "downscale_factor": int(downscale_factor),
                   "chunk_size": int(chunk_size), "shard_factor": int(shard_factor),
                   "level_shapes": [[lv["w"], lv["h"]] for lv in plan["levels"]]},
        "transform": crop_scale_transform(x0, y0, d),
    }
    if provenance:
        _deep_update(record, deepcopy(provenance))
    return record


def _crop_mode(crop, roi: dict) -> str:
    if isinstance(crop, bool):
        return "roi" if crop and roi else "none"
    return "explicit"


def _pick_source_level(downsamples: list[float], d: int) -> tuple[int, int]:
    """Stored slide level to read for an output downsample ``d``: the
    coarsest one whose downsample divides ``d`` (to 0.1 %), and the integer
    reduction ``r`` still needed from it. Level 0 always qualifies."""
    best = (0, d)
    for k, ds in enumerate(downsamples):
        if ds <= 0 or ds > d * (1 + 1e-3):
            continue
        r = d / ds
        if abs(r - round(r)) < 1e-3 * r and ds >= downsamples[best[0]]:
            best = (k, int(round(r)))
    return best


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
        start_level: int = 0,
        provenance: dict | None = None,
) -> dict:
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
    :param start_level: start at this level of the planned pyramid (see
        :func:`plan_import`): it becomes level 0, calibration is scaled
        accordingly. The base image is read from the coarsest stored slide
        level that divides the needed downsample, so full resolution is
        not read when the scanner stored that level.
    :param provenance: extra fields merged into the import record.
    :return: the import record (:func:`import_record`), also stored as the
        group attribute ``"import"``.
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
    crop_xywh = resolve_crop(wsi.info["width"], wsi.info["height"],
                             wsi.info["roi"], crop)
    plan = plan_import(wsi.info["width"], wsi.info["height"], crop_xywh,
                       start_level, downscale_factor, min_size)
    nlevels = len(plan["levels"])
    d = plan["downsample"]
    ex0, ey0, ew, eh = plan["crop"]
    bw, bh = plan["levels"][0]["w"], plan["levels"][0]["h"]

    logger.info("wsi2zarr: %d levels to generate (level 0 is just a copy)", nlevels)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_path:
        tmp_path = Path(tmp_path)

        # base image: read the coarsest stored level that divides d, reduce
        # the rest (r) with pyvips
        src_level, r = _pick_source_level(
            wsi.info.get("level_downsamples", [1.0]), d)
        s = d // r  # stored level's downsample
        logger.info("wsi2zarr: base level from slide level %d (x%d), crop "
                    "(%d, %d, %d, %d) at level 0, reduce x%d",
                    src_level, s, ex0, ey0, ew, eh, r)
        im = pyvips.Image.new_from_file(str(wsi_path), level=src_level, autocrop=False)
        im = im.crop(ex0 // s, ey0 // s, bw * r, bh * r)
        im = im.flatten(background=wsi.info['background'])
        if r > 1:
            im = im.resize(1.0 / r, gap=2.0)
            if (im.width, im.height) != (bw, bh):  # resize rounding
                im = im.crop(0, 0, min(bw, im.width), min(bh, im.height))
        im.write_to_file(str(tmp_path / "level0.v"))

        for level in range(1, nlevels):
            logger.info("wsi2zarr: generating level %d...", level)
            im = pyvips.Image.new_from_file(str(tmp_path / f"level{level - 1}.v"))
            im_scaled = im.resize(1.0 / downscale_factor, gap=2.0)
            # pyvips rounds odd sizes to nearest, plan_levels() floors: drop
            # the extra trailing row/column so arrays match the import record
            pw, ph = plan["levels"][level]["w"], plan["levels"][level]["h"]
            if (im_scaled.width, im_scaled.height) != (pw, ph):
                im_scaled = im_scaled.crop(0, 0, min(pw, im_scaled.width),
                                           min(ph, im_scaled.height))
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
                            mpp_x=wsi.info['mpp_x'] * d, mpp_y=wsi.info['mpp_y'] * d,
                            objective_power=wsi.info['objective_power'] / d,
                            mag_step=downscale_factor,
                            channel_names=["R", "G", "B"])

        record = import_record(
            src_path=wsi_path, src_format="wsi",
            source={k: wsi.info[k] for k in
                    ("width", "height", "mpp_x", "mpp_y", "objective_power")},
            crop_mode=_crop_mode(crop, wsi.info["roi"]), plan=plan,
            dst_path=dst_path, downscale_factor=downscale_factor,
            chunk_size=chunk_size, shard_factor=shard_factor,
            provenance=provenance)
        root.attrs["import"] = record

    logger.info("wsi2zarr: done")
    return record
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
        start_level: int = 0,
        provenance: dict | None = None,
) -> dict:
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
    :param start_level: start at this level of the planned pyramid (see
        :func:`plan_import`); the base level is box-averaged from full
        resolution while copying (reads the whole crop at full resolution).
    :param provenance: extra fields merged into the import record.
    :return: the import record (:func:`import_record`), also stored as the
        group attribute ``"import"``.
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

        src_width, src_height = int(width), int(height)
        plan = plan_import(src_width, src_height,
                           resolve_crop(src_width, src_height, {}, crop),
                           start_level, downscale_factor, min_size)
        levels = plan["levels"]
        nlevels = len(levels)
        d = plan["downsample"]
        x0, y0, _, _ = plan["crop"]
        width, height = levels[0]["w"], levels[0]["h"]
        logger.info("ometiff2zarr: series %d (%s %s, %s), %d levels to generate",
                    series, sr.axes, sr.shape, sr.dtype, nlevels)

        src = zarr.open(sr.aszarr(level=0), mode="r")

        def read_src(x, y, w, h):
            """(h, w, c) block at source level 0."""
            idx = list(index)
            idx[iy] = slice(y, y + h)
            idx[ix] = slice(x, x + w)
            block = np.transpose(src[tuple(idx)], perm)
            return block if block.ndim == 3 else block[..., np.newaxis]

        def read_base(x, y, w, h):
            """(h, w, c) block of the output level 0 at (x, y)."""
            if d == 1:
                return read_src(x0 + x, y0 + y, w, h)
            # box-average d x d source pixels per output pixel, in sub-reads
            # of ~_BASE_READ_EDGE source pixels to bound memory
            out = np.empty((h, w, nchannels), dtype=sr.dtype)
            t = max(1, _BASE_READ_EDGE // d)
            for j in range(0, h, t):
                th = min(t, h - j)
                for i in range(0, w, t):
                    tw = min(t, w - i)
                    blk = read_src(x0 + (x + i) * d, y0 + (y + j) * d, tw * d, th * d)
                    out[j:j + th, i:i + tw] = box_downsample(blk, d)
            return out

        root = zarr.create_group(str(dst_path), overwrite=overwrite)

        logger.info("ometiff2zarr: base level, crop (%d, %d, %d, %d) at level 0, "
                    "reduce x%d", *plan["crop"], d)
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
                        mpp_x=mpp_x * d, mpp_y=mpp_y * d,
                        objective_power=objective_power / d,
                        mag_step=downscale_factor,
                        channel_names=channel_names)

    record = import_record(
        src_path=src_path, src_format="ometiff",
        source={"width": src_width, "height": src_height,
                "mpp_x": float(mpp_x), "mpp_y": float(mpp_y),
                "objective_power": float(objective_power),
                "series": int(series), "z": z, "t": t,
                "dtype": str(np.dtype(sr.dtype)), "channel_names": list(channel_names)},
        crop_mode=_crop_mode(crop, {}), plan=plan,
        dst_path=dst_path, downscale_factor=downscale_factor,
        chunk_size=chunk_size, shard_factor=shard_factor,
        provenance=provenance)
    root.attrs["import"] = record

    logger.info("ometiff2zarr: done")
    return record
##
