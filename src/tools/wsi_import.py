#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Bake a whole-slide image or OME-TIFF into the COMPASS pyramidal Zarr v3
store read by ``compass.core.ZarrRaster``.

Thin CLI over :func:`compass.connector.wsi2zarr` (vendor WSI formats, via
OpenSlide/pyvips) and :func:`compass.connector.ometiff2zarr` (OME-TIFF, via
tifffile). The input format is auto-detected; OME-TIFF is checked first
(some scanner formats are TIFF-based but lack OME-XML).

Usage
-----
Run from the repository root with the connector dependencies installed
(``uv sync --extra connector``; WSI also needs system libvips/OpenSlide):

    uv run python src/tools/wsi_import.py SLIDE [DST] [options]

Examples
--------
    # vendor slide -> slide.zarr, calibration from the scanner metadata
    uv run python src/tools/wsi_import.py slide.mrxs

    # explicit destination; crop to the scanner scan-region
    uv run python src/tools/wsi_import.py slide.mrxs out/slide.zarr --auto-crop

    # OME-TIFF without NominalMagnification in its OME-XML
    uv run python src/tools/wsi_import.py cytx.ome.tif --objective-power 40

    # inspect calibration and the pyramid that would be generated
    uv run python src/tools/wsi_import.py slide.mrxs --info
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import argparse
import logging
import sys
from pathlib import Path

from compass.connector import ometiff2zarr, plan_levels, wsi2zarr

logger = logging.getLogger("wsi_import")

OME_SUFFIXES = (".ome.tif", ".ome.tiff")

CONNECTOR_HINT = ("WSI conversion needs OpenSlide and libvips on the system "
                  "and `uv sync --extra connector`; see docs/architecture.md "
                  "section 3.")


def is_ome_tiff(path: Path) -> bool:
    """True if ``path`` is a TIFF carrying an OME-XML description.

    Extension-named OME files are accepted directly; other TIFFs (and
    TIFF-based scanner formats) are sniffed with tifffile. Non-TIFF formats
    (MRXS, NDPI, ...) fail the tifffile open and return False.
    """
    name = path.name.lower()
    if name.endswith(OME_SUFFIXES):
        return True
    if not (name.endswith(".tif") or name.endswith(".tiff")):
        return False
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            return bool(tif.is_ome)
    except Exception:
        return False


def default_dst(src: Path) -> Path:
    """``slide.mrxs`` -> ``slide.zarr``; ``slide.ome.tif`` -> ``slide.zarr``."""
    name = src.name
    lower = name.lower()
    for suffix in OME_SUFFIXES:
        if lower.endswith(suffix):
            name = name[:-len(suffix)]
            break
    else:
        name = src.stem
    return src.parent / f"{name}.zarr"


def _ovan(args: argparse.Namespace) -> tuple[int, int, int, int] | bool:
    """Resolve --auto-crop/--crop against the target format."""
    if args.crop is not None:
        return tuple(args.crop)
    if args.auto_crop:
        if args.format == "ometiff":
            logger.warning("--auto-crop ignored: OME-TIFF has no scan ROI")
            return False
        return True
    return False


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wsi_import.py",
        description="Convert a WSI or OME-TIFF to the COMPASS pyramidal "
                    "Zarr v3 store.",
        epilog="OME-TIFF / WSI selection is automatic; use --format to force.")
    parser.add_argument("src", type=Path,
                        help="source image (vendor WSI or OME-TIFF)")
    parser.add_argument("dst", type=Path, nargs="?", default=None,
                        help="destination Zarr group (default: SRC with a "
                             ".zarr suffix)")
    parser.add_argument("--format", choices=["auto", "wsi", "ometiff"],
                        default="auto",
                        help="input format (default: auto-detect)")
    parser.add_argument("--info", action="store_true",
                        help="print slide metadata and the pyramid plan; "
                             "do not convert")

    pyramid = parser.add_argument_group("pyramid")
    pyramid.add_argument("--downscale-factor", type=int, default=2,
                         help="integer factor between levels (default: 2)")
    pyramid.add_argument("--min-size", type=int, default=256,
                         help="stop the pyramid before a level drops below "
                              "this edge (default: 256)")
    pyramid.add_argument("--chunk-size", type=int, default=512,
                         help="Zarr chunk edge in pixels; match the viewer's "
                              "tile size (default: 512)")
    pyramid.add_argument("--shard-factor", type=int, default=8,
                         help="shard edge in chunks (default: 8)")

    crop = parser.add_mutually_exclusive_group()
    crop.add_argument("--auto-crop", action="store_true",
                      help="crop to the scanner scan-region ROI (WSI only)")
    crop.add_argument("--crop", type=int, nargs=4,
                      metavar=("X0", "Y0", "W", "H"),
                      help="explicit crop at full resolution")

    ome = parser.add_argument_group("OME-TIFF options")
    ome.add_argument("--series", type=int, default=None,
                     help="image series index (default: largest plane)")
    ome.add_argument("--z", type=int, default=None,
                     help="Z plane, required for multi-plane Z stacks")
    ome.add_argument("--t", type=int, default=None,
                     help="T plane, required for multi-plane T stacks")
    ome.add_argument("--mpp", type=float, nargs="+", metavar="MPP",
                     help="microns-per-pixel, one value or MPP_X MPP_Y; "
                          "required if the file has no PhysicalSizeX/Y")
    ome.add_argument("--objective-power", type=float, default=None,
                     help="objective power; required if the OME-XML has no "
                          "NominalMagnification")
    ome.add_argument("--channel-names", nargs="+", default=None,
                     metavar="NAME",
                     help="channel names, overriding the OME-XML ones")

    wsi = parser.add_argument_group("WSI options")
    wsi.add_argument("--tmp-dir", type=Path, default=None,
                     help="directory for the intermediate .v images "
                          "(default: next to DST; use a fast local disk)")

    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite an existing store at DST")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="progress logging to stderr")

    args = parser.parse_args(argv)
    if args.mpp is not None and len(args.mpp) not in (1, 2):
        parser.error("--mpp takes one value or two (MPP_X MPP_Y)")
    for flag in ("downscale_factor", "min_size", "chunk_size", "shard_factor"):
        if getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    if args.downscale_factor < 2:
        parser.error("--downscale-factor must be >= 2")
    return args


def _mpp_arg(args) -> float | tuple[float, float] | None:
    if args.mpp is None:
        return None
    return args.mpp[0] if len(args.mpp) == 1 else (args.mpp[0], args.mpp[1])


def _width_height(args) -> tuple[int, int]:
    """(width, height) at full resolution for the pyramid plan."""
    if args.format == "ometiff":
        import tifffile
        with tifffile.TiffFile(args.src) as tif:
            series = args.series
            if series is None:
                series = max(range(len(tif.series)),
                             key=lambda k: tif.series[k].size)
            sr = tif.series[series]
            return (int(sr.shape[sr.axes.index("X")]),
                    int(sr.shape[sr.axes.index("Y")]))
    try:
        from compass.connector.wsi import WSI
    except ImportError as e:
        raise SystemExit(f"wsi_import: {e}\n{CONNECTOR_HINT}") from e
    wsi = WSI(args.src)
    return int(wsi.info["width"]), int(wsi.info["height"])


def _print_info(args) -> None:
    width, height = _width_height(args)
    print(f"source: {args.src}  [{args.format}]")
    if args.format == "ometiff":
        import tifffile
        from compass.connector.ometiff import read_ome_metadata
        with tifffile.TiffFile(args.src) as tif:
            print(f"series: {len(tif.series)}")
            for k, sr in enumerate(tif.series):
                mark = "*" if args.series in (None, k) and sr is tif.series[
                    args.series if args.series is not None else k] else " "
                print(f"  {mark} [{k}] axes={sr.axes} shape={tuple(sr.shape)} "
                      f"dtype={sr.dtype}")
            if tif.is_ome:
                idx = 0
                if args.series is not None:
                    idx = args.series
                else:
                    idx = max(range(len(tif.series)),
                              key=lambda k: tif.series[k].size)
                print("calibration:",
                      read_ome_metadata(tif.ome_metadata, idx))
    else:
        try:
            from compass.connector.wsi import WSI
        except ImportError as e:
            raise SystemExit(f"wsi_import: {e}\n{CONNECTOR_HINT}") from e
        info = dict(WSI(args.src).info)
        info["channels"] = ["R", "G", "B"]
        print("metadata:", info)
    crop = _ovan(args)
    x0, y0, w, h = (0, 0, width, height) if crop is False else (
        crop if crop is not True else (0, 0, width, height))
    levels = plan_levels(w, h, x0, y0,
                         downscale_factor=args.downscale_factor,
                         min_size=args.min_size)
    print(f"pyramid plan ({len(levels)} levels):")
    for lv, plan in enumerate(levels):
        print(f"  [{lv}] {plan['w']} x {plan['h']}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="{asctime} {name} {levelname}: {message}",
                        style="{")

    if not args.src.is_file():
        print(f"wsi_import: no such file: {args.src}", file=sys.stderr)
        return 2
    if args.format == "auto":
        args.format = "ometiff" if is_ome_tiff(args.src) else "wsi"
    if args.dst is None:
        args.dst = default_dst(args.src)

    if args.info:
        _print_info(args)
        return 0

    logger.info("wsi_import: %s [%s] -> %s", args.src, args.format, args.dst)
    kwargs = dict(crop=_ovan(args),
                  downscale_factor=args.downscale_factor,
                  min_size=args.min_size,
                  chunk_size=args.chunk_size,
                  shard_factor=args.shard_factor,
                  overwrite=args.overwrite)
    try:
        if args.format == "ometiff":
            ometiff2zarr(args.src, args.dst,
                         series=args.series, z=args.z, t=args.t,
                         mpp=_mpp_arg(args),
                         objective_power=args.objective_power,
                         channel_names=args.channel_names,
                         **kwargs)
        else:
            wsi2zarr(args.src, args.dst, tmp_dir=args.tmp_dir, **kwargs)
    except ImportError as e:
        print(f"wsi_import: {e}\n{CONNECTOR_HINT}", file=sys.stderr)
        return 2
    print(f"wsi_import: wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
