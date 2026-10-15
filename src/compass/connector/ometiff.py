# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""OME-TIFF metadata extraction for :func:`compass.connector.ometiff2zarr`.

Reads calibration (mpp), objective power and channel names from the
OME-XML of one image, and maps a tifffile series' axes onto the canonical
YXC layout. Pure functions over the XML string / axes string, so they are
testable without any TIFF file.
"""

import xml.etree.ElementTree as ET

# OME length units -> microns
_UNIT_TO_UM = {
    "pm": 1e-6,
    "Å": 1e-4,
    "nm": 1e-3,
    "µm": 1.0,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
}

# axes that hold channels (tifffile: C = OME channel, S = samples, e.g. RGB)
_CHANNEL_AXES = "CS"


def _to_um(value: str | None, unit: str | None) -> float | None:
    if value is None:
        return None
    unit = "µm" if unit is None else unit
    if unit not in _UNIT_TO_UM:
        raise ValueError(f"unsupported OME length unit: {unit!r}")
    return float(value) * _UNIT_TO_UM[unit]


def read_ome_metadata(ome_xml: str, image_index: int = 0) -> dict:
    """Extract calibration and channel metadata of one OME ``Image``.

    Args:
        ome_xml: the OME-XML document (``tifffile.TiffFile.ome_metadata``).
        image_index: index of the ``Image`` element (tifffile builds one
            series per OME image, in document order).

    Returns:
        dict with keys ``mpp_x``, ``mpp_y`` (microns, or None if absent),
        ``objective_power`` (or None) and ``channel_names`` (list, possibly
        empty or with None entries for unnamed channels).
    """
    root = ET.fromstring(ome_xml)
    images = root.findall("{*}Image")
    if not 0 <= image_index < len(images):
        raise IndexError(f"OME-XML has {len(images)} image(s), no index {image_index}")
    image = images[image_index]

    pixels = image.find("{*}Pixels")
    if pixels is None:
        raise ValueError(f"OME Image {image_index} has no Pixels element")
    mpp_x = _to_um(pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeXUnit"))
    mpp_y = _to_um(pixels.get("PhysicalSizeY"), pixels.get("PhysicalSizeYUnit"))

    channel_names = [ch.get("Name") for ch in pixels.findall("{*}Channel")]

    return {
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "objective_power": _objective_power(root, image),
        "channel_names": channel_names,
    }


def _objective_power(root: ET.Element, image: ET.Element) -> float | None:
    """NominalMagnification of the image's objective: the one referenced by
    ``ObjectiveSettings``; failing that, the only objective of the
    referenced instrument (as written by :func:`raster2tiff`'s minimal
    OME-XML); failing that, the only objective in the document."""
    objectives = root.findall("{*}Instrument/{*}Objective")
    settings = image.find("{*}ObjectiveSettings")
    candidates = []
    if settings is not None:
        candidates = [o for o in objectives if o.get("ID") == settings.get("ID")]
    if not candidates:
        inst_ref = image.find("{*}InstrumentRef")
        if inst_ref is not None:
            candidates = root.findall(
                f"{{*}}Instrument[@ID='{inst_ref.get('ID')}']/{{*}}Objective")
    if not candidates:
        candidates = objectives
    if len(candidates) != 1:
        return None
    mag = candidates[0].get("NominalMagnification")
    return None if mag is None else float(mag)


def yxc_selection(axes: str, shape: tuple[int, ...],
                  selection: dict[str, int] | None = None) -> tuple[list, list[int], int]:
    """Map a tifffile series to the canonical YXC layout.

    Args:
        axes: tifffile axes string, e.g. ``"YXS"``, ``"CYX"``, ``"TCZYX"``.
        shape: the series shape (same length as ``axes``).
        selection: index to take along non-spatial, non-channel axes (e.g.
            ``{"Z": 0, "T": 0}``). Such axes of size 1 need no entry; larger
            ones must be selected explicitly.

    Returns:
        ``(index, perm, nchannels)``: ``index`` is a per-axis list with
        ``slice(None)`` for Y, X and the channel axis and an int elsewhere —
        replace the Y/X entries with the wanted ranges, index the series,
        then ``np.transpose(block, perm)`` gives (h, w) (when
        ``nchannels == 1``) or (h, w, nchannels).
    """
    selection = {} if selection is None else selection
    if len(axes) != len(shape):
        raise ValueError(f"axes {axes!r} do not match shape {shape}")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"series axes {axes!r} lack Y/X")

    index: list = []
    kept: list[str] = []
    for ax, n in zip(axes, shape):
        if ax in "YX" or (ax in _CHANNEL_AXES and n > 1):
            index.append(slice(None))
            kept.append(ax)
        elif ax in selection:
            if not 0 <= selection[ax] < n:
                raise IndexError(f"{ax}={selection[ax]} out of range (size {n})")
            index.append(selection[ax])
        elif n == 1:
            index.append(0)
        else:
            raise ValueError(
                f"axis {ax!r} has size {n}: select one plane (e.g. {ax.lower()}=0)")

    chan = [ax for ax in kept if ax in _CHANNEL_AXES]
    if len(chan) > 1:
        raise ValueError(f"more than one channel axis with size > 1 in {axes!r}")
    perm = [kept.index("Y"), kept.index("X")] + [kept.index(ax) for ax in chan]
    nchannels = shape[axes.index(chan[0])] if chan else 1
    return index, perm, nchannels
