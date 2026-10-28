# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Annotation overlay data for the viewer — the headless half.

Reads one or more annotation stores (read-only) and turns every group into
drawable arrays in level-0 pixel coordinates, the coordinate system of the
canvas: line segments (polygon rings and polylines) and marker positions
(points). Colours come from a colour file or, failing that, from a fixed
qualitative palette. No Qt, no OpenGL: the vispy visuals and the Qt panel that
consume these objects are in :mod:`compass.viewer.overlay_visuals` and
:mod:`compass.viewer.app`.

The whole group is loaded into memory, which suits QC of region/landmark
annotations (hundreds to a few thousand objects per slide). A group above
``max_objects`` is skipped with a warning rather than freezing the viewer;
dense cell-level layers will need the viewport-queried, level-of-detail
overlay of docs/architecture.md section 7, step 4.
"""

import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import shapely

from ..core.annot import AnnotationStore

log = logging.getLogger(__name__)

#: Okabe-Ito colour-blind-safe qualitative palette, used for groups without
#: an explicit colour (cycled)
DEFAULT_PALETTE: tuple[str, ...] = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
)

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

#: groups with more objects than this are not loaded
DEFAULT_MAX_OBJECTS = 200_000


Rgba = tuple[float, float, float, float]


def hex_to_rgba(color: str, alpha: float = 1.0) -> Rgba:
    """``"#rrggbb"`` -> ``(r, g, b, a)`` floats in [0, 1]."""
    if not _HEX.match(str(color)):
        raise ValueError(f"not a #rrggbb colour: {color!r}")
    r, g, b = (int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return (r, g, b, float(alpha))


def read_colors(path: str | Path) -> dict[str, str]:
    """Group colours from a TOML file: group name -> ``"#rrggbb"``.

    Two layouts are accepted, and may be mixed::

        [colors]                      # a flat table
        "Pattern CT" = "#0072B2"

        [[group]]                     # a list of groups (extra keys ignored,
        name = "Pattern DE"           # so a vocabulary file works as is)
        color = "#D55E00"

    Raises ``ValueError`` for a malformed colour.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    out: dict[str, str] = {}
    for name, col in (data.get("colors") or {}).items():
        out[str(name)] = str(col)
    for g in data.get("group") or []:
        if "name" in g and "color" in g:
            out[str(g["name"])] = str(g["color"])
    bad = {n: c for n, c in out.items() if not _HEX.match(c)}
    if bad:
        raise ValueError(f"{path}: colours must be #rrggbb, got {bad}")
    return out


def assign_colors(names: Sequence[str],
                  given: Mapping[str, str] | None = None,
                  palette: Sequence[str] = DEFAULT_PALETTE) -> dict[str, str]:
    """A colour for every name: the given one, else the next palette colour.

    Palette colours are handed out in order of first appearance and skip the
    ones already taken by explicit colours while any are left, so unlisted
    groups stay distinguishable from listed ones.
    """
    given = dict(given or {})
    taken = {c.lower() for c in given.values()}
    free = [c for c in palette if c.lower() not in taken] or list(palette)
    out: dict[str, str] = {}
    k = 0
    for n in names:
        if n in out:
            continue
        if n in given:
            out[n] = given[n]
        else:
            out[n] = free[k % len(free)]
            k += 1
    return out


@dataclass
class OverlayGroup:
    """One annotation group, ready to draw.

    Attributes:
        uid: viewer-wide unique id (what ``ViewerState.visible_groups`` holds;
            store group ids alone would collide across stores).
        source: label of the store it came from (file stem).
        layer: layer name in that store.
        name: group name.
        n_objects: number of objects in the group.
        segments: float32 ``(2k, 2)`` array of segment end points (pairs of
            rows); polygon rings (exterior and holes) and polylines.
        points: float32 ``(m, 2)`` marker positions (points, point sets).
        color: RGBA floats.
    """
    uid: int
    source: str
    layer: str
    name: str
    n_objects: int
    segments: np.ndarray
    points: np.ndarray
    color: Rgba = (1.0, 1.0, 1.0, 1.0)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.n_objects})"

    @property
    def is_empty(self) -> bool:
        return len(self.segments) == 0 and len(self.points) == 0


@dataclass
class Overlay:
    """All loaded groups plus anything worth telling the user."""
    groups: list[OverlayGroup] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_uid(self) -> dict[int, OverlayGroup]:
        return {g.uid: g for g in self.groups}

    def sources(self) -> list[str]:
        seen: list[str] = []
        for g in self.groups:
            if g.source not in seen:
                seen.append(g.source)
        return seen


def geometry_arrays(geoms: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Drawable arrays for shapely geometries.

    Returns ``(segments, points)``: segment end points as consecutive row
    pairs (polygon rings, holes included, are closed), and point
    coordinates. Multi-part geometries and collections are split into parts;
    ``None``/empty geometries are ignored.
    """
    arr = np.asarray(list(geoms), dtype=object)
    if arr.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    arr = arr[~shapely.is_missing(arr)]
    arr = arr[~shapely.is_empty(arr)] if arr.size else arr
    parts = shapely.get_parts(arr) if arr.size else arr
    # nested collections: one more level is enough for what the store holds
    kind = shapely.get_type_id(parts) if parts.size else np.empty(0, int)
    if parts.size and (kind == 7).any():
        parts = shapely.get_parts(parts)
        kind = shapely.get_type_id(parts)

    points = shapely.get_coordinates(parts[kind == 0]).astype(np.float32)
    lines = parts[(kind == 1) | (kind == 2)]
    polys = parts[kind == 3]
    rings = shapely.get_rings(polys) if polys.size else polys
    paths = np.concatenate([lines, rings]) if (lines.size or rings.size) else lines

    if paths.size == 0:
        return np.empty((0, 2), np.float32), points.reshape(-1, 2)
    xy, idx = shapely.get_coordinates(paths, return_index=True)
    same = idx[1:] == idx[:-1]
    a, b = xy[:-1][same], xy[1:][same]
    seg = np.empty((2 * len(a), 2), np.float32)
    seg[0::2], seg[1::2] = a, b
    return seg, points.reshape(-1, 2)


def load_store_groups(path: str | Path, *, first_uid: int = 0,
                      max_objects: int = DEFAULT_MAX_OBJECTS
                      ) -> tuple[list[OverlayGroup], list[str]]:
    """Load every non-empty group of one store.

    Returns ``(groups, warnings)``. Groups are listed layer by layer, in group
    id order. Colours are left at their default; see :func:`load_overlay`.
    """
    path = Path(path)
    source = path.name
    for suf in (".ann.sqlite", ".sqlite", ".db"):
        if source.endswith(suf):
            source = source[:-len(suf)]
            break
    warns: list[str] = []
    groups: list[OverlayGroup] = []
    with AnnotationStore(path, mode="r") as store:
        layers = {int(r["layer_id"]): str(r["name"]) for r in store.get_layers().iter_rows(named=True)}
        mem = store.get_memberships()
        uid = first_uid
        for row in store.get_groups().sort(["layer_id", "group_id"]).iter_rows(named=True):
            gid = int(row["group_id"])
            ids = mem.filter(mem["group_id"] == gid)["object_id"].to_numpy()
            if ids.size == 0:
                continue
            if ids.size > max_objects:
                warns.append(f"{path.name}: group {row['name']!r} has {ids.size} objects "
                             f"(> {max_objects}); not shown")
                continue
            gdf = store.get_objects(ids)
            seg, pts = geometry_arrays(gdf.geometry.to_numpy())
            groups.append(OverlayGroup(uid=uid, source=source,
                                       layer=layers.get(int(row["layer_id"]), "?"),
                                       name=str(row["name"]), n_objects=int(ids.size),
                                       segments=seg, points=pts))
            uid += 1
    return groups, warns


def load_overlay(paths: Sequence[str | Path],
                 colors: Mapping[str, str] | None = None, *,
                 max_objects: int = DEFAULT_MAX_OBJECTS) -> Overlay:
    """Load stores and colour their groups.

    A group name has one colour across all stores (a name given in ``colors``
    uses it, the others are assigned from the palette in order of first
    appearance), so the same morphotype looks the same in every file.
    """
    ov = Overlay()
    for p in paths:
        gs, ws = load_store_groups(p, first_uid=len(ov.groups), max_objects=max_objects)
        ov.groups.extend(gs)
        ov.warnings.extend(ws)
    cmap = assign_colors([g.name for g in ov.groups], colors)
    for g in ov.groups:
        g.color = hex_to_rgba(cmap[g.name])
    unused = sorted(set(colors or ()) - {g.name for g in ov.groups})
    if unused:
        log.info("colour file lists groups not in the loaded stores: %s", unused)
    return ov
