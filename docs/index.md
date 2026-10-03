---
title: "COMPASS"
subtitle: "Developer documentation — current (non-legacy) codebase, v0.2.0"
author: "Vlad Popovici"
format:
  html:
    toc: true
    toc-depth: 3
    toc-location: left
    number-sections: true
    code-copy: true
    embed-resources: true
    theme: cosmo
---

COMPASS is a local, non-browser, single-process Qt application for viewing
large pathology whole-slide images (rasters up to ~200,000 × 200,000 px,
multi-channel, micron-calibrated) together with millions of associated
vector and raster annotations (cell-level points, region polygons,
segmentation maps, sparse molecular profiles).

This document covers everything under `src/compass/` **except**
`src/compass/legacy/` (the original flat modules, kept unchanged while they
are ported). It is the *what/how*: package layout, API, file formats and
worked examples. The *why* behind each design choice — Zarr instead of
HDF5, no browser/JS, OpenSlide confined to ingestion, no napari — is the
decision log in [`architecture.md`](architecture.md).

Unless marked otherwise, every example below was run against the code while
writing this document; output blocks are real output. Examples that need
OpenGL, libvips or OpenSlide are marked as such.

# Overview

## Packages and status

| Package | Purpose | Status |
|---|---|---|
| `compass.core` | `PyramidalRaster` protocol, Zarr raster backend, magnification math, vectorized SQLite/R-tree annotation reader | Ported, tested |
| `compass.connector` | WSI ingestion (OpenSlide → pyvips → Zarr), OME-TIFF export | Ported; ingestion tested against stand-ins (no libvips/OpenSlide needed) |
| `compass.processing` | Domain algorithms (tissue detection, stain normalization, sampling, registration, Visium) | Empty stub |
| `compass.viewer` | psygnal state, tile grid/cache/fetcher, vispy tile-streaming canvas, `python -m compass.viewer` | Steps 1–3 of the build order (architecture §7). No annotation overlays or analysis panels yet |

## Dependency direction

```
core          <- everything (minimal deps; safe to import from anywhere)
connector     -> core             (never imported by viewer or processing)
processing    -> core             (independent of connector)
viewer        -> core, processing (never connector)
```

`import compass.core` stays fast and never pulls in Qt, OpenSlide, pyvips,
OpenCV, h5py or AnnData. That's what lets `core` be imported from a
notebook, a batch script or another package without the heavy stack.
`compass.connector` imports pyvips and OpenSlide lazily, so importing the
package itself needs neither system library either.

## Conventions used throughout

- **Coordinates are `(x, y)`**, in pixels at a stated pyramid level, unless
  explicitly in microns. Arrays are indexed `[y, x, channel]`.
- **Level 0** is full resolution; level *L* is downscaled by
  `magnif_step ** L` (2 by default).
- **World coordinates** in the viewer and the annotation store are level-0
  pixels, y pointing down.
- **Bulk data comes back vectorized**: numpy/dask arrays, polars
  DataFrames, geopandas GeoDataFrames — never one Python object per
  annotation.

# Installation and tests

The project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                     # core deps + dev group
uv sync --extra connector   # + ingestion/export: pyvips, OpenSlide, tifffile
uv sync --extra processing  # + scikit-image, scikit-learn, OpenCV, SimpleITK
uv sync --extra viewer      # + vispy, PySide6, psygnal (needs OpenGL)
```

`pyvips` and `openslide-python` are bindings; the connector also needs the
**libvips** system library (OpenSlide's binary ships in `openslide-bin`).
The viewer needs a working OpenGL driver — vispy can't even be *imported*
without `libGL`.

```bash
uv run pytest src/compass/tests -q
```

The suite covers `core` (magnification math, Zarr raster I/O, annotation
store), the headless parts of `viewer` (state, tile grid/cache, fetcher),
and `connector.ingest` (planning helpers, and the full `wsi2zarr` pipeline
against numpy stand-ins for pyvips and OpenSlide). It doesn't exercise the
vispy canvas or the Qt entry point.

# Quick start

The typical path is: bake a vendor slide into Zarr once, then read it (and
its annotations) from Python or the viewer.

```python
# 1. one-time ingestion (needs `uv sync --extra connector` + libvips)
from compass.connector import wsi2zarr

wsi2zarr("slide.mrxs", "slide.zarr", crop=True)
```

```python
# 2. read pixels and annotations — core only, no heavy deps
from compass.core import ZarrRaster, AnnotationStore

raster = ZarrRaster("slide.zarr")
level = raster.get_level_for_mpp(1.0)                     # ~10x
tile = raster.get_region_px(0, 0, 1024, 1024, level=level)

with AnnotationStore("slide.ann.sqlite") as store:
    cells = store.get_objects_in_roi(store.get_layer_id("cells"),
                                     roi=(0, 0, 20_000, 20_000))
```

```bash
# 3. look at it (needs `uv sync --extra viewer` + OpenGL)
uv run python -m compass.viewer slide.zarr
```

# File formats

## Raster pyramid (Zarr v3, sharded)

One Zarr group per image; one array per level, named `"0"`, `"1"`, ….
Written by `core.write_pyramid` (in-memory arrays) and
`connector.wsi2zarr` (streamed from a slide); read by `core.ZarrRaster`.

| Item | Value |
|---|---|
| Array shape | `(height, width, channels)` or `(height, width)` |
| Chunks | `chunk_size × chunk_size` (default 512 — the viewer's tile size), clipped on levels smaller than one chunk |
| Shards | `shard_factor × shard_factor` chunks (default 8 → 4096 × 4096 px) |
| Codec | Blosc zstd, level 3, byte shuffle |
| Group attrs | `compass_version` (= 2), `level_count`, `base_mpp`, `base_mag_step`, `base_objective_power`, `channel_names`, `dimension_names` |
| Per-level attrs | `mpp_x`, `mpp_y`, `objective_power`, `scale_factor` |

Format version 1 was the legacy HDF5 layout.

## Annotation database (SQLite + R-tree)

The schema is defined in `compass.core.SQLITE_DDL` (byte-identical to the
legacy writer's, checked by a test). Create an empty database with
`compass.core.init_db(path)`.

| Table | Contents |
|---|---|
| `meta` | key/value: `schema_version`, plus free-form keys (`name`, `mpp`, image size, …) |
| `layers` | `layer_id`, unique `name` |
| `groups` | `group_id`, `layer_id`, `name` (unique within a layer) |
| `objects` | `object_id`, `type_code`, `name`, `wkb` geometry, `circle_r`, bbox `min_x/min_y/max_x/max_y` |
| `object_groups` | many-to-many object ↔ group; a trigger forbids one object being in groups of *different* layers |
| `objects_rtree` | R-tree over the bboxes, kept in sync by triggers |
| `scalar_attributes` | attribute dictionary `attr_id` ↔ `attr_name` (e.g. gene names) |
| `object_scalar_sparse` | per object: packed `uint32` attribute ids + `float32` values |

Geometry type codes (`ANNOT_TYPE_CODE`): `POINT=1`, `POINTSET=2`,
`POLYLINE=3`, `POLYGON=4`, `CIRCLE=5`. A circle is stored as its center
`POINT` plus `circle_r`; its bbox must include the radius.

# `compass.core`

```python
from compass.core import (ZarrRaster, write_pyramid, PyramidalRaster,
                          Magnification, AnnotationStore, init_db,
                          ImageShape, Px, rasterize_polygon)
```

Dependencies: numpy, zarr, shapely, geopandas, polars, pandas, pydantic,
dask. Depends on nothing else in the project.

## Value types: `ImageShape`, `Px`

Small pydantic models.

```python
from compass.core import ImageShape, Px

ImageShape(width=6144, height=4096)
Px(x=100, y=60)
```

## `Magnification`

Converts between pyramid level, microns per pixel (mpp) and objective
power, from a reference point (by default level 0) and the scale factor
between levels. Rasters build one from their metadata. Construct one
yourself only for a pyramid that has no raster wrapper.

```python
from compass.core import Magnification

mag = Magnification(magnif=40.0, mpp=0.25, n_levels=6, magnif_step=2.0)
mag.get_level_for_mpp(1.0)       # 2
mag.get_mpp_for_level(2)         # 1.0
mag.get_magnif_for_level(2)      # 10.0
mag.get_level_for_magnif(10.0)   # 2
mag.get_mpp_for_magnif(20.0)     # 0.5
mag.get_magnif_for_mpp(0.5)      # 20.0
```

Lookups snap to the nearest level. Requests slightly outside the pyramid
(≤ 10 % relative error) clamp to the end level; further out they raise:

```python
mag.get_level_for_mpp(0.24)   # 0   (within tolerance of 0.25)
mag.get_level_for_mpp(0.1)    # RuntimeError: mpp outside supported interval
```

Properties: `base_magnif`, `base_mpp`, `magnif_step`, `nlevels`.

## `PyramidalRaster` (protocol)

Abstract base class for every raster source (`ZarrRaster` at runtime,
`connector.wsi.WSI` during ingestion). Implementations provide `info` and
`get_region_px`; everything else comes from the base class:

| Member | Description |
|---|---|
| `nlevels`, `widths`, `heights`, `pyramid_levels` | Pyramid geometry (arrays per level; `pyramid_levels` is `2 × nlevels`) |
| `shape(level)` | `ImageShape` at a level |
| `native_magnification`, `native_resolution` | Level-0 objective power and mpp |
| `get_level_for_mpp`, `get_level_for_magnification`, `get_mpp_for_level`, `get_magnification_for_level`, `get_mpp_for_magnification`, `get_magnification_for_mpp` | Delegated to the `Magnification` |
| `between_level_scaling_factor(from_level, to_level)` | Multiplier taking coordinates from one level to another |
| `convert_px(point, from_level, to_level)` | Convert a `Px` between levels |
| `get_region_px(x0, y0, width, height, level)` | Rectangular read (abstract) |
| `get_region(top_left: Px, shape: ImageShape, level)` | Same, with value types |
| `get_plane(level)` | Whole level |
| `get_polygonal_region_px(contour, level, border=0)` | Bounding-box read of a polygon with outside pixels zeroed |

## `ZarrRaster`

Read-only `PyramidalRaster` over a Zarr store in the layout above. The
examples in this section use a synthetic 3-level pyramid written with
`write_pyramid`:

```python
import numpy as np
from compass.core import write_pyramid, ZarrRaster

rng = np.random.default_rng(0)
base = rng.integers(0, 255, size=(4096, 6144, 3), dtype=np.uint8)
write_pyramid("demo.zarr", [base, base[::2, ::2], base[::4, ::4]],
              base_mpp=0.25, base_objective_power=40.0, overwrite=True)

r = ZarrRaster("demo.zarr")
r.nlevels, r.shape(0), r.shape(2)
```

```
(3, ImageShape(width=6144, height=4096), ImageShape(width=1536, height=1024))
```

**Metadata**

```python
r.channel_names, r.nchannels     # (['R', 'G', 'B'], 3)
r.get_mpp_for_level(2)           # 1.0
r.get_level_for_mpp(0.5)         # 1
r.get_magnification_for_level(1) # 20.0
r.level_info(1)
# {'mpp_x': 0.5, 'mpp_y': 0.5, 'objective_power': 20.0, 'scale_factor': 2.0}
r.info
# {'compass_version': 2, 'level_count': 3, 'base_mpp': 0.25,
#  'base_mag_step': 2.0, 'base_objective_power': 40.0,
#  'channel_names': ['R', 'G', 'B'], 'dimension_names': ['y', 'x', 'c']}
```

`r.path` and `r.storage` (the open `zarr.Group`) are also exposed.

**Reading pixels**

```python
from compass.core import Px, ImageShape

tile = r.get_region_px(1024, 512, 512, 256, level=1)   # x0, y0, w, h
tile.shape, tile.dtype                                 # ((256, 512, 3), uint8)

same = r.get_region(Px(x=1024, y=512), ImageShape(width=512, height=256), level=1)

thumb = r.get_plane(level=r.nlevels - 1)               # whole coarsest level
```

Reads outside the level extent raise `RuntimeError("region out of layer's
extent")`. Unlike `WSI`, `ZarrRaster` does not pad or clip. Pixels come
back in their stored dtype; the `as_type` argument is accepted for
interface compatibility and ignored.

**Lazy reads (dask)** — chunked at the Zarr chunk size, so computations
stream chunk by chunk and parallelize across threads (Zarr has no global
lock):

```python
lazy = r.get_region_px_dask(0, 0, 2048, 2048, level=0)
lazy.shape, lazy.chunksize          # ((2048, 2048, 3), (512, 512, 3))
lazy.mean(axis=(0, 1)).compute()    # per-channel mean, without loading 12 MB at once

whole = r.get_plane_dask(level=0)   # the full level, lazily
```

**Converting coordinates between levels**

```python
r.convert_px(Px(x=100, y=60), from_level=2, to_level=0)   # Px(x=400, y=240)
r.between_level_scaling_factor(2, 0)                      # 4.0
```

**Polygonal regions** — reads the polygon's bounding box (plus an optional
`border`) and zeroes everything outside it. The polygon must be in pixel
coordinates *of the requested level*:

```python
import shapely

poly = shapely.Polygon([(100, 100), (400, 120), (300, 380)])
crop = r.get_polygonal_region_px(poly, level=0, border=8)
crop.shape    # (296, 316, 3)
```

## `write_pyramid`

Writes the Zarr layout from fully materialized per-level arrays — for maps
(segmentations, masks), test fixtures and small images. Use
`connector.wsi2zarr` for whole slides.

```python
write_pyramid(path, levels, *, base_mpp, base_objective_power,
              mag_step=2.0, channel_names=None,
              chunk_size=512, shard_factor=8, overwrite=False)
```

`levels` is level 0 first; all levels must share dtype and channel count.
`channel_names` defaults to `["R", "G", "B"]` for 3 channels. A
single-channel map, pixel-aligned with the slide:

```python
labels = np.zeros((2048, 2048), np.uint8)
write_pyramid("regions.zarr", [labels, labels[::2, ::2]],
              base_mpp=0.25, base_objective_power=40.0,
              channel_names=["region"], overwrite=True)

m = ZarrRaster("regions.zarr")
m.nchannels, m.channel_names, m.get_region_px(0, 0, 4, 4).shape
# (1, ['region'], (4, 4))
```

Subsample nearest-neighbor (`[::2, ::2]`) for label maps, never
interpolate.

## `rasterize_polygon`

Binary (`uint8`) mask of a polygon; a pixel is 1 if its center is on or
inside the polygon (holes excluded). Used by `get_polygonal_region_px`,
handy on its own:

```python
from compass.core import rasterize_polygon

rasterize_polygon(shapely.Polygon([(0, 0), (4, 0), (4, 4)]), shape=(5, 5))
```

```
[[1 1 1 1 1]
 [0 1 1 1 1]
 [0 0 1 1 1]
 [0 0 0 1 1]
 [0 0 0 0 1]]
```

## `AnnotationStore`

Read-only, vectorized access to one annotation database. Bulk queries
return **polars DataFrames** (tables, ids, scalars) or **geopandas
GeoDataFrames indexed by `object_id`** (geometry). Geometry is decoded
with a single `shapely.from_wkb` call per query.

### Creating a database to work with

The bulk *writer* is not ported out of `legacy` yet. Until it is, a
database can be built with `init_db` and plain SQL. This one (used by the
examples below) has 1,000 cells in two groups, a region polygon, a circle
and sparse marker values:

```python
import sqlite3
import numpy as np
import shapely
from compass.core import init_db

init_db("demo.ann.sqlite")
con = sqlite3.connect("demo.ann.sqlite")
con.execute("PRAGMA foreign_keys = ON")
con.executemany("INSERT INTO meta(key, value) VALUES (?, ?)",
                [("name", "demo"), ("mpp", "0.25")])
con.execute("INSERT INTO layers VALUES (1, 'cells'), (2, 'regions')")
con.execute("INSERT INTO groups VALUES (1, 1, 'tumor'), (2, 1, 'stroma'), "
            "(3, 2, 'annotated')")

def add(oid, code, geom, group, name=None, r=None):
    x0, y0, x1, y1 = geom.bounds
    if r is not None:  # circle: bbox includes the radius
        x0, y0, x1, y1 = x0 - r, y0 - r, x1 + r, y1 + r
    con.execute("INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?)",
                (oid, code, name, shapely.to_wkb(geom), r, x0, y0, x1, y1))
    con.execute("INSERT INTO object_groups VALUES (?, ?)", (oid, group))

rng = np.random.default_rng(0)
for i, (x, y) in enumerate(rng.uniform(0, 6000, size=(1000, 2)), start=1):
    add(i, 1, shapely.Point(x, y), group=1 if x < 3000 else 2, name=f"cell{i}")
add(1001, 4, shapely.box(500, 500, 2500, 2000), group=3, name="roi A")
add(1002, 5, shapely.Point(4000, 4000), group=3, name="spot", r=150.0)

con.executemany("INSERT INTO scalar_attributes VALUES (?, ?)",
                [(0, "CD3"), (1, "CD8"), (2, "PanCK")])
for i in range(1, 1001):
    ids = np.array([0, 2] if i % 2 else [1], np.uint32)
    vals = rng.random(ids.size).astype(np.float32)
    con.execute("INSERT INTO object_scalar_sparse VALUES (?, ?, ?)",
                (i, ids.tobytes(), vals.tobytes()))
con.commit()
con.close()
```

The R-tree is filled by triggers; only `objects` needs inserting.

### Opening and browsing the hierarchy

The store opens the file read-only. Use it as a context manager (or call
`close()`).

```python
from compass.core import AnnotationStore

store = AnnotationStore("demo.ann.sqlite")
store.meta                      # {'schema_version': '1', 'name': 'demo', 'mpp': '0.25'}
store.get_layers()              # layer_id, name
store.get_groups(layer_id=1)    # group_id, layer_id, name
store.count_objects_per_group()
```

```
shape: (3, 4)
┌──────────┬──────────┬────────────┬───────────┐
│ group_id ┆ layer_id ┆ group_name ┆ n_objects │
│ ---      ┆ ---      ┆ ---        ┆ ---       │
│ i64      ┆ i64      ┆ str        ┆ i64       │
╞══════════╪══════════╪════════════╪═══════════╡
│ 1        ┆ 1        ┆ tumor      ┆ 472       │
│ 2        ┆ 1        ┆ stroma     ┆ 528       │
│ 3        ┆ 2        ┆ annotated  ┆ 2         │
└──────────┴──────────┴────────────┴───────────┘
```

Names resolve to ids. Group names are unique only within a layer, so pass
`layer_id` to disambiguate:

```python
cells = store.get_layer_id("cells")                     # 1
tumor = store.get_group_id("tumor", layer_id=cells)     # 1
store.get_layer_id("nuclei")
# ValueError: Unknown layer name 'nuclei'. Available layers: cells, regions
```

### Viewport queries

All ROIs are `(x0, y0, x1, y1)` in level-0 pixels and match on bounding
box intersection via the R-tree.

```python
roi = (0, 0, 2000, 2000)
store.count_objects_in_roi(cells, roi)     # 107 — R-tree only, no geometry

gdf = store.get_objects_in_roi(cells, roi, group_ids=[tumor])
gdf.head(3)
```

```
            type    name  circle_r                  geometry
object_id
1          POINT   cell1       NaN  POINT (1470.754 181.037)
13         POINT  cell13       NaN   POINT (196.729 153.311)
18         POINT  cell18       NaN   POINT (695.879 917.332)
```

`group_ids` restricts the query to the currently visible groups. The
columns are always `type` (categorical), `name`, `circle_r` (NaN except
circles) and `geometry`.

### Fetching by id

```python
store.get_object_ids_in_layer(cells)            # np.ndarray[int64], 1000 ids
store.get_object_ids_in_group(tumor)            # 472 ids
store.get_object_ids_in_roi(cells, roi)         # array([ 1, 13, 18, 30, 32, ...])

regions = store.get_objects(store.get_object_ids_in_layer(2))
```

```
              type   name  circle_r                                       geometry
object_id
1001       POLYGON  roi A       NaN  POLYGON ((2500 500, 2500 2000, 500 2000, 500 5...
1002        CIRCLE   spot     150.0  POLYGON ((4150 4000, 4147.118 3970.736, 4138.5...
```

Circles are decoded to polygonal disks by default so every geometry can
be drawn uniformly. Pass `circles_as_disks=False` to keep the center
points (the radius stays in `circle_r`):

```python
store.get_objects([1002], circles_as_disks=False).geometry.iloc[0].geom_type   # 'Point'
```

Id lists of any length work; queries are chunked below SQLite's
bound-parameter limit.

### Bounding boxes and memberships

When geometry isn't needed (coarse rendering, density maps, hit-testing),
skip decoding entirely:

```python
store.get_bboxes(layer_id=2)
```

```
┌───────────┬────────┬────────┬────────┬────────┐
│ object_id ┆ min_x  ┆ min_y  ┆ max_x  ┆ max_y  │
╞═══════════╪════════╪════════╪════════╪════════╡
│ 1001      ┆ 500.0  ┆ 500.0  ┆ 2500.0 ┆ 2000.0 │
│ 1002      ┆ 3850.0 ┆ 3850.0 ┆ 4150.0 ┆ 4150.0 │
└───────────┴────────┴────────┴────────┴────────┘
```

```python
store.get_memberships([1, 2, 1001])   # object_id, group_id (all objects if omitted)
```

### Sparse per-object scalars (expression profiles)

```python
store.get_scalar_attribute_names()      # attr_id, attr_name
store.get_scalar_data_long([1, 2])      # COO form
```

```
┌───────────┬─────────┬──────────┐
│ object_id ┆ attr_id ┆ value    │
╞═══════════╪═════════╪══════════╡
│ 1         ┆ 0       ┆ 0.501237 │
│ 1         ┆ 2       ┆ 0.419153 │
│ 2         ┆ 1       ┆ 0.012871 │
└───────────┴─────────┴──────────┘
```

Wide form, restricted to the attributes you actually need (with ~20,000
genes, always restrict):

```python
store.get_scalar_data(gdf.index.to_numpy()[:4], attributes=["CD3", "CD8"])
```

```
┌───────────┬──────────┬──────────┐
│ object_id ┆ CD3      ┆ CD8      │
╞═══════════╪══════════╪══════════╡
│ 1         ┆ 0.501237 ┆ null     │
│ 13        ┆ 0.393233 ┆ null     │
│ 18        ┆ null     ┆ 0.308101 │
│ 30        ┆ null     ┆ 0.613048 │
└───────────┴──────────┴──────────┘
```

Absent values are `null`. Columns follow the order attributes appear in
the data, so select by name rather than position. Objects with no values
at all get no row.

### Method summary

| Method | Returns |
|---|---|
| `meta` | `dict[str, str]` |
| `get_layers()` | polars: `layer_id`, `name` |
| `get_groups(layer_id=None)` | polars: `group_id`, `layer_id`, `name` |
| `get_layer_id(name)`, `get_group_id(name, layer_id=None)` | `int` (`ValueError` if unknown) |
| `count_objects_per_group()` | polars: `group_id`, `layer_id`, `group_name`, `n_objects` |
| `count_objects_in_roi(layer_id, roi)` | `int` |
| `get_object_ids_in_layer / _in_group / _in_roi(...)` | `np.ndarray[int64]` |
| `get_objects(ids, circles_as_disks=True)` | GeoDataFrame |
| `get_objects_in_roi(layer_id, roi, group_ids=None, circles_as_disks=True)` | GeoDataFrame |
| `get_bboxes(layer_id=None)` | polars: `object_id`, `min_x`, `min_y`, `max_x`, `max_y` |
| `get_memberships(ids=None)` | polars: `object_id`, `group_id` |
| `get_scalar_attribute_names()` | polars: `attr_id`, `attr_name` |
| `get_scalar_data_long(ids)` | polars: `object_id`, `attr_id`, `value` |
| `get_scalar_data(ids, attributes=None)` | polars, wide: `object_id` + one column per attribute |

Module-level constants: `SQLITE_DDL`, `SCHEMA_VERSION` (`"1"`),
`ANNOT_TYPE_CODE`, `ANNOT_CODE_TYPE`.

# `compass.connector`

```python
from compass.connector import wsi2zarr, raster2tiff, build_omexml, plan_levels, resolve_crop
from compass.connector.wsi import WSI   # imports OpenSlide
```

One-shot ingestion and interchange tools. Never imported by the viewer or
by `processing`. Needs `uv sync --extra connector` and libvips for the
actual conversions. `plan_levels` and `resolve_crop` are pure Python.

## `WSI` — ingestion-only slide reader

An OpenSlide-backed `PyramidalRaster`, used by `wsi2zarr` to read
metadata. **Do not use it as a runtime raster.** OpenSlide and
Bio-Formats disagree on MRXS coordinate offsets, so OpenSlide-read pixels
are only trusted after the converted pyramid has been checked against
known landmark annotations (architecture §3).

```python
from compass.connector.wsi import WSI      # needs OpenSlide

wsi = WSI("slide.mrxs")
wsi.info
# {'objective_power': 20.0, 'width': ..., 'height': ...,
#  'mpp_x': ..., 'mpp_y': ..., 'n_levels': ..., 'magnification_step': ...,
#  'roi': {'x0': ..., 'y0': ..., 'width': ..., 'height': ...},  # {} if none
#  'background': 255}
wsi.get_region_px(0, 0, 512, 512, level=2)  # RGB; transparent pixels -> background
```

## `wsi2zarr` — bake a slide into the Zarr pyramid

```python
wsi2zarr(wsi_path, dst_path, crop=False, downscale_factor=2, min_size=256,
         chunk_size=512, shard_factor=8, tmp_dir=None, overwrite=False)
```

| Parameter | Meaning |
|---|---|
| `crop` | `False`: whole slide. `True`: crop to the scanner's scan-region (ROI) metadata, if any. `(x0, y0, w, h)`: explicit level-0 region, clamped to the image |
| `downscale_factor` | Scale between consecutive levels |
| `min_size` | Stop before either side of a level would go below this |
| `chunk_size` | Chunk edge in pixels — match the viewer's `--tile-size` |
| `shard_factor` | Shard edge in chunks |
| `tmp_dir` | Where the temporary `.v` intermediates go (default: next to `dst_path`) |
| `overwrite` | Replace an existing store; otherwise an existing one raises |

```python
from compass.connector import wsi2zarr
import logging

logging.basicConfig(level=logging.INFO)    # progress: "generating level N", "copying level N"
wsi2zarr("slide.mrxs", "/data/slides/slide.zarr", crop=True)
```

**How it works.** Level 0 is the (optionally cropped) slide, flattened
onto the background color and written to a temporary uncompressed
`level0.v`. Each further level is a pyvips downscale of the previous `.v`.
Each level is then copied into Zarr **one whole shard at a time**. Shards
are never partially written, and the copy step's memory stays at about one
shard (~50 MB at the defaults) whatever the slide size. The resulting
pyramid is regenerated rather than copied from the scanner's own levels,
so the level structure is always the same.

**Disk and memory.** The `.v` intermediates add up to a few times the
uncompressed size of level 0 (tens of GB for a large slide), and they are
written and read back several times. If `dst_path` is on a slow
external/USB or network drive, put them on a fast local disk:

```python
wsi2zarr("slide.mrxs", "/media/usb/slide.zarr", tmp_dir="/home/me/scratch")
```

Otherwise the OS write cache fills with data waiting for the slow disk and
squeezes the RAM available to the process. Don't point `tmp_dir` at a
RAM-backed `/tmp` (tmpfs, the default on several Linux distributions),
because the intermediates are slide-sized. A finished Zarr store is a
plain directory, so converting locally and copying it afterwards also
works.

**Verify before trusting.** For each new scanner/format combination,
check a few landmark annotations against the converted raster
(architecture §3).

## Planning helpers

The pure functions `wsi2zarr` uses to decide what it will do — useful to
preview a conversion:

```python
from compass.connector import plan_levels, resolve_crop

resolve_crop(1000, 800, roi={}, crop=True)        # (0, 0, 1000, 800): no ROI -> whole image
resolve_crop(1000, 800, roi={'x0': 10, 'y0': 20, 'width': 300, 'height': 400}, crop=True)
# (10, 20, 300, 400)
resolve_crop(1000, 800, {}, (-5, 900, 5000, 100)) # (0, 799, 1000, 1): clamped

plan_levels(1400, 1100, downscale_factor=2, min_size=256)
# [{'w': 1400, 'h': 1100, 'x0': 0, 'y0': 0},
#  {'w': 700,  'h': 550,  'x0': 0, 'y0': 0},
#  {'w': 350,  'h': 275,  'x0': 0, 'y0': 0}]
```

## `raster2tiff` — OME-TIFF export

Zarr is the internal format. For QuPath, ASAP, MIKAIA and clinical
viewers, export a pyramidal, tiled, LZW-compressed BigTIFF with OME-XML
metadata:

```python
from compass.connector import raster2tiff
from compass.core import ZarrRaster

xml = raster2tiff(ZarrRaster("slide.zarr"), "slide.tiff",
                  start_level=0,          # highest resolution to export
                  tile_shape=(1024, 1024),
                  minimal_omexml=True,    # best compatibility (MIKAIA needs it)
                  overwrite=True)
```

The output must end in `.tiff`. Only `uint8`/`uint16` RGB rasters are
supported. `build_omexml(...)` produces the fuller OME-XML used with
`minimal_omexml=False`.

**Memory:** `raster2tiff` loads the whole `start_level` into RAM before
handing it to pyvips — about 3 bytes × width × height (30 GB for a
100,000 × 100,000 slide). For large slides, export from `start_level=1`
or higher, or run it on a machine with enough RAM.

# `compass.processing`

An empty package so far. It will hold the domain algorithms (tissue
detection, Macenko/Reinhard stain normalization, patch sampling, image
registration, Visium tools) as numpy/geometry in → numpy/geometry out
functions, depending only on `compass.core`. See architecture §5 for the
legacy modules that will move here.

# `compass.viewer`

```python
from compass.viewer import ViewerState, TileGrid, TileCache, TileFetcher, TileKey
from compass.viewer.canvas import SlideCanvas   # needs vispy + OpenGL
```

The package `__init__` is GUI-free: state and tile machinery work (and are
tested) without Qt or OpenGL. Only `compass.viewer.canvas` and
`compass.viewer.app` touch vispy/Qt.

## Running the viewer

```bash
uv sync --extra viewer
uv run python -m compass.viewer slide.zarr [--tile-size 512] [--cache-mb 768] [--workers 2] [-v]
```

| Option | Default | Meaning |
|---|---|---|
| `--tile-size` | 512 | Tile edge in px; keep equal to the store's `chunk_size` |
| `--cache-mb` | 768 | RAM budget for decoded tiles |
| `--workers` | 2 | Background tile-reader threads |
| `-v` | off | Debug logging to stderr |

Controls are vispy's `PanZoomCamera`: drag to pan, wheel to zoom.

## `ViewerState` — shared reactive state

A psygnal `EventedModel` that every panel reads and writes; each field has
its own signal under `state.events.<field>`.

| Field | Type | Written by |
|---|---|---|
| `viewport` | `(x0, y0, width, height)`, level-0 px | canvas |
| `canvas_size` | `(width, height)`, device px | canvas |
| `current_level` | `int` | canvas |
| `visible_groups` | `frozenset[int]` (group ids) | panels |
| `selected_ids` | `frozenset[int]` (object ids) | panels |
| `hovered_id` | `int` or `None` | panels |

`state.scale` is the derived zoom: device pixels per level-0 pixel.

```python
from compass.viewer import ViewerState

state = ViewerState()
state.events.selected_ids.connect(lambda ids: print("selected ->", sorted(ids)))

state.selected_ids = frozenset({7, 42})   # prints: selected -> [7, 42]
state.selected_ids = frozenset({7, 42})   # equal value: no signal

state.viewport = (0.0, 0.0, 2048.0, 1536.0)
state.canvas_size = (1024, 768)
state.scale                                # 0.5
```

All fields are immutable types. Change them by assigning a new value, not
by mutating in place, so change detection works.

## `TileGrid` — which level, which tiles

```python
from compass.core import ZarrRaster
from compass.viewer import TileGrid, TileKey

grid = TileGrid.from_raster(ZarrRaster("demo.zarr"), tile_size=512)
grid.grid_size(0), grid.grid_size(2)      # ((12, 8), (3, 2))  columns, rows
```

`level_for_scale(scale, bias=0.0)` picks the coarsest level that is still
at least screen resolution:

```python
[grid.level_for_scale(s) for s in (2.0, 1.0, 0.5, 0.3, 0.25, 0.1)]
# [0, 0, 1, 1, 2, 2]
```

A positive `bias` accepts slightly coarser levels (less data, softer
image).

`tiles_in_viewport(viewport, level, margin=0)` lists the tiles that cover
a viewport, **center first**, so fetching in list order fills the middle
of the screen first. `margin` adds rings of off-screen tiles for
prefetching:

```python
grid.tiles_in_viewport((0, 0, 2048, 1536), level=1)
# [TileKey(level=1, ix=0, iy=0), TileKey(level=1, ix=1, iy=0),
#  TileKey(level=1, ix=0, iy=1), TileKey(level=1, ix=1, iy=1)]
len(grid.tiles_in_viewport((0, 0, 2048, 1536), level=1, margin=1))   # 9
```

Tile geometry, at the tile's own level and in world (level-0) coordinates.
Edge tiles are clipped:

```python
k = TileKey(level=1, ix=5, iy=3)
grid.tile_bounds_px(k)       # (2560, 1536, 512, 512)
grid.tile_bounds_level0(k)   # (5120.0, 3072.0, 1024.0, 1024.0)
grid.downsample(1)           # 2.0
```

## `TileCache` — byte-budgeted LRU

Thread-safe. Evicts least-recently-used tiles once `max_bytes` is
exceeded, but never the tile just inserted.

```python
from compass.viewer import TileCache

cache = TileCache(max_bytes=768 * 2**20)
cache.put(key, array)
cache.get(key)            # array or None; refreshes recency
key in cache, len(cache), cache.nbytes
cache.clear()
```

## `TileFetcher` — background reads

A deduplicating thread pool. `on_tile` runs **on a worker thread**. The
caller must hand results back to its own (GUI) thread; `SlideCanvas`
does it with a queue drained by a timer.

```python
import threading
from compass.viewer import TileFetcher

raster = ZarrRaster("demo.zarr")
cache = TileCache(max_bytes=256 * 2**20)

def read_tile(key):
    x0, y0, w, h = grid.tile_bounds_px(key)
    return raster.get_region_px(x0, y0, w, h, level=key.level)

def on_tile(key, arr):           # worker thread
    cache.put(key, arr)

fetcher = TileFetcher(read_tile, on_tile, max_workers=2)
keys = grid.tiles_in_viewport((0, 0, 2048, 1536), level=1)
fetcher.request(keys)    # 4 newly queued
fetcher.request(keys)    # 0 — already queued or running
# on viewport change, drop queued reads that are no longer needed:
fetcher.cancel_except(new_keys)
fetcher.pending          # reads queued or running
fetcher.close()
```

Read errors are logged and don't stop the pool.

## `SlideCanvas` — the gigapixel canvas

*Needs vispy, a Qt backend and OpenGL; not run while writing this
document.*

```python
SlideCanvas(raster, state=None, tile_size=512, cache_bytes=768 * 2**20,
            prefetch_margin=1, max_workers=2, **vispy_canvas_kwargs)
```

- World coordinates are level-0 pixels, y down — the same as the
  annotation store, so overlays need no transform.
- The coarsest level is always drawn as a static backdrop, so the view is
  never blank.
- On every pan/zoom the canvas picks the level, shows cached tiles at
  once, queues missing ones center-first, cancels stale requests, and
  writes `viewport`, `canvas_size` and `current_level` into the shared
  state.
- Tiles of the previous level stay on screen until the new level covers
  the whole viewport, so zooming never shows holes.

`canvas.native` is the Qt widget; `canvas.view.scene` is where future
annotation overlays attach. `canvas.close()` stops the timer and the
fetcher.

### Embedding in your own window

How a larger app shares one `ViewerState` between the canvas and other
panels:

```python
import sys
from vispy import app as vispy_app
vispy_app.use_app("pyside6")            # before creating any canvas

from PySide6 import QtWidgets
from compass.core import ZarrRaster
from compass.viewer import ViewerState
from compass.viewer.canvas import SlideCanvas

qapp = QtWidgets.QApplication(sys.argv)
state = ViewerState()

canvas = SlideCanvas(ZarrRaster("slide.zarr"), state=state)

# a minimal "panel": a status label following the navigation state
label = QtWidgets.QLabel()
def show_nav(*_):
    x0, y0, w, h = state.viewport
    label.setText(f"level {state.current_level}  |  "
                  f"x={x0:.0f} y={y0:.0f}  {w:.0f}×{h:.0f} px")
state.events.viewport.connect(show_nav)

win = QtWidgets.QMainWindow()
win.setCentralWidget(canvas.native)
dock = QtWidgets.QDockWidget("Navigation")
dock.setWidget(label)
win.addDockWidget(QtWidgets.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
win.resize(1280, 900)
win.show()
qapp.exec()
```

# Recipes

## Choosing live vectors vs. a density raster

Before fetching geometry for a viewport, count it with the R-tree. Live
vector rendering is only for small counts; above a threshold, rasterize
(e.g. bounding-box centers with `datashader.Canvas`):

```python
LIVE_MAX = 50_000
x0, y0, w, h = state.viewport
roi = (x0, y0, x0 + w, y0 + h)

if store.count_objects_in_roi(cells, roi) <= LIVE_MAX:
    gdf = store.get_objects_in_roi(cells, roi, group_ids=list(state.visible_groups))
    ...  # draw as vispy markers/polygons
else:
    bb = store.get_bboxes(layer_id=cells)
    ...  # rasterize (min_x+max_x)/2, (min_y+max_y)/2 with datashader
```

## Coloring cells by a marker

Join the geometry and the attributes on `object_id`:

```python
gdf = store.get_objects_in_roi(cells, (0, 0, 2000, 2000))
expr = store.get_scalar_data(gdf.index.to_numpy(), attributes=["CD3"])
cd3 = (expr.to_pandas().set_index("object_id")["CD3"]
           .reindex(gdf.index).fillna(0.0))
gdf["CD3"] = cd3            # ready for a colormap
```

## Cutting out an annotated region at a given resolution

Polygons are stored in level-0 pixels. Scale them to the target level
before reading:

```python
import shapely.affinity as sha

raster = ZarrRaster("demo.zarr")
level = raster.get_level_for_mpp(1.0)                    # 2
s = raster.between_level_scaling_factor(0, level)        # 0.25

poly0 = store.get_objects([1001]).geometry.iloc[0]
poly = sha.scale(poly0, xfact=s, yfact=s, origin=(0, 0))
img = raster.get_polygonal_region_px(poly, level=level)
img.shape    # (375, 500, 3)
```

## Pixels ↔ microns

```python
mpp = raster.get_mpp_for_level(level)
area_um2 = poly.area * mpp**2              # polygon area in µm² at that level
dist_px0 = 100 / raster.native_resolution  # 100 µm in level-0 pixels (400.0)
```

# Known limitations

- **No annotation writer outside `legacy`** yet; `AnnotationStore` is
  read-only (see the SQL example above).
- **Viewer**: no annotation overlays, selection or analysis panels yet
  (build steps 4–5, architecture §7). The canvas isn't covered by automated
  tests.
- **`processing`** is empty.
- **`raster2tiff`** loads the exported level fully into RAM and assumes
  RGB.
- **`ZarrRaster`** ignores `as_type` and raises on out-of-bounds reads
  rather than padding.
- **Ingestion** reads through OpenSlide only. The Bio-Formats cross-check
  for MRXS is agreed in principle but not implemented (architecture §8).
