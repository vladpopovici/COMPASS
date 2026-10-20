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
decision log in [`architecture.md`](architecture.md). Two parts have their
own guides: annotations in [`annotations.md`](annotations.md) and the
viewer in [`viewer.md`](viewer.md).

Unless marked otherwise, every example below was run against the code while
writing this document; output blocks are real output. Examples that need
OpenGL, libvips or OpenSlide are marked as such.

# Overview

## Packages and status

| Package | Purpose | Status |
|---|---|---|
| `compass.core` | `PyramidalRaster` protocol, Zarr raster backend, magnification math, annotations (object classes, vectorized transforms, SQLite/R-tree store with read and write, legacy API) | Ported, tested. Annotation guide: [`annotations.md`](annotations.md) |
| `compass.connector` | WSI ingestion (OpenSlide → pyvips → Zarr), OME-TIFF export | Ported; ingestion tested against stand-ins (no libvips/OpenSlide needed) |
| `compass.processing` | Domain algorithms (tissue detection, stain normalization, sampling, registration, Visium) | Empty stub |
| `compass.viewer` | psygnal state, tile grid/cache/fetcher, vispy tile-streaming canvas, keyboard/mouse navigation, overview thumbnail, `python -m compass.viewer` | Steps 1–3 of the build order (architecture §7). No annotation overlays or analysis panels yet. Full guide: [`viewer.md`](viewer.md) |

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

The suite covers `core` (magnification math, Zarr raster I/O,
annotations: objects, transforms, store reads and writes, the legacy API —
plus file exchange with `compass.legacy`, run only when its dependencies
are installed), the headless parts of `viewer` (state, tile grid/cache, fetcher),
and `connector.ingest` (planning helpers, the full `wsi2zarr` pipeline
against numpy stand-ins for pyvips and OpenSlide, and `ometiff2zarr`
end-to-end on real OME-TIFFs written by tifffile — which is in the dev
group, so these run under a plain `uv sync`). It doesn't exercise the
vispy canvas or the Qt entry point.

# Quick start

The typical path is: bake a vendor slide into Zarr once, then read it (and
its annotations) from Python or the viewer.

```python
# 1. one-time ingestion (needs `uv sync --extra connector` + libvips)
from compass.connector import wsi2zarr

wsi2zarr("slide.mrxs", "slide.zarr", crop=True)

# ...or, for an OME-TIFF (pure Python, no libvips needed)
from compass.connector import ometiff2zarr

ometiff2zarr("slide.ome.tif", "slide.zarr")
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
Written by `core.write_pyramid` (in-memory arrays),
`connector.wsi2zarr` (streamed from a vendor slide) and
`connector.ometiff2zarr` (streamed from an OME-TIFF); read by
`core.ZarrRaster`. All three create arrays and metadata through
`core.create_level_array` / `core.write_pyramid_attrs`, so the layout
below is defined in one place.

| Item | Value |
|---|---|
| Array shape | `(height, width, channels)` or `(height, width)` (the ingestion tools always write the 3-D form, single-channel as `(h, w, 1)`) |
| Dtype | as ingested: `uint8` for vendor slides, the source dtype (e.g. `uint16`) for OME-TIFF |
| Chunks | `chunk_size × chunk_size` (default 512 — the viewer's tile size), clipped on levels smaller than one chunk |
| Shards | `shard_factor × shard_factor` chunks (default 8 → 4096 × 4096 px) |
| Codec | Blosc zstd, level 3, byte shuffle |
| Group attrs | `compass_version` (= 2), `level_count`, `base_mpp`, `base_mag_step`, `base_objective_power`, `channel_names`, `dimension_names` |
| Per-level attrs | `mpp_x`, `mpp_y`, `objective_power`, `scale_factor` |

Format version 1 was the legacy HDF5 layout.

## Annotation database (SQLite + R-tree)

One SQLite file per slide: WKB geometries with bounding boxes, an R-tree
kept in sync by triggers, a layer → group → object hierarchy and sparse
float32 scalars per object. The schema is the legacy writer's, unchanged,
so legacy and current files are interchangeable. Tables, type codes and
circle encoding: [`annotations.md`](annotations.md), *File format*.

# `compass.core`

```python
from compass.core import (ZarrRaster, write_pyramid, PyramidalRaster,
                          Magnification, AnnotationStore, Annotation,
                          Point, PointSet, PolyLine, Polygon, Circle,
                          init_db, ImageShape, Px, rasterize_polygon,
                          create_level_array, write_pyramid_attrs)
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
`connector.wsi2zarr` / `connector.ometiff2zarr` for whole slides.

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

## Streaming writers: `create_level_array`, `write_pyramid_attrs`

The two building blocks under `write_pyramid`, for writers that fill a
level piece by piece instead of from one in-memory array (the
`connector` ingestion functions use them):

```python
root = zarr.create_group("big.zarr")
arr = create_level_array(root, 0, (h, w, 3), "uint8",
                         chunk_size=512, shard_factor=8)   # empty array "0"
... # fill arr, ideally one whole shard per write (arr.shards)
write_pyramid_attrs(root, nlevels, mpp_x=0.25, mpp_y=0.25,
                    objective_power=40.0, mag_step=2,
                    channel_names=["R", "G", "B"])
```

`create_level_array` applies the canonical chunking (clipped on small
levels), sharding and codec; channels are never split across chunks.
`write_pyramid_attrs` writes the group and per-level attributes once all
level arrays exist; `base_mpp` is the mean of `mpp_x` and `mpp_y`.

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

## Annotations: `compass.core.annot`

Annotation objects, transforms and the annotation store. Everything is in
[`annotations.md`](annotations.md); in short:

- **Objects** — `Point`, `PointSet`, `PolyLine`, `Polygon`, `Circle`: build
  and inspect single objects, with `translate`, `scale`, `resize`,
  `affine`.
- **Frames** — `compass.core.annot.transform` applies the same transforms
  to whole GeoDataFrames in one numpy pass.
- **Store** — `AnnotationStore(path, mode="r" | "r+" | "w")`: spatial and
  hierarchy queries returning GeoDataFrames / polars frames; adding,
  updating and removing objects, groups, layers and scalars; whole-store
  transforms (including an import record's matrix); in-memory stores.
- **Legacy API** — `Annotation`, `save_annotation`, `load_annotation`,
  `query_*`, `fetch_*`, on top of the store.

```python
from compass.core import AnnotationStore, Circle

with AnnotationStore("slide.ann.sqlite", mode="w") as store:
    ids = store.add_objects(cells_gdf, layer="cells", group="tumor", data=markers)
    store.add_object(Circle((4000, 4000), 150.0), layer="regions", group="spots")

with AnnotationStore("slide.ann.sqlite") as store:          # read-only
    gdf = store.get_objects_in_roi(store.get_layer_id("cells"), (0, 0, 2000, 2000))
```

# `compass.connector`

```python
from compass.connector import (wsi2zarr, ometiff2zarr, raster2tiff,
                               build_omexml, plan_levels, resolve_crop,
                               box_downsample)
from compass.connector.wsi import WSI   # imports OpenSlide
```

One-shot ingestion and interchange tools. Never imported by the viewer or
by `processing`. Needs `uv sync --extra connector`; `wsi2zarr` and
`raster2tiff` also need the libvips system library, while `ometiff2zarr`
only needs tifffile (pure Python). `plan_levels`, `resolve_crop` and
`box_downsample` are pure Python.

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
         chunk_size=512, shard_factor=8, tmp_dir=None, overwrite=False,
         start_level=0, provenance=None) -> dict
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
| `start_level` | Start from this level of the planned pyramid; see [Partial pyramids and the import record](#partial-pyramids-and-the-import-record) |
| `provenance` | Extra fields merged into the import record |

Returns the import record, which is also stored in the group's `import`
attribute.

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

## `ometiff2zarr` — bake an OME-TIFF into the Zarr pyramid

```python
ometiff2zarr(src_path, dst_path, series=None, z=None, t=None, crop=False,
             mpp=None, objective_power=None, channel_names=None,
             downscale_factor=2, min_size=256, chunk_size=512,
             shard_factor=8, overwrite=False, start_level=0,
             provenance=None) -> dict
```

For OME-TIFF files: exports from other tools (QuPath, Bio-Formats
`bfconvert`, `raster2tiff`), and multiplex/fluorescence images with many
`uint16` channels. Reads through tifffile — no pyvips, no OpenSlide.

| Parameter | Meaning |
|---|---|
| `series` | Image series to convert; default: the one with the largest full-resolution plane (skips label/macro images) |
| `z`, `t` | Plane to take along Z/T. Required when that axis has more than one plane; a stack is never reduced silently |
| `crop` | `False`: whole image. `(x0, y0, w, h)`: explicit level-0 region, clamped. OME-TIFF has no scan ROI, so `True` behaves like `False` |
| `mpp` | Microns per pixel at level 0, one value or `(mpp_x, mpp_y)`. Overrides the file's `PhysicalSizeX/Y`; **required** if the file has none |
| `objective_power` | Overrides the file's `NominalMagnification`; **required** if the file has none |
| `channel_names` | Overrides the OME channel names |
| `downscale_factor` | Integer scale between consecutive levels (≥ 2) |
| `min_size`, `chunk_size`, `shard_factor`, `overwrite`, `start_level`, `provenance` | As for `wsi2zarr` (with a start level, the base level is box-averaged from full resolution while copying) |

```python
from compass.connector import ometiff2zarr

# brightfield RGB export, calibrated
ometiff2zarr("slide.ome.tif", "slide.zarr")

# multiplex immunofluorescence, uint16, no magnification in the file
ometiff2zarr("mif.ome.tiff", "mif.zarr", objective_power=20.0)

# a z-stack: pick the plane explicitly
ometiff2zarr("stack.ome.tif", "stack.zarr", z=3, mpp=0.325, objective_power=40.0)
```

**Metadata.** Read from the OME-XML of the chosen image:

- *mpp*: `Pixels/@PhysicalSizeX|Y`, converted from `PhysicalSizeXUnit`
  (pm, Å, nm, µm, mm, cm, m; µm if absent). If only one axis is given it
  is used for both.
- *objective power*: the `Objective` referenced by the image's
  `ObjectiveSettings`; failing that, the only objective of the image's
  instrument (as in `raster2tiff`'s minimal OME-XML); failing that, the
  only objective in the file.
- *channel names*: `Channel/@Name`, used if every channel has one;
  otherwise `R, G, B` for interleaved RGB and `C0 … Cn` for the rest.

Missing calibration is an **error**, not a default: a wrong mpp silently
misaligns every annotation in microns. Pass `mpp=` / `objective_power=`
explicitly when the file lacks them.

**Axes and dtype.** Any layout tifffile reports — interleaved `YXS` (RGB),
planar `CYX`, `TCZYX`, … — is normalized to `(height, width, channels)`;
single-channel images are stored as `(h, w, 1)`. The dtype is kept
(`uint16` stays `uint16`). Channels are not split across chunks, so a
chunk of a 40-channel `uint16` image is 40 × 512 × 512 × 2 bytes (20 MB)
uncompressed.

**How it works.** As with `wsi2zarr`, the pyramid is regenerated rather
than copied from the file's own levels. Level 0 is copied one whole shard
at a time from tifffile's lazy Zarr view of the TIFF tiles. Each further
level is built from the previous **Zarr** level: an output shard maps onto
a `downscale_factor`-times larger, shard-aligned block above, which is
box-averaged (mean over `f × f` pixels, integers rounded). There are no
temporary files and no `tmp_dir`; peak memory is about
`downscale_factor²` shards (~200 MB at the defaults for RGB `uint8`).
Box averaging differs slightly from the pyvips resampling `wsi2zarr`
uses, so lower levels of the two paths are not bit-identical.

`box_downsample(block, factor)` is the filter on its own:

```python
box_downsample(np.array([[0, 1], [1, 1]], np.uint8), 2)   # [[1]]  (0.75 rounds up)
```

Series are matched to OME `Image` elements by index (tifffile builds one
series per OME image, in order); a file mixing OME and non-OME series may
pair metadata with the wrong image.

## Partial pyramids and the import record

**Starting level.** `start_level=k` imports from level *k* of the pyramid
that would otherwise be built (the plan `wsi_import.py --info` prints).
That level becomes level 0, so the store has *k* fewer levels, and its
calibration is scaled to match: `mpp × factor^k`,
`objective_power ÷ factor^k`. The crop is still given in source level-0
coordinates. Its origin is snapped down to the start level's pixel grid,
so the source → output mapping is exact.

- `wsi2zarr` reads the base image from the coarsest **stored** slide level
  whose downsample divides `factor^k` and only resizes the remainder. On a
  scanner with ×2 levels, importing from level 3 never reads full
  resolution: a 57,344 × 50,944 NDPI imports from level 3 in about 4 s,
  pixel-identical to OpenSlide's level 3.
- `ometiff2zarr` box-averages from full resolution while copying, in
  bounded sub-reads.

```python
from compass.connector import closest_level_for_mpp, plan_import, plan_levels

plan = plan_import(1500, 1200, crop=(9, 17, 1400, 1100), start_level=1)
plan["downsample"], plan["crop"], plan["levels"]
# (2, (8, 16, 1400, 1100), [{'w': 700, 'h': 550}, {'w': 350, 'h': 275}])

# level of the plan closest to 1 um/px (by ratio; ties -> finer)
closest_level_for_mpp(0.25, 1.0, nlevels=len(plan["planned"]))   # 2
```

**Import record.** Both converters return a JSON-serializable record and
store it in the group's `import` attribute
(`ZarrRaster(dst).info["import"]`):

| Key | Content |
|---|---|
| `record_version`, `timestamp` | Layout version (1); ISO-8601 time with time zone |
| `source` | `path` (absolute), `name`, `format` (`wsi`/`ometiff`), `width`, `height`, `mpp_x`, `mpp_y`, `objective_power`; OME-TIFF adds `series`, `z`, `t`, `dtype`, `channel_names` |
| `crop` | `mode` (`none`, `roi`, `explicit`, `detect`), effective `x0`, `y0`, `width`, `height` at **source level 0** (the full extent if not cropped), and the `requested` crop before snapping |
| `start` | `level`, `downsample`, `mpp_x`, `mpp_y` of the start level (+ `requested_level` / `requested_mpp` from the CLI) |
| `output` | `path`, `format_version`, `levels`, `width`, `height`, `mpp_x`, `mpp_y`, `objective_power`, `downscale_factor`, `chunk_size`, `shard_factor`, `level_shapes` |
| `transform` | Source level-0 → output level-0 pixels (below) |
| `tool`, `dry_run` | Added by `wsi_import.py` |

`transform.type` is one of the controlled vocabulary
`compass.connector.TRANSFORM_TYPES` = `shift`, `rotate`, `scale`,
`affine`: the most specific type that describes the mapping.
`x_out = (x_src - crop.x0) / start.downsample` (same for y) is a `shift`
(`dx`, `dy`) without a start level, a `scale` (`sx`, `sy`) without a crop
offset, and `affine` otherwise. `matrix` (2 × 3, row-major, applied to
`(x, y, 1)`) is always present, so a consumer can ignore `type`:

```python
import numpy as np
from compass.core import ZarrRaster

t = ZarrRaster("slide.zarr").info["import"]["transform"]
M = np.asarray(t["matrix"])
xy_out = (M @ np.c_[xy_src, np.ones(len(xy_src))].T).T   # (N, 2) points
```

**Command line.** `src/tools/wsi_import.py` exposes both:

```bash
# from planned level 2, or from the level closest to 1 um/px
uv run python src/tools/wsi_import.py slide.ndpi --start-level 2
uv run python src/tools/wsi_import.py slide.ndpi --start-mpp 1.0 --crop=detect

# print the record (stdout has only the record; messages go to stderr)
uv run python src/tools/wsi_import.py slide.ndpi --start-mpp 1.0 --show-meta > rec.json

# dry run: the record of what would be imported, as shell variables
eval "$(uv run python src/tools/wsi_import.py slide.ndpi --start-mpp 1.0 \
        --info --show-meta --meta-format kv)"
echo "$CROP_X0 $CROP_Y0 $START_DOWNSAMPLE $TRANSFORM_TYPE $TRANSFORM_MATRIX"
```

`--meta-format kv` prints one shell-quoted `KEY=value` per line. Keys are
the JSON paths, upper-cased and joined by `_` (`OUTPUT_MPP_X`,
`CROP_REQUESTED_X0`); lists are space-separated (`TRANSFORM_MATRIX='0.125
0.0 -82.0 0.0 0.125 -396.0'`) and null is empty. An out-of-range
`--start-level` stops before any work, listing the planned levels with
their µm/px.

## Planning helpers

The pure functions `wsi2zarr` and `ometiff2zarr` use to decide what it will do — useful to
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

The package `__init__` is GUI-free: state, tile machinery and navigation
maths work (and are tested) without Qt or OpenGL. Only
`compass.viewer.canvas`, `compass.viewer.inset` and `compass.viewer.app`
touch vispy/Qt.

This section is the API reference for the building blocks. The user guide
(every key and mouse binding, the info bar, the thumbnail) and the
internals (rendering pipeline, navigation model, testing, vispy pitfalls)
are in [`viewer.md`](viewer.md).

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

Drag to pan, wheel to zoom, `?` for all keys and mouse actions. The
full list is in [`viewer.md`](viewer.md).

## `ViewerState` — shared reactive state

A psygnal `EventedModel` that every panel reads and writes; each field has
its own signal under `state.events.<field>`.

| Field | Type | Written by |
|---|---|---|
| `viewport` | `(x0, y0, width, height)`, level-0 px | canvas |
| `canvas_size` | `(width, height)`, device px | canvas |
| `current_level` | `int` | canvas |
| `cursor` | `(x, y)` level-0 px, or `None` | canvas |
| `cursor_value` | `tuple[float, ...]` (one per channel), or `None` | canvas |
| `cursor_value_level` | `int` or `None` | canvas |
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

*Needs vispy, a Qt backend and OpenGL.* Navigation methods
(`get_view`/`set_view`, `go_to`, `navigate`, `overview_visible`) are
documented in [`viewer.md`](viewer.md), *Embedding `SlideCanvas`*.

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

- **Annotation writes** are bounded by SQLite's R-tree upkeep: adding or
  transforming a million objects takes tens of seconds
  ([`annotations.md`](annotations.md), *Performance*).
- **Viewer**: no annotation overlays, selection or analysis panels yet
  (build steps 4–5, architecture §7). The canvas isn't covered by automated
  tests.
- **`processing`** is empty.
- **`raster2tiff`** loads the exported level fully into RAM and assumes
  RGB.
- **`ZarrRaster`** ignores `as_type` and raises on out-of-bounds reads
  rather than padding.
- **Vendor-slide ingestion** (`wsi2zarr`) reads through OpenSlide only.
  The Bio-Formats cross-check for MRXS is agreed in principle but not
  implemented (architecture §8). OME-TIFF files go through tifffile
  (`ometiff2zarr`) instead.
- **`ometiff2zarr`** doesn't reuse the file's own pyramid levels (always
  rebuilt) and keeps all channels in one chunk; both may need revisiting
  for large multiplex images. Not yet run on a real scanner/multiplex file.
