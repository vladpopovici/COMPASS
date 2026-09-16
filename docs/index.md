# COMPASS

A local, non-browser, single-process Qt application for viewing large
pathology whole-slide images (rasters up to ~200,000 x 200,000 px,
multi-channel, micron-calibrated) together with millions of associated
vector/raster annotations.

This page documents the **current, non-legacy** codebase — everything under
`src/compass/` except `src/compass/legacy/` (the original flat modules,
unchanged, being ported piece by piece; see their target mapping in
[`architecture.md`](architecture.md) rather than here).

For the *why* behind every design choice below (Zarr vs HDF5, no
browser/JS, OpenSlide confined to ingestion, ...), see the decision log in
[`architecture.md`](architecture.md). This page is the *what/how*: package
layout, current API, and how to run things.

## Migration status

| Package | Status |
|---|---|
| `core/` | Ported: pyramid protocol, `Magnification`, `ZarrRaster`, vectorized `AnnotationStore`. |
| `connector/` | Ported: `wsi.py` (OpenSlide reader), `ingest.py` (`wsi2zarr`), `export.py` (`raster2tiff`, OME-TIFF interchange). |
| `viewer/` | Build steps 1-3 of `architecture.md` §7: reactive state, headless tile grid/cache/fetcher, vispy pan/zoom canvas with level-switching tile streaming, `python -m compass.viewer` entry point. Annotation overlays and analytical dock panels (pyqtgraph/matplotlib) are not yet built. |
| `processing/` | Scaffolded stub package only — no algorithms ported yet. |

The vispy/Qt layer cannot be imported on a machine without `libGL` (e.g.
most CI/dev containers) — it is compile-checked only there; run it on a
machine with OpenGL via `uv sync --extra viewer`.

## Installation

Managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                    # core deps + dev group only
uv sync --extra connector  # + WSI ingestion (pyvips, OpenSlide, h5py, tifffile)
uv sync --extra processing # + domain-algorithm deps (scikit-image, opencv, ...)
uv sync --extra viewer     # + Qt/vispy (PySide6, vispy, psygnal) — needs OpenGL
```

`uv run python -c "import compass.core"` is expected to stay fast and
never pull in Qt, OpenSlide, pyvips, cv2, h5py or AnnData — that guarantee
is what lets `core` be imported from anywhere (notebook, batch script,
another package) without dragging in the heavy stuff.

## Running the tests

```bash
uv run pytest src/compass/tests -q
```

The suite covers `core` (magnification math, Zarr raster I/O, the
annotation store) and the headless parts of `viewer` (state, tile
grid/cache, fetcher) plus `connector.ingest`'s pure planning helpers. It
does not touch `compass.legacy`, and it does not exercise the vispy
canvas or the Qt entry point (both need a real OpenGL context).

---

## `compass.core`

```python
from compass.core import ZarrRaster, AnnotationStore, Magnification, write_pyramid
```

Minimal-dependency package (numpy, zarr, shapely, geopandas, polars,
pandas, pydantic, dask). Depends on nothing else in this project — every
other package depends on it. Exposes plain numpy/pandas/polars/geopandas/
dask objects only, never one Python object per annotation.

### Raster access — `PyramidalRaster` / `ZarrRaster`

`PyramidalRaster` (`pyramid.py`) is the abstract protocol every raster
source implements: a pyramid of levels (level 0 = full resolution),
level<->mpp<->objective-power conversions via a `Magnification`, and
pixel-region reads. Coordinates are always `(x, y)`, pixel units at a
specified level.

`ZarrRaster` (`zarr_raster.py`) is the concrete, read-only implementation
backed by a Zarr v3 store (sharding codec), one array per pyramid level
(`"0"`, `"1"`, ...), with `mpp`/channel-name/scale-factor metadata as
group and per-level attrs.

```python
from compass.core import ZarrRaster

raster = ZarrRaster("slide.zarr")
raster.nlevels                       # number of pyramid levels
raster.shape(0)                      # ImageShape(width=..., height=...) at level 0
raster.get_mpp_for_level(0)          # microns per pixel at level 0
raster.get_level_for_mpp(0.5)        # nearest level for a target resolution

tile = raster.get_region_px(0, 0, 512, 512, level=2)   # np.ndarray (H, W[, C])
plane = raster.get_plane(level=raster.nlevels - 1)      # whole (coarsest) level

# lazy, dask-backed variants chunked at the Zarr chunk granularity:
lazy_tile = raster.get_region_px_dask(0, 0, 512, 512, level=2)
```

`write_pyramid` writes that same on-disk layout from fully materialized
per-level arrays — meant for maps, test fixtures and small images.
Whole-slide ingestion (streaming, via pyvips) goes through
`compass.connector.wsi2zarr` instead, which targets the same layout
without ever holding a full level in memory.

```python
from compass.core import write_pyramid
import numpy as np

write_pyramid("mask.zarr",
              levels=[np.zeros((4096, 4096), dtype=np.uint8),
                      np.zeros((2048, 2048), dtype=np.uint8)],
              base_mpp=0.25, base_objective_power=40.0)
```

### Magnification

`Magnification` converts between pyramid level, microns-per-pixel and
objective power, given the base (level 0) mpp/objective and the scale
factor between levels (default 2.0). `ZarrRaster` and `WSI` both build one
internally from their stored metadata; construct one directly only when
working with an in-memory array pyramid that has no raster wrapper yet.

```python
from compass.core import Magnification

mag = Magnification(magnif=40.0, mpp=0.25, n_levels=6, magnif_step=2.0)
mag.get_level_for_mpp(1.0)     # -> 2
mag.get_mpp_for_level(2)       # -> 1.0
mag.get_magnif_for_level(2)    # -> 10.0
```

### Annotations — `AnnotationStore`

Read-oriented, vectorized access to a SQLite/R-tree annotation database
(layer -> group -> object hierarchy; WKB geometry; sparse per-object
scalar attributes such as per-cell gene expression). The physical schema
is unchanged from the legacy store; what changed is the access layer —
every bulk query returns a **geopandas GeoDataFrame** or **polars
DataFrame**, decoded in one vectorized pass, never a list of per-object
Python instances.

```python
from compass.core import AnnotationStore

with AnnotationStore("slide.ann.sqlite") as store:
    layer_id = store.get_layer_id("cells")

    # cheap, R-tree-only counts — use to decide live-vector vs. datashader LOD
    n = store.count_objects_in_roi(layer_id, roi=(x0, y0, x1, y1))

    # the viewport query: geometry + type + name for everything in the ROI
    gdf = store.get_objects_in_roi(layer_id, roi=(x0, y0, x1, y1))

    # sparse per-object scalars (e.g. marker expression), pivoted wide
    expr = store.get_scalar_data(gdf.index.to_numpy(), attributes=["CD3", "CD8"])
```

Other entry points: `get_layers`/`get_groups` (hierarchy), `get_bboxes`
(no geometry decoding, for coarse rendering/hit-testing),
`get_memberships` (object<->group table), `count_objects_per_group`.
`init_db(path)` creates a fresh, empty database with this schema.

---

## `compass.connector`

```python
from compass.connector import wsi2zarr, raster2tiff
```

Ingestion and interchange tooling: OpenSlide-backed WSI reading
(`wsi.py`), one-shot pyramid baking into the canonical `core` Zarr format
(`ingest.py`), and OME-TIFF/BigTIFF export for other pathology tools
(`export.py`). Heavy/volatile dependencies (pyvips, OpenSlide) are
isolated here and are never imported by `viewer` or `processing`; import
`compass.connector` itself needs neither system library since both are
imported lazily where used.

**`WSI` is ingestion-only** — never use it as a runtime `PyramidalRaster`
in the viewer. OpenSlide and Bio-Formats disagree on MRXS coordinate
interpretation, so OpenSlide-read pixels are only trusted after the
converted Zarr pyramid has been checked against known landmark
annotations.

```python
from compass.connector import wsi2zarr

wsi2zarr("slide.mrxs", "slide.zarr",
         crop=True,            # auto-crop to the scanner's ROI metadata
         downscale_factor=2,
         chunk_size=512,       # match the viewer's tile size
         shard_factor=8)
```

Internally: level 0 is the (optionally cropped) base image; every further
level is generated by downscaling the previous one with pyvips through
temporary `.v` files (kept next to the destination, low RAM pressure);
data is copied into the Zarr store in horizontal bands aligned to whole
shards. `plan_levels`/`resolve_crop` are the pure, dependency-free
planning helpers (level extents, crop-region resolution) — they're what
`src/compass/tests/test_connector_ingest.py` exercises without needing
pyvips/OpenSlide installed.

```python
from compass.connector import raster2tiff
from compass.core import ZarrRaster

raster2tiff(ZarrRaster("slide.zarr"), "slide.tiff", minimal_omexml=True)
```

Zarr stays the internal format; OME-TIFF/BigTIFF is produced on demand
for interchange with QuPath, ASAP, MIKAIA and similar tools.

---

## `compass.processing`

Scaffolded stub — not yet implemented. Will hold domain algorithms
(tissue detection, stain normalization, patch sampling, image
registration, Visium spatial-transcriptomics tools) as pure
numpy/geometry in, numpy/geometry out functions, depending only on
`compass.core`. See the legacy-module mapping in `architecture.md` §5 for
what's expected to land here.

---

## `compass.viewer`

```python
from compass.viewer import ViewerState, TileGrid, TileCache, TileFetcher
```

Qt viewer built on vispy (gigapixel canvas), psygnal (reactive state) and
— eventually — pyqtgraph/matplotlib (analytical dock panels). The package
`__init__` itself stays GUI-free (`ViewerState` and the tile machinery
need no OpenGL/Qt and are unit-tested headless); the vispy canvas and Qt
entry point live only in the `canvas`/`app` submodules.

### Running it

```bash
uv run python -m compass.viewer slide.zarr [--tile-size 512] [--cache-mb 768] [--workers 2] [-v]
```

Opens one `QMainWindow` with a `SlideCanvas` as its central widget. Needs
`uv sync --extra viewer` and a machine with working OpenGL.

### Reactive state — `ViewerState`

A single `psygnal.EventedModel` instance is the source of truth every
panel subscribes to: the canvas writes `viewport`/`canvas_size`/
`current_level` as the user navigates; other panels are expected to write
`visible_groups`/`selected_ids`/`hovered_id`. All fields are immutable
(tuples/frozensets) — mutate by assignment, not in place, so psygnal's
equality-based change detection fires signals correctly.

```python
from compass.viewer import ViewerState

state = ViewerState()
state.events.viewport.connect(lambda: print(state.viewport))
state.visible_groups = frozenset({1, 2})   # triggers the signal
```

### Tile geometry and cache — `TileGrid` / `TileCache`

Pure logic, no GUI dependency, so it's exactly what
`src/compass/tests/test_viewer_tiling.py` exercises directly. `TileGrid`
answers, for a given zoom factor, which pyramid level to render
(`level_for_scale`) and which tiles a viewport needs
(`tiles_in_viewport`, center-first so prefetch fills the middle of the
screen first). `TileCache` is a byte-budgeted, thread-safe LRU of decoded
tile arrays.

```python
from compass.viewer import TileGrid, TileCache
from compass.core import ZarrRaster

raster = ZarrRaster("slide.zarr")
grid = TileGrid.from_raster(raster, tile_size=512)
level = grid.level_for_scale(scale=1.0)          # device px per level-0 px
keys = grid.tiles_in_viewport((0, 0, 2048, 2048), level)

cache = TileCache(max_bytes=768 * 2**20)
```

### Background fetching — `TileFetcher`

Deduplicating thread pool that reads tiles off the GUI thread — Zarr
shard reads are independent byte-range reads with no global lock (unlike
HDF5), so this genuinely parallelizes. `request(keys)` queues reads
(skipping ones already in flight); `cancel_except(keep)` drops stale
prefetches on viewport change; `on_tile` fires on a worker thread, so the
caller (the vispy canvas) is responsible for marshalling results back to
its own thread — it does so via a queue drained on a vispy timer.

### The canvas — `compass.viewer.canvas.SlideCanvas`

```python
from vispy import app as vispy_app
vispy_app.use_app("pyside6")   # select a backend before instantiating

from compass.viewer.canvas import SlideCanvas
from compass.core import ZarrRaster

canvas = SlideCanvas(ZarrRaster("slide.zarr"), tile_size=512)
canvas.native   # the Qt widget to embed (e.g. as a QMainWindow's central widget)
```

Import-only is safe without OpenGL; instantiating needs it. World
coordinates are level-0 pixels, y-down, shared with the annotation store.
A static coarsest-level backdrop is always present so no viewport is ever
blank; on every camera change the wanted tile set for the current zoom's
level is computed, cached tiles become `Image` visuals immediately, and
missing ones are fetched in the background and swapped in as they arrive.
`canvas.view` is the vispy `ViewBox` — future annotation overlays attach
under `canvas.view.scene` in the same level-0 coordinate system.

---

## Package dependency direction

```
core          <- everything (minimal deps; safe to import from anywhere)
connector     -> core            (never imported by viewer or processing)
processing    -> core            (independent of connector)
viewer        -> core, processing (never connector)
```
