# COMPASS Architecture

This document is the decision log for the COMPASS pathology viewer. It
records what was decided, why, and what alternatives were rejected, so
future sessions (human or Claude) don't relitigate settled questions
without new information.

## 1. Problem statement

Build a local, non-browser, Python-first desktop application for viewing
pathology whole-slide images and their annotations.

- **Core data object**: a large raster image, multi-channel, up to roughly
  200,000 x 200,000 px, with a native resolution in microns-per-pixel (mpp).
- **Annotations**, produced elsewhere (not authored in-app), of two kinds:
  - **Vector objects** in the image's pixel or micron coordinate system:
    points, polygons/contours, with attached attributes (e.g. cells with a
    molecular expression profile). Potentially millions of objects.
  - **Maps**: single-channel rasters, pixel-aligned 1:1 with the base image
    (e.g. a morphological region segmentation).
- The user selects individual objects for inspection and toggles
  annotation **groups/planes** on/off, but does not create annotations.
- The viewer widget must be embeddable in a larger UI app, which will grow
  to include data-summary panels (clustering, PCA, histograms).

## 2. Raster storage: Zarr (v3, sharding), not HDF5

**Decision**: pyramidal image and map rasters are stored as Zarr v3 arrays
using the sharding codec, one array group per pyramid level (analogous to
OME-Zarr's `0/`, `1/`, `2/` convention), with `mpp`, channel names, and
label lookup tables as group/array attributes.

**Why not HDF5** (the legacy code's format): HDF5's C library, when built
thread-safe, serializes all library calls behind a single global lock —
independent of chunk layout or file count, concurrent tile reads from
multiple threads end up serialized. This directly undermines a tiled
viewer's prefetch strategy. Zarr has no equivalent global lock: each
chunk/shard read is an independent file/byte-range read, so concurrent
reads are trivially parallel.

**Why not plain (unsharded) Zarr**: at this scale, one-file-per-chunk
across many pyramid levels produces millions of small files —
filesystem/inode overhead, slow directory listings, awkward to move or
back up.

**Why not Icechunk**: solves a different problem (many writers, evolving
datasets, versioning, cloud object storage). This workload is write-once
(at ingestion), read-many, local-only — the transactional/versioning
machinery isn't needed and would be a dependency taken on for nothing.

**Practical notes**:
- Use `hdf5plugin`-equivalent fast codecs (blosc/zstd) rather than gzip —
  decompression speed matters more than ratio here.
- Chunk shape should match the tile-fetch granularity used by the viewer
  (e.g. 512x512) so a tile request maps to whole chunks.
- For genuine interchange with other pathology tools (QuPath, ASAP,
  clinical viewers), export to OME-TIFF/BigTIFF on demand — Zarr is the
  *internal* format, not required to be the interchange format.

### Legacy code mapping

- `compass/_pyr.py` (`PyramidalImage` ABC): the level/magnification/mpp
  conversion logic and the abstract `get_region_px` interface are sound
  and map directly onto the target `core.PyramidalRaster` protocol. Port
  largely unchanged.
- `compass/_magnif.py` (`Magnification`): keep unchanged — clean
  self-contained level<->mpp<->objective-power conversion utility.
- `compass/_mri.py` (`MRI`, HDF5-backed `PyramidalImage`): reference
  implementation for the *access pattern* (per-level image, `get_region_px`,
  `get_plane`, dask-backed variants). Reimplement against Zarr instead of
  `h5py`; keep the same public shape.
- `compass/core.py: wsi2hdf5`: the pyramid-generation algorithm (crop to
  ROI, iterative downsample via pyvips, write per-level with `mpp`/
  `scale_factor` attrs) is sound. Retarget the write step to Zarr v3 +
  sharding. Keep in `connector/` (depends on `pyvips`).
- `compass/core.py` top-level `logging.basicConfig(filename=".compass-core.log")`:
  runs at import time, writes to a relative path. Fix when this module is
  touched — configure logging explicitly per entry point instead.

## 3. WSI ingestion: isolated from the runtime viewer

**Decision**: OpenSlide-based (and libvips, which wraps OpenSlide for these
formats) reading is ingestion-only. It lives in `connector/`, runs once per
slide to bake a canonical Zarr pyramid, and is never imported by the
runtime viewer.

**Why**: OpenSlide and Bio-Formats disagree on MRXS coordinate
interpretation (non-empty scan-region offset handling), which silently
shifts coordinates — exactly the kind of bug that surfaces later as
"annotations are subtly misaligned" and is hard to trace back. libvips'
slide loader is built on OpenSlide for these formats, so it is *not* an
independent code path for cross-checking.

**Practical approach**: for formats with known ambiguity, cross-check
against Bio-Formats (JVM-based, via a one-time ingestion tool — does not
need to be a runtime dependency, does not need to live in the same Python
environment as the viewer). Verify a handful of known landmark annotations
align correctly on the converted raster before trusting a new ingestion
pipeline for a given scanner/format combination.

### Legacy code mapping

- `compass/_wsi.py` (`WSI`, OpenSlide-backed `PyramidalImage`): keep the
  metadata-extraction logic (objective power, mpp, ROI bounds, background
  color) — it's a reasonable reference for what OpenSlide exposes — but
  move to `connector/`, used only during ingestion, never as a runtime
  `PyramidalRaster` implementation.

## 4. Annotation storage: SQLite + R-tree, vectorized access

**Decision**: keep the SQLite schema already implemented in
`compass/annot_sqlite_storage.py` as the physical annotation store —
WKB geometry, an R-tree virtual table with insert/delete/bbox-update
triggers for spatial ROI queries, and a layer -> group -> object hierarchy.
**Replace** the in-memory access layer (`compass/annot.py`'s
`AnnotationObject`/`Annotation` classes, one Python object + one shapely
geometry per annotation) with a vectorized query layer returning
**geopandas GeoDataFrames** (for geometry + a handful of dense per-object
attributes) directly from SQL, rather than materializing one Python
instance per row.

**Why keep the SQLite/R-tree layer**: it already does the right thing —
`query_object_ids_in_layer_roi` is exactly the spatial-index-backed
viewport query the viewer needs, and it already existed before this
project's design discussion started. No reason to replace working,
appropriately-scaled infrastructure.

**Why replace the Python object model above it**: `annot.py`'s
`Annotation.asdict()` / `load_annotation()` fully materializes *all*
objects as individual `AnnotationObject` + shapely-geometry instances.
At millions of objects this is worse overhead than the AnnData-style
containers this project deliberately avoided elsewhere — the fix is a
thin layer that reads WKB rows in bulk and constructs a
`geopandas.GeoDataFrame` in one vectorized pass (e.g. via
`shapely.from_wkb` over a numpy array of blobs), not the current
row-by-row Python-object loop.

**Layer/group model maps directly onto product requirement**: the
"toggle annotation groups/planes on/off" requirement from the original
brief is already a first-class part of this schema (layers contain
groups; objects belong to one or more groups within a layer) — no new
data-modeling needed for that feature.

### Sparse per-object attributes (e.g. molecular expression)

**Decision**: keep the existing sparse packing (`object_scalar_sparse`
table: parallel `uint32` attribute-id array + `float32` value array per
object, with a shared `scalar_attributes` name<->id dictionary). This is
already the right representation for e.g. Visium-style sparse gene
expression (~20,000 genes total, ~3,000-4,000 detected per spot/cell) —
functionally equivalent to the long/COO-format Parquet table considered
during design, just stored per-object rather than as a separate table.
Revisit only if profiling shows a need for the alternative access pattern
(bulk per-gene queries across all objects, which favors a
CSR/CSC-materialized cache) — not needed until that's demonstrated.

### Legacy code mapping

- `compass/annot_sqlite_storage.py`: keep, schema and query functions
  largely as-is. This becomes the core of `core`'s annotation data-access
  module.
- `compass/annot.py`: superseded as the *bulk* access path. The geometry
  classes (`Point`, `PointSet`, `PolyLine`, `Polygon`, `Circle`) may still
  be useful as single-object convenience wrappers for interactive
  create/edit workflows if those are ever added, but bulk read/write for
  the viewer should go through the new vectorized layer, not through
  `Annotation.asdict()`/`fromdict()`.
- `compass/sample.py` (`SampleManager`): imports `from wsitk_annot import
  Annotation` — a different package name, not `compass.annot`. Stale,
  likely predates a package rename or an abandoned refactor. The
  underlying concept (a per-sample `.cp` folder holding pyramid files +
  an annotation registry) is reasonable and worth keeping if a
  per-sample project-folder layout is still wanted — fix the import once
  that's confirmed, don't fix it blind.

## 5. Package split

```
core/         PyramidalRaster protocol + Zarr backend.
              Annotation data-access layer (SQLite/R-tree -> geopandas/polars).
              Deps: numpy, zarr, shapely, geopandas, polars, pydantic.
              Depends on nothing else in this project. Everything else
              depends on this.

connector/    AnnData / SpatialData conversion tools (isolated: these
              packages are opinionated, pin older numpy, and conflict
              with other deps — never imported outside this package).
              WSI ingestion: OpenSlide/Bio-Formats reading, pyvips-based
              Zarr pyramid writing.
              Depends on: core (writes into its formats).
              Never imported by viewer/ or processing/.

processing/   Domain algorithms, numpy/geometry in -> numpy/geometry out:
              tissue detection, stain normalization (Macenko/Reinhard),
              patch sampling (sliding window / random / tiler),
              image registration, Visium spot detection/mapping.
              Depends on: core (reads via raster/annotation protocols).
              Independent of connector/ — doesn't know about ingestion
              formats, doesn't know about the UI.

viewer/       vispy/Qt gigapixel canvas, psygnal reactive state,
              pyqtgraph (live-interactive panels), matplotlib+seaborn
              (via FigureCanvasQTAgg, refined statistical plots).
              Depends on: core (reads data), processing (optionally,
              to trigger on-demand analysis from the UI).
```

Rationale for the split: dependency direction, not just topic. `core` is
minimal and stable so it's safe to import from anywhere (a Claude Code
session, a notebook, the viewer, a batch script) without dragging in Qt,
OpenSlide, or AnnData. `connector` and `processing` are peers, not
layered. `viewer` sits on top and may call into `processing`, never the
reverse.

### Legacy code mapping (orthogonal, move ~as-is into `processing/`)

`compass/tissue.py`, `compass/stain.py`, `compass/sampler.py`,
`compass/imreg.py`, `compass/sptx/visium_v1.py` (and
`tools/visium_v1/*`) — domain image-processing utilities, no viewer or
storage-format coupling. Confirm none of these are meant to drive
interactive overlays before assuming a clean move (e.g. is tissue
detection used to produce a live mask overlay in the viewer? — if so
that's a `viewer` <-> `processing` integration point worth designing
explicitly, not just a code move).

## 6. UI stack: single-toolkit Qt, no browser/JS

**Decision**: vispy (custom-built Qt canvas, not napari) for the
gigapixel image+annotation view; `psygnal` (`EventedModel`) for shared
reactive state across widgets; pyqtgraph for live-interactive analytical
panels (scatter/histogram/heatmap the user brushes or drags on directly);
matplotlib+seaborn (embedded via `FigureCanvasQTAgg`) for refined
statistical plots (PCA, clustering, anything wanting seaborn's aesthetic
breadth) where redraw-on-selection-change is the right interaction model
anyway. One process, one Qt `QMainWindow` hosting all panels as dock
widgets.

**Why not HoloViews/hvPlot/Panel**, despite Datashader being genuinely
excellent for this problem: Panel apps run as a local Tornado/Bokeh
server: pan/zoom triggers Python-side recompute, pushed to BokehJS in a
browser or Chromium-based webview. Even packaged to look native, this is
architecturally a local web app, which the author wants to avoid
specifically because it reintroduces JS-adjacent tooling — even though
authoring itself stays pure Python. `datashader.Canvas` is still used
directly, as explained below — the objection is to Bokeh/Panel/HoloViews
as the *app layer*, not to Datashader.

**Why not napari**: known friction with dynamic cross-widget updates in
past experience (no clean single-source-of-truth reactive model without
extra work), a Shapes-layer scaling ceiling for millions of live
polygons, and a somewhat informal/undocumented API for embedding its Qt
widget inside a larger custom Qt application. Building the vispy/Qt
canvas directly avoids inheriting a ceiling or an embedding pattern the
author doesn't control. Trade-off accepted: more upfront engineering
(tile/chunk streaming into GPU textures, prefetch/eviction, level
selection, pan/zoom camera, layer compositing, hit-testing — all things
napari would have provided) in exchange for full control of exactly
these things.

**Why not VTK**: dropped early — less mature/precedented tooling for the
pathology-pyramid-plus-vector-annotations combination than the
vispy/napari-adjacent ecosystem, no compelling advantage identified.

**Why not pyqtgraph/matplotlib for everything (skip vispy)**: continuous
GPU-resident pan/zoom (moving already-uploaded textures) is qualitatively
different from recompute-then-redraw, and matters specifically for
fluid deep-zoom slide navigation. pyqtgraph and matplotlib are the right
tools for the *analytical* panels (where redraw-per-interaction is fine
or even correct UX) but not for the main gigapixel canvas.

**Reactive state (`psygnal`)**: standalone package (no napari
dependency, despite being extracted from napari's internals and used by
it) — `Signal`/`SignalInstance`, `EventedModel` (pydantic-based,
signal-per-field-change), `EventedList`/`EventedSet`/`EventedDict`. A
shared `EventedModel` (e.g. holding current selection, visible annotation
groups, current viewport) is the single source of truth every panel
subscribes to — this is what makes cross-widget coordination (click a
cluster in a pyqtgraph scatter -> highlight cells in the vispy canvas ->
update a matplotlib summary) tractable without ad hoc wiring between
widgets.

### Large annotation counts: level-of-detail split, using Datashader as a library

**Decision**: `datashader.Canvas` is used directly as a numpy-in/numpy-out
rasterization library (`.points()`, `.polygons()`, `.line()`, `.raster()`)
— no Bokeh/HoloViews/Panel dependency required to use it this way. As of
datashader >=0.16 it accepts geopandas GeoDataFrames directly (no forced
detour through spatialpandas), which fits the geopandas-based annotation
layer directly. `scverse/spatialdata` independently validates this
approach — it uses datashader internally once polygon counts exceed a
few hundred.

**LOD rule**: below a per-viewport object-count threshold, render live
vector primitives (GPU-instanced points / real polygon fills, accurate
hover/hit-testing). Above the threshold, rasterize the current viewport
with `datashader.Canvas` into an aggregate image and composite it as a
texture — selection at this zoom level resolves via the R-tree spatial
index rather than GPU picking against a live scene. This split is
zoom-driven and applies uniformly to both the main vispy canvas and any
analytical panel that needs to show large point/polygon counts (e.g. a
pyqtgraph or matplotlib panel can display a datashader-produced density
image just as well as the main canvas can).

## 7. Build order for the vispy tile viewer

This is the piece of the whole system with the most unknowns — tile
streaming into GPU textures at real scale is where "works in principle"
and "feels smooth in practice" are most likely to diverge. Recommended
order, timeboxing each step rather than trying to get it fully general
immediately:

1. `PyramidalRaster` (Zarr-backed) reads -> single vispy `ImageVisual`
   showing one static region, no zoom.
2. Add camera/pan/zoom (`vispy.scene.PanZoomCamera`).
3. Add pyramid-level switching (pick level from current scale factor) and
   background-thread tile fetch/texture upload with a prefetch/eviction
   policy.
4. Add annotation overlay layer(s): live vispy markers/lines below the
   LOD threshold, datashader-rasterized texture above it.
5. Wire selection/visibility to shared `psygnal` state; connect the
   pyqtgraph/matplotlib panels.

## 8. Open items / deliberately deferred

- **HDF5 vs Zarr concurrency claim**: reasoned from HDF5's documented
  global-lock behavior; not yet empirically benchmarked against this
  project's actual access pattern. Low risk (Zarr's concurrent-read
  behavior doesn't depend on this benchmark going any particular way),
  but worth a real measurement before writing it off entirely as settled.
- **napari vs custom vispy/Qt viewer**: resolved in favor of custom (see
  above), but the underlying uncertainty (does napari's Shapes-layer
  ceiling actually bite at this project's real polygon counts?) was
  never empirically tested — decided on engineering-control grounds
  instead. If the custom-viewer build order above turns out to be more
  expensive than expected, this is the one prior decision worth
  revisiting with fresh data.
- **`compass/sample.py`'s stale `wsitk_annot` import**: not yet resolved
  — depends on whether the per-sample `.cp` folder concept is still
  wanted (see section 4).
- **`processing/` <-> `viewer/` integration points**: not yet designed.
  Whether/how e.g. tissue detection or stain normalization surface as
  interactive, triggerable operations in the viewer (vs. purely
  offline/batch use) is open.
- **MRXS Bio-Formats cross-check tooling**: approach agreed (JVM-based,
  ingestion-only, one-time), concrete implementation (scyjava vs.
  shelling out to `bfconvert`, where it lives) not yet decided.
