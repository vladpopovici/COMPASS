# COMPASS

Desktop (non-browser) Python application for viewing large pathology whole-slide
images (rasters up to ~200,000 x 200,000 px, multi-channel, micron-calibrated)
together with millions of associated vector/raster annotations
(cell-level points, region polygons, segmentation maps, sparse molecular
profiles). Local-only: no browser, no JS, single Qt process.

Full design rationale and decision log: @docs/architecture.md

## Package layout

```
src/compass/
  core/         # PyramidalRaster protocol + Zarr backend (ZarrRaster,
                # write_pyramid), annotation data-access layer
                # (AnnotationStore: SQLite/R-tree -> geopandas/polars).
                # Minimal deps. Everything else depends on this; this
                # depends on nothing else in this project.
  connector/    # AnnData/SpatialData conversion, WSI ingestion
                # (OpenSlide / Bio-Formats), pyvips pyramid writer.
                # Heavy/volatile deps isolated here (optional-dependency
                # extra "connector"). Never imported by the running viewer.
  processing/   # Domain algorithms: tissue detection, stain
                # normalization, patch sampling, image registration,
                # spatial-transcriptomics (Visium) tools.
                # numpy/geometry in, numpy/geometry out. Extra "processing".
  viewer/       # vispy/Qt canvas, psygnal reactive state, pyqtgraph
                # + matplotlib/seaborn analytical panels.
  legacy/       # the original flat compass modules, unchanged, being
                # ported per @docs/architecture.md (extra "legacy" pulls
                # in everything they import). No new code here.
  tests/        # pytest suite (test_core_*.py) + legacy notebooks.
```

Migration status: `core/` is ported (pyramid protocol, Magnification, Zarr
raster, vectorized AnnotationStore); `connector/` is ported (`wsi.py`
OpenSlide reader, `ingest.py: wsi2zarr` pyvips->Zarr pyramid baking,
`export.py: raster2tiff` OME-TIFF interchange); `processing/` and
`viewer/` are scaffolded stubs. See @docs/architecture.md for the
legacy-module -> target mapping and what changes on the way.

## Hard constraints (do not relitigate without asking)

- **No browser/JS rendering path.** No Bokeh/Panel/HoloViews as the app's
  UI layer, no QWebEngineView-embedded viewers. Author is deliberately
  avoiding JS. `datashader.Canvas` may be used as a plain numpy-in/numpy-out
  rasterization library (no Bokeh dependency), never through HoloViews/Panel.
- **No OpenSlide (or libvips' OpenSlide-backed loader) in the runtime
  viewer path.** OpenSlide/Bio-Formats disagree on MRXS coordinate
  interpretation. OpenSlide-based reading is ingestion-only, lives in
  `connector/`, and is used to bake a canonical, verified pyramid once.
- **No AnnData / SpatialData as a runtime dependency of `core/`,
  `processing/`, or `viewer/`.** They're isolated inside `connector/` as
  one-shot conversion tools. `core/` exposes plain
  numpy / pandas / polars / geopandas / dask objects only.
- **Raster storage is Zarr (v3, sharding codec), not HDF5.** Chosen over
  HDF5 specifically because HDF5's global library lock serializes
  concurrent tile reads across threads; Zarr chunks/shards don't have
  that problem. OME-Zarr-like layout by convention (per-level groups,
  `mpp`/channel metadata as attrs).
- **UI is single-toolkit: Qt.** vispy for the gigapixel canvas, pyqtgraph
  for live-interactive analytical widgets, matplotlib+seaborn (via
  `FigureCanvasQTAgg`) for refined statistical plots, `psygnal`
  (`EventedModel`) for cross-widget reactive state. No napari dependency —
  vispy/Qt canvas is hand-built to avoid napari's Shapes-layer scaling
  ceiling and its informal Qt-embedding API.
- **Millions of annotation objects**: never materialize as one Python
  object per annotation for bulk operations. Query results from the
  annotation store should come back as geopandas/polars DataFrames
  (vectorized), not lists of per-object instances. Use `datashader`
  rasterization once viewport object counts get large; live vector
  rendering only below that threshold.

## Commands

Managed with `uv` (`~/.local/bin/uv`; the poetry lockfile is gone).

- `uv sync` — core deps + dev group only.
- `uv sync --extra legacy` — everything the legacy modules import
  (implies extras `connector` + `processing`). Note: importing
  `compass.legacy.*` additionally needs system libs not present on this
  machine (libvips, OpenSlide); core tests deliberately don't.
- `uv run pytest src/compass/tests -q` — run the test suite.
- `uv run python -c "import compass.core"` must stay fast and must not
  pull in Qt/OpenSlide/pyvips/cv2/h5py/AnnData (tested implicitly; keep
  it that way).

## Conventions

- Python 3.12+ (legacy code has 3.12/3.13 bytecode caches).
- Type hints throughout; `pydantic` for small value types (`ImageShape`, `Px`).
- Coordinates are always (x, y), pixel units at a specified pyramid level
  unless explicitly in microns — see `Magnification` class for conversions.
- Logging: configure explicitly per entry point. Do not call
  `logging.basicConfig()` at module import time (legacy `core.py` does this
  — fix when touched, don't propagate the pattern).
