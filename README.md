COMPASS: Computational Pathology and Spatial Statistics
===============================================================

COMPASS is both a library, collection of tools for analyzing images
in the context of molecular data, and a viewer. The viewer is 
a local, non-browser, single-process Qt application (and Python library)
for viewing large pathology whole-slide images — rasters up to roughly
200,000 x 200,000 px, multi-channel, micron-calibrated — together with
millions of associated vector/raster annotations (cell-level points,
region polygons, segmentation maps, sparse molecular profiles).

Documentation
-------------

* [`docs/index.md`](docs/index.md) — package layout, current API, and
  how to install/run/test the code in `src/compass/` (excluding
  `src/compass/legacy/`, the original flat modules being ported over).
* [`docs/architecture.md`](docs/architecture.md) — the design decision
  log: what was decided, why, and what alternatives were rejected.

Package layout
---------------

```
src/compass/
  core/         PyramidalRaster protocol + Zarr backend, vectorized
                annotation data-access layer. Minimal deps.
  connector/    AnnData/SpatialData conversion, WSI ingestion
                (OpenSlide/Bio-Formats), pyvips pyramid writer.
  processing/   Domain algorithms (tissue detection, stain
                normalization, patch sampling, registration, Visium).
  viewer/       vispy/Qt gigapixel canvas, psygnal reactive state,
                pyqtgraph/matplotlib analytical panels.
  legacy/       original flat modules, unchanged, being ported per
                docs/architecture.md. No new code here.
```

Raster images are stored as pyramidal Zarr v3 (sharding codec) groups,
one array per level (`0` = highest resolution), with `mpp`/channel
metadata as attrs — see `docs/architecture.md` for why Zarr rather than
the HDF5 layout this project originally used.

Quick start
-----------

```bash
uv sync                    # core deps + dev group
uv run pytest src/compass/tests -q

# optional extras, as needed:
uv sync --extra connector  # WSI ingestion (pyvips, OpenSlide)
uv sync --extra processing # domain algorithms (scikit-image, opencv, ...)
uv sync --extra viewer     # Qt/vispy viewer (needs OpenGL)
```

See [`docs/index.md`](docs/index.md) for usage examples of each package.
