# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""The vispy gigapixel canvas (build-order steps 1-3 of
docs/architecture.md section 7): Zarr-backed tile streaming into GPU
textures with pan/zoom and pyramid-level switching.

GUI-only module — importing it needs vispy; instantiating
:class:`SlideCanvas` needs OpenGL and a vispy backend (select one first,
e.g. ``vispy.app.use_app("pyside6")``). All tile logic lives in the
headless-testable :mod:`compass.viewer.tiling` / :mod:`compass.viewer.fetch`;
this module only owns the scene graph and thread marshalling.

How rendering works:
- world coordinates are level-0 pixels, y down (``PanZoomCamera`` with a
  flipped y axis), shared with the annotation store;
- a static overview of the coarsest level is always present as backdrop,
  so no viewport is ever blank;
- on every camera change the wanted tile set for the current zoom's level
  is computed; cached tiles become ``Image`` visuals immediately, missing
  ones are fetched by background threads (center-of-screen first) and
  arrive via a queue drained on a vispy timer (GUI thread);
- visuals from other levels are kept as stale backdrop until every wanted
  tile of the current level is on screen, then pruned; draw order is
  coarse-below-fine.
"""

import queue

import numpy as np
from vispy import scene
from vispy.app import Timer
from vispy.visuals.transforms import STTransform

from ..core import PyramidalRaster
from .fetch import TileFetcher
from .state import ViewerState
from .tiling import TileCache, TileGrid, TileKey


#####
class SlideCanvas(scene.SceneCanvas):
    """vispy canvas showing one pyramidal raster with pan/zoom tile
    streaming. Embed in Qt via the ``.native`` widget.

    Args:
        raster: the slide (e.g. ``compass.core.ZarrRaster``).
        state: shared reactive state; ``viewport``/``canvas_size``/
            ``current_level`` are kept up to date by the canvas. A private
            one is created if not given.
        tile_size: tile edge in pixels; match the raster's chunk size.
        cache_bytes: decoded-tile RAM budget.
        prefetch_margin: rings of off-screen tiles fetched around the
            viewport.
        max_workers: tile reader threads.
    """

    def __init__(self,
                 raster: PyramidalRaster,
                 state: ViewerState | None = None,
                 tile_size: int = 512,
                 cache_bytes: int = 768 * 2 ** 20,
                 prefetch_margin: int = 1,
                 max_workers: int = 2,
                 **canvas_kwargs):
        canvas_kwargs.setdefault("keys", "interactive")
        canvas_kwargs.setdefault("size", (1024, 768))
        canvas_kwargs.setdefault("bgcolor", "#202020")
        super().__init__(**canvas_kwargs)
        self.unfreeze()

        self._raster = raster
        self._state = state if state is not None else ViewerState()
        self._grid = TileGrid.from_raster(raster, tile_size=tile_size)
        self._cache = TileCache(max_bytes=cache_bytes)
        self._margin = int(prefetch_margin)

        self._wanted: set[TileKey] = set()
        self._wanted_level: int = raster.nlevels - 1
        self._visuals: dict[TileKey, scene.visuals.Image] = {}
        self._ready: queue.SimpleQueue = queue.SimpleQueue()

        self._fetcher = TileFetcher(self._read_tile, self._tile_done,
                                    max_workers=max_workers)

        # --- scene graph ---
        self._view = self.central_widget.add_view()
        camera = scene.PanZoomCamera(aspect=1)
        camera.flip = (False, True, False)  # y down, like image coordinates
        self._view.camera = camera

        w0, h0 = raster.shape(0).width, raster.shape(0).height
        self._overview = self._make_overview()
        camera.set_range(x=(0, w0), y=(0, h0), margin=0.02)

        # pan/zoom hook: the camera reassigns the subscene's transform on
        # every move, so that node's transform_change is the reliable signal
        # (the camera's own node transform never changes)
        self._view.scene.events.transform_change.connect(self._on_view_change)
        self.events.resize.connect(self._on_view_change)

        # tiles decoded on worker threads land on the GUI thread here
        self._drain_timer = Timer(interval=1 / 30,
                                  connect=self._drain_ready, start=True)

        self.freeze()
        self._update_tiles()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @property
    def state(self) -> ViewerState:
        return self._state

    @property
    def view(self) -> scene.ViewBox:
        """The scene viewbox — annotation overlays attach under
        ``canvas.view.scene`` in level-0 coordinates."""
        return self._view

    def close(self) -> None:
        self._drain_timer.stop()
        self._fetcher.close()
        super().close()

    # ------------------------------------------------------------------
    # tile plumbing
    # ------------------------------------------------------------------

    def _read_tile(self, key: TileKey) -> np.ndarray:
        x0, y0, w, h = self._grid.tile_bounds_px(key)
        return self._raster.get_region_px(x0, y0, w, h, level=key.level)

    def _tile_done(self, key: TileKey, arr: np.ndarray) -> None:
        # worker thread: cache is thread-safe, the scene graph is not —
        # only enqueue; the timer drains on the GUI thread.
        self._cache.put(key, arr)
        self._ready.put(key)

    def _drain_ready(self, event=None) -> None:
        added = False
        while True:
            try:
                key = self._ready.get_nowait()
            except queue.Empty:
                break
            if key in self._wanted and key not in self._visuals:
                self._add_visual(key)
                added = True
        if added:
            self._prune_visuals()
            self.update()

    def _on_view_change(self, event=None) -> None:
        self._update_tiles()

    def _update_tiles(self) -> None:
        rect = self._view.camera.rect
        if rect.width <= 0 or rect.height <= 0:
            return
        viewport = (rect.left, rect.bottom, rect.width, rect.height)
        scale = self.physical_size[0] / rect.width
        level = self._grid.level_for_scale(scale)

        keys = self._grid.tiles_in_viewport(viewport, level, margin=self._margin)
        self._wanted = set(keys)
        self._wanted_level = level

        missing = []
        for key in keys:  # center-first order
            if key in self._visuals:
                continue
            if key in self._cache:
                self._add_visual(key)
            else:
                missing.append(key)
        self._fetcher.request(missing)
        self._fetcher.cancel_except(self._wanted)
        self._prune_visuals()

        # publish navigation state for the panels
        self._state.viewport = viewport
        self._state.canvas_size = tuple(self.physical_size)
        self._state.current_level = level

    def _add_visual(self, key: TileKey) -> None:
        arr = self._cache.get(key)
        if arr is None:  # evicted between decode and drain; will be re-requested
            return
        im = scene.visuals.Image(arr, parent=self._view.scene,
                                 interpolation="linear")
        d = self._grid.downsample(key.level)
        x0, y0, _, _ = self._grid.tile_bounds_level0(key)
        im.transform = STTransform(scale=(d, d, 1), translate=(x0, y0, 0))
        im.order = -key.level  # coarse below fine
        self._visuals[key] = im

    def _prune_visuals(self) -> None:
        complete = self._wanted <= self._visuals.keys()
        for key in list(self._visuals):
            if key in self._wanted:
                continue
            # stale other-level tiles remain as backdrop until the current
            # level fully covers the viewport
            if key.level != self._wanted_level and not complete:
                continue
            self._visuals.pop(key).parent = None

    def _make_overview(self) -> scene.visuals.Image:
        """Static coarsest-level backdrop so the viewport is never blank."""
        level = self._raster.nlevels - 1
        plane = self._raster.get_plane(level)
        im = scene.visuals.Image(plane, parent=self._view.scene,
                                 interpolation="linear")
        d = self._grid.downsample(level)
        im.transform = STTransform(scale=(d, d, 1))
        im.order = -1000
        return im
##
