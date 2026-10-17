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
import time
from dataclasses import dataclass

import numpy as np
from vispy import scene
from vispy.app import Timer
from vispy.visuals.transforms import STTransform

from ..core import PyramidalRaster
from .fetch import TileFetcher
from .inset import OverviewInset
from .navigation import (WHEEL_NOTCHES_PER_LEVEL, Action, Navigator, View,
                         ViewHistory, action_for_key)
from .state import ViewerState
from .tiling import TileCache, TileGrid, TileKey

#: mouse travel (logical px) before a press counts as a drag, not a click
_DRAG_THRESHOLD = 4.0
#: wheel notches further apart than this (s) start a new history step
_WHEEL_GESTURE_GAP = 0.6


#####
class _NotifyingPanZoomCamera(scene.PanZoomCamera):
    """PanZoomCamera that calls ``on_change()`` after every view change.

    Needed because the obvious hook, the subscene node's
    ``transform_change`` event, fires only once: PanZoomCamera mutates the
    same ``MatrixTransform`` in place, and ``Node.transform`` ignores
    re-assignment of the identical object. ``view_changed`` is the
    camera's own per-pan/zoom/resize entry point.
    """

    # class attrs so they can be set on the frozen node
    on_change = None
    #: ``factor -> factor`` hook limiting zoom; vispy's factor scales the
    #: view size, so > 1 zooms out
    zoom_clamp = None

    def view_changed(self):
        super().view_changed()
        if self.on_change is not None:
            self.on_change()

    def viewbox_key_event(self, event):
        # vispy binds Backspace to camera.reset() (back to the construction
        # rect, (0,0)-(1,1)); all keys are handled by SlideCanvas instead
        pass

    def viewbox_mouse_event(self, event):
        # mouse pan/zoom is handled by SlideCanvas; keep only the trackpad
        # pinch (goes through zoom() and its clamp)
        if event.type == "gesture_zoom":
            super().viewbox_mouse_event(event)

    def zoom(self, factor, center=None):
        # camera-driven zooms (trackpad pinch) are limited here
        if self.zoom_clamp is not None:
            # aspect is locked, so vispy uses the y factor for both axes
            f = factor if np.isscalar(factor) else factor[1]
            factor = self.zoom_clamp(float(f))
        super().zoom(factor, center)
##


@dataclass
class _Drag:
    """A mouse press in progress."""
    mode: str            # "pan" | "box" | "inset" | "menu"
    start: tuple[float, float]   # canvas logical px
    start_view: View
    moved: bool = False


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
        # no vispy default keys: its Esc closes just the canvas, F11 belongs
        # to the main window; navigation keys are handled in _on_key
        canvas_kwargs.setdefault("keys", None)
        canvas_kwargs.setdefault("size", (1024, 768))
        canvas_kwargs.setdefault("bgcolor", "#202020")
        super().__init__(**canvas_kwargs)
        self.unfreeze()

        self._raster = raster
        self._state = state if state is not None else ViewerState()
        self._grid = TileGrid.from_raster(raster, tile_size=tile_size)
        self._cache = TileCache(max_bytes=cache_bytes)
        self._margin = int(prefetch_margin)
        self._nav = Navigator(raster.shape(0).width, raster.shape(0).height,
                              self._grid.downsample, raster.nlevels)
        self._history = ViewHistory()
        self._drag: _Drag | None = None
        self._last_wheel = 0.0
        self._menu = None  # keeps the context menu alive while shown

        self._wanted: set[TileKey] = set()
        self._wanted_level: int = raster.nlevels - 1
        self._visuals: dict[TileKey, scene.visuals.Image] = {}
        self._ready: queue.SimpleQueue = queue.SimpleQueue()

        self._fetcher = TileFetcher(self._read_tile, self._tile_done,
                                    max_workers=max_workers)

        # --- scene graph ---
        self._view = self.central_widget.add_view()
        camera = _NotifyingPanZoomCamera(aspect=1)
        camera.flip = (False, True, False)  # y down, like image coordinates
        self._view.camera = camera

        w0, h0 = raster.shape(0).width, raster.shape(0).height
        top = raster.nlevels - 1
        self._overview_plane = raster.get_plane(top)
        self._overview = self._make_overview()
        camera.set_range(x=(0, w0), y=(0, h0), margin=0.02)

        self._band = scene.visuals.Line(pos=np.zeros((5, 2), np.float32),
                                        color=(1.0, 0.85, 0.0, 1.0), width=1.5,
                                        parent=self._view.scene, method="gl")
        self._band.set_gl_state("translucent", depth_test=False)
        self._band.order = 1000
        self._band.visible = False

        self._inset = OverviewInset(self, self._overview_plane,
                                    self._grid.downsample(top), w0, h0)

        # pan/zoom/resize hook (see _NotifyingPanZoomCamera for why not
        # transform_change); connected only after the scene graph is built
        camera.on_change = self._on_view_change
        camera.zoom_clamp = self._clamp_zoom_factor
        self.events.key_press.connect(self._on_key)
        self.events.mouse_press.connect(self._on_mouse_press)
        self.events.mouse_move.connect(self._on_mouse_move)
        self.events.mouse_release.connect(self._on_mouse_release)
        self.events.mouse_double_click.connect(self._on_mouse_double_click)
        self.events.mouse_wheel.connect(self._on_mouse_wheel)

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

    @property
    def navigator(self) -> Navigator:
        return self._nav

    def get_view(self) -> View:
        """Current camera position (center + zoom)."""
        sw, _ = self._screen_size()
        return View.from_rect(self._visible_rect(), sw)

    def set_view(self, view: View) -> None:
        """Move the camera (not recorded in the history)."""
        sw, sh = self._screen_size()
        self._view.camera.rect = view.rect(sw, sh)

    @property
    def overview_visible(self) -> bool:
        return self._inset.visible

    @overview_visible.setter
    def overview_visible(self, value: bool) -> None:
        self._inset.visible = value
        self.update()

    def go_to(self, target: View | None, kind: str | None = None) -> bool:
        """Move to ``target``, recording the current view in the history
        (see ``ViewHistory.record`` for ``kind``). False if no move."""
        current = self.get_view()
        if target is None or target == current:
            return False
        self._history.record(current, kind)
        self.set_view(target)
        return True

    def navigate(self, action: Action) -> bool:
        """Run a navigation action (see ``navigation.CANVAS_KEYS``).
        Returns False if it does not apply (e.g. missing pyramid level,
        empty history)."""
        sw, sh = self._screen_size()
        current = self.get_view()
        kind = action[0]
        if kind in ("back", "forward"):
            step = self._history.back if kind == "back" else self._history.forward
            target = step(current)
            if target is None:
                return False
            self.set_view(target)
            return True
        if kind == "zoom":
            target = self._nav.zoom(current, action[1], sw, sh)
        elif kind == "level":
            target = self._nav.at_level(current, action[1], sw, sh)
        elif kind == "fit":
            target = self._nav.fit(sw, sh)
        elif kind == "pan":
            target = self._nav.pan(current, action[1], action[2], sw, sh)
        else:
            raise ValueError(f"unknown navigation action {action!r}")
        # repeated zooms/pans form one history step; jumps are separate
        return self.go_to(target, kind if kind in ("zoom", "pan") else repr(action))

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

    def _on_key(self, event) -> None:
        name = event.key.name if event.key is not None else None
        mods = {m.name for m in event.modifiers}
        action = action_for_key(name, event.text or "", mods)
        if action is None:
            return
        self.navigate(action)
        event.handled = True

    # ------------------------------------------------------------------
    # mouse
    # ------------------------------------------------------------------

    def _to_world(self, pos) -> tuple[float, float]:
        """Canvas logical px -> level-0 px."""
        tr = self.scene.node_transform(self._view.scene)
        x, y = tr.map(np.asarray(pos, float)[:2])[:2]
        return float(x), float(y)

    def _on_mouse_press(self, event) -> None:
        if event.button in (4, 5):  # side buttons
            self.navigate(("back",) if event.button == 4 else ("forward",))
            return
        pos = tuple(map(float, event.pos[:2]))
        mods = {m.name for m in event.modifiers}
        if event.button == 1 and self._inset.slide_point(*pos) is not None:
            mode = "inset"
        elif event.button == 1 and "Shift" in mods:
            mode = "box"
        elif event.button in (1, 3):
            mode = "pan"
        elif event.button == 2:
            mode = "menu"
        else:
            return
        self._drag = _Drag(mode, pos, self.get_view())
        if mode == "inset":
            self._inset_jump(pos)
        event.handled = True

    def _on_mouse_move(self, event) -> None:
        pos = tuple(map(float, event.pos[:2]))
        self._update_cursor(pos)
        d = self._drag
        if d is None:
            return
        if not d.moved:
            if np.hypot(pos[0] - d.start[0], pos[1] - d.start[1]) < _DRAG_THRESHOLD:
                return
            d.moved = True
            self._set_cursor({"pan": "ClosedHandCursor", "box": "CrossCursor"}.get(d.mode))
        if d.mode == "pan":
            # from the press position, so rounding does not accumulate
            k = d.start_view.upp * self.pixel_scale  # level-0 px per logical px
            sw, sh = self._screen_size()
            self.set_view(self._nav.clamp(
                View(d.start_view.cx - (pos[0] - d.start[0]) * k,
                     d.start_view.cy - (pos[1] - d.start[1]) * k,
                     d.start_view.upp), sw, sh))
        elif d.mode == "box":
            (x0, y0), (x1, y1) = self._to_world(d.start), self._to_world(pos)
            self._band.set_data(pos=np.array([[x0, y0], [x1, y0], [x1, y1],
                                              [x0, y1], [x0, y0]], np.float32))
            self._band.visible = True
            self.update()
        elif d.mode == "inset":
            self._inset_jump(pos, clamp_to_inset=True)

    def _on_mouse_release(self, event) -> None:
        d, self._drag = self._drag, None
        if d is None:
            return
        self._set_cursor(None)
        pos = tuple(map(float, event.pos[:2]))
        if d.mode in ("pan", "inset"):
            if self.get_view() != d.start_view:
                self._history.record(d.start_view)  # one step per drag
        elif d.mode == "box":
            self._band.visible = False
            self.update()
            if d.moved:
                sw, sh = self._screen_size()
                (x0, y0), (x1, y1) = self._to_world(d.start), self._to_world(pos)
                self.go_to(self._nav.to_box(x0, y0, x1, y1, sw, sh))
        elif d.mode == "menu" and not d.moved:
            self._show_context_menu(pos)

    def _on_mouse_double_click(self, event) -> None:
        pos = event.pos[:2]
        if event.button != 1 or self._inset.slide_point(*pos) is not None:
            return
        factor = 0.5 if "Shift" in {m.name for m in event.modifiers} else 2.0
        sw, sh = self._screen_size()
        x, y = self._to_world(pos)
        self.go_to(self._nav.zoom_at(self.get_view(), factor, x, y, sw, sh))

    def _on_mouse_wheel(self, event) -> None:
        notches = float(event.delta[1])
        if notches == 0 or self._inset.slide_point(*event.pos[:2]) is not None:
            return
        current = self.get_view()
        now = time.monotonic()
        if now - self._last_wheel > _WHEEL_GESTURE_GAP:
            self._history.record(current)  # a burst of notches = one step
        self._last_wheel = now
        sw, sh = self._screen_size()
        x, y = self._to_world(event.pos)
        factor = 2.0 ** (notches / WHEEL_NOTCHES_PER_LEVEL)
        self.set_view(self._nav.zoom_at(current, factor, x, y, sw, sh))
        event.handled = True

    def _inset_jump(self, pos, clamp_to_inset: bool = False) -> None:
        p = self._inset.slide_point(*pos)
        if p is None and clamp_to_inset:  # dragged out of the thumbnail
            p = self._inset.slide_point_clamped(*pos)
        if p is None:
            return
        sw, sh = self._screen_size()
        self.set_view(self._nav.center_on(self.get_view(), *p, sw, sh))

    def _update_cursor(self, pos) -> None:
        x, y = self._to_world(pos)
        self._state.cursor = (x, y)
        hit = self._pixel_at(x, y)
        self._state.cursor_value_level = None if hit is None else hit[0]
        self._state.cursor_value = None if hit is None else hit[1]

    def _pixel_at(self, x: float, y: float) -> tuple[int, tuple[float, ...]] | None:
        """(level, channel values) at level-0 point (x, y) from the finest
        tile already in memory (no reads), else from the overview."""
        if not (0 <= x < self._nav.slide_w and 0 <= y < self._nav.slide_h):
            return None
        ts = self._grid.tile_size
        top = self._raster.nlevels - 1
        for level in range(top + 1):
            d = self._grid.downsample(level)
            lx, ly = int(x // d), int(y // d)
            if level == top:
                arr, ox, oy = self._overview_plane, 0, 0
            else:
                key = TileKey(level, lx // ts, ly // ts)
                arr = self._cache.get(key)
                if arr is None:
                    continue
                ox, oy = key.ix * ts, key.iy * ts
            iy, ix = ly - oy, lx - ox
            if 0 <= iy < arr.shape[0] and 0 <= ix < arr.shape[1]:
                v = np.atleast_1d(arr[iy, ix])
                return level, tuple(float(c) for c in v)
        return None

    def _set_cursor(self, shape: str | None) -> None:
        from PySide6 import QtCore
        if shape is None:
            self.native.unsetCursor()
        else:
            self.native.setCursor(getattr(QtCore.Qt.CursorShape, shape))

    def _show_context_menu(self, pos) -> None:
        from PySide6 import QtCore, QtGui, QtWidgets

        x, y = self._to_world(pos)
        sw, sh = self._screen_size()
        current = self.get_view()
        menu = QtWidgets.QMenu(self.native)
        menu.addAction("Center here",
                       lambda: self.go_to(self._nav.center_on(current, x, y, sw, sh)))
        levels = menu.addMenu("Zoom to level here")
        for lv in range(self._raster.nlevels):
            try:
                mag = f" ({self._raster.get_magnification_for_level(lv):.3g}x)"
            except Exception:
                mag = ""
            target = self._nav.at_level(View(x, y, current.upp), lv, sw, sh)
            levels.addAction(f"level {lv}{mag}",
                             lambda t=target, lv=lv: self.go_to(t, f"level{lv}"))
        menu.addAction("Fit slide", lambda: self.navigate(("fit",)))
        menu.addSeparator()
        clip = QtGui.QGuiApplication.clipboard()
        px_txt = f"{x:.0f}, {y:.0f}"
        menu.addAction(f"Copy position: {px_txt} px",
                       lambda: clip.setText(px_txt))
        try:
            mpp = float(self._raster.native_resolution)
            um_txt = f"{x * mpp:.1f}, {y * mpp:.1f}"
            menu.addAction(f"Copy position: {um_txt} µm",
                           lambda: clip.setText(um_txt))
        except Exception:
            pass
        self._menu = menu
        menu.popup(self.native.mapToGlobal(QtCore.QPoint(int(pos[0]), int(pos[1]))))

    def _screen_size(self) -> tuple[float, float]:
        """Viewbox size in physical (device) pixels."""
        w, h = self._view.size
        return max(w, 1) * self.pixel_scale, max(h, 1) * self.pixel_scale

    def _visible_rect(self) -> tuple[float, float, float, float]:
        """(x0, y0, w, h) in level-0 px actually on screen. Not
        ``camera.rect``: with a locked aspect vispy widens that rect to the
        window's aspect ratio when drawing."""
        w, h = self._view.size
        pts = self._view.camera.transform.imap(np.array([[0, 0], [w, h]], float))
        (xa, ya), (xb, yb) = pts[0, :2], pts[1, :2]
        return min(xa, xb), min(ya, yb), abs(xb - xa), abs(yb - ya)

    def _clamp_zoom_factor(self, factor: float) -> float:
        sw, sh = self._screen_size()
        upp = self.get_view().upp
        if upp <= 0:
            return factor
        lo, hi = self._nav.upp_limits(sw, sh)
        return min(max(upp * factor, lo), hi) / upp

    def _update_tiles(self) -> None:
        viewport = self._visible_rect()
        if viewport[2] <= 0 or viewport[3] <= 0:
            return
        sw, sh = self._screen_size()
        scale = sw / viewport[2]
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

        self._inset.layout(*self.size)
        self._inset.set_viewport(viewport)

        # publish navigation state for the panels
        self._state.viewport = viewport
        self._state.canvas_size = (int(round(sw)), int(round(sh)))
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
        _layer_by_order(im)
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
        _layer_by_order(im)
        im.order = -1000
        return im
##


def _layer_by_order(visual) -> None:
    """Stack a visual purely by draw order (``visual.order``).

    vispy's Image uses the 'translucent' GL preset, which has depth testing
    on; all slide layers sit at z=0, so with the default 'less' depth test
    the first-drawn layer (the coarse overview) wins and every finer tile
    drawn later is discarded — the screen shows the upscaled overview no
    matter which level was loaded.
    """
    visual.set_gl_state("translucent", depth_test=False, cull_face=False)
##
