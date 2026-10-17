---
title: "COMPASS viewer"
subtitle: "User guide and developer notes for `compass.viewer`"
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

The COMPASS viewer shows a pyramidal whole-slide image stored as Zarr (see
[`index.md`](index.md), *File formats*) in a native Qt window. You can pan
and zoom continuously from the whole slide down to beyond full resolution,
with the keyboard or the mouse. This document has two parts:

- **Part I — Using the viewer**: starting it, the screen, every key and
  mouse binding.
- **Part II — How it works**: rendering pipeline, navigation model, module
  layout, embedding API, testing and the vispy pitfalls behind some of the
  code.

The *why* behind the toolkit choices (vispy + Qt, no napari, no
browser/JS) is in [`architecture.md`](architecture.md), §6–7.

**Status.** Build steps 1–3 of architecture §7 are done: tile streaming
with pyramid-level switching, plus keyboard/mouse navigation and an
overview thumbnail. Annotation overlays (step 4) and analysis panels
(step 5) are not built yet. See [Limitations](#limitations-and-next-steps).

# Part I — Using the viewer {.unnumbered}

# Starting the viewer

```bash
uv sync --extra viewer        # vispy, PySide6, psygnal; needs OpenGL
uv run python -m compass.viewer slide.zarr
```

| Option | Default | Meaning |
|---|---|---|
| `--tile-size N` | 512 | Tile edge in pixels. Keep it equal to the store's chunk size (512 for stores written by `wsi2zarr` / `ometiff2zarr`) so one tile maps to whole chunks |
| `--cache-mb N` | 768 | RAM budget for decoded tiles |
| `--workers N` | 2 | Background tile-reader threads |
| `-v` | off | Debug logging to stderr |

The slide must be a Zarr pyramid written by
`compass.connector.wsi2zarr`, `compass.connector.ometiff2zarr` or
`compass.core.write_pyramid`. The viewer never reads `.ndpi`/`.mrxs`/…
directly (architecture §3): convert first.

# The screen

```
┌──────────────────────────────────────────────────────────┐
│                                          ┌────────────┐  │
│                                          │ thumbnail  │  │
│                                          │  [▭]       │  │
│                  slide canvas            └────────────┘  │
│                                                          │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ level 2 | zoom 0.145x (≈2.91x objective) | 3.12 µm/screen px | cursor … │  ← info bar
└──────────────────────────────────────────────────────────┘
```

- **Slide canvas**: the image. While tiles of a finer level are loading,
  you briefly see the coarser level, upscaled. The coarsest level is
  always drawn underneath, so the screen is never empty.
- **Overview thumbnail** (top-right): the whole slide, with the part on
  screen outlined in yellow. A very small outline (deep zoom) is drawn at
  a minimum size so it stays visible.
- **Info bar** (bottom): see [The info bar](#the-info-bar).

`i` shows or hides the info bar **and** the thumbnail together.

# Keyboard

Keys act on the slide when the canvas has keyboard focus (it has it at
start; click the canvas to give it back). `?` or `F1` shows this list in
the application.

## Zoom

| Key | Action |
|---|---|
| `+` or `=` | Zoom in ×2 (one pyramid level), around the view center |
| `-` | Zoom out ×2 |
| `0` | Pyramid level 0 at 1:1 — one screen pixel per slide pixel (full resolution) |
| `1` … `9` | Level *n* at 1:1, around the view center. Ignored if the slide has no level *n* |
| `Home` or `f` | Fit the whole slide in the window |

## Pan

| Key | Action |
|---|---|
| Arrow keys | Pan by ¼ of the view |
| `Shift` + arrow | Pan by a full screen |
| `PgUp` / `PgDn` | Pan up / down by a full screen |

## View history

| Key | Action |
|---|---|
| `Backspace` or `Alt+←` | Previous view |
| `Alt+→` | Next view |

## Window

| Key | Action |
|---|---|
| `q` or `Ctrl+Q` | Quit |
| `F11` | Toggle fullscreen |
| `i` | Show/hide the info bar and the overview thumbnail |
| `?` or `F1` | Show the list of keys and mouse actions |
| `Esc` | Nothing (reserved: will clear the selection once annotations exist) |

# Mouse

| Mouse | Action |
|---|---|
| Left-drag | Pan (cursor becomes a closed hand) |
| Middle-drag | Pan |
| Wheel / two-finger scroll | Zoom around the cursor; 4 notches = one pyramid level (×2) |
| Trackpad pinch | Zoom |
| Double-click | Zoom in ×2, keeping the clicked point under the cursor |
| `Shift` + double-click | Zoom out ×2, same anchoring |
| `Shift` + left-drag | Draw a box (yellow); on release the view fits the box |
| Back / forward side buttons | Previous / next view |
| Right-click | Context menu (below) |
| Click or drag in the thumbnail | Jump to that part of the slide; dragging moves the view continuously |
| Single left-click | Nothing yet — reserved for selecting annotation objects |

**Right-click menu**, acting on the point you clicked:

- **Center here** — keep the zoom, move the point to the center.
- **Zoom to level here ▸** — level *n* at 1:1 centered on the point (each
  entry shows the level's objective power).
- **Fit slide.**
- **Copy position** — the point's coordinates, in level-0 pixels or in
  µm, to the clipboard (as `x, y`).

# Zoom limits and history

**Limits.** Every zoom, whether from keys, wheel, double-click, box,
pinch or menu, stays between:

- **8× past full resolution** (8 screen pixels per level-0 pixel), and
- **fit-to-window** (whole slide visible, with a 2 % margin).

When the view is larger than the slide along an axis, the slide is
centered on that axis; otherwise the view center is kept on the slide.
You can't lose the slide off-screen.

**History** works like a browser's back/forward. These each make one
history step:

- each jump: a number key, `Home`/`f`, double-click, box zoom, any
  right-click menu entry, a thumbnail click/drag;
- each mouse drag (one step per drag, recorded on release);
- a burst of wheel notches (notches less than 0.6 s apart are one step);
- a run of zoom keys, or a run of pan keys — holding an arrow key is one
  step, not dozens.

Going back and then navigating anywhere new clears the forward history.

# The info bar

```
level 0 | zoom 1x (≈20x objective) | 0.455 µm/screen px | cursor (34734, 25296) px = (15789.4, 11499.0) µm | RGB (234, 233, 231) @ level 0
```

| Field | Meaning |
|---|---|
| `level` | Pyramid level being drawn |
| `zoom` | Screen pixels per level-0 pixel. `1x` = full resolution |
| `≈…x objective` | Equivalent microscope objective: slide's native objective power × zoom |
| `µm/screen px` | Physical size of one screen pixel at the current zoom |
| `cursor` | Mouse position in level-0 pixels and in µm |
| pixel value | Value under the cursor and the level it was read from (see below) |

The pixel value is read from the **finest tile already in memory** at
the cursor, without reading anything from disk. Zoomed out, it may come
from a coarse level (an average of many level-0 pixels); the
`@ level n` suffix says which. RGB slides show `RGB (r, g, b)`.
Multi-channel slides show `name=value` per channel (names from the
store's `channel_names`), up to 8 channels.

# Part II — How it works {.unnumbered}

# Modules

```
src/compass/viewer/
  __init__.py     GUI-free exports: ViewerState, TileGrid, TileCache, TileFetcher, TileKey
  state.py        ViewerState (psygnal EventedModel): shared cross-widget state
  tiling.py       TileGrid (level choice, tiles per viewport), TileCache (LRU)
  fetch.py        TileFetcher: deduplicating background reads
  navigation.py   View, Navigator, ViewHistory, key map, thumbnail geometry
  canvas.py       SlideCanvas (vispy): tiles -> textures, input handling
  inset.py        OverviewInset (vispy): the thumbnail
  app.py          `python -m compass.viewer`: main window, app keys, info bar, help
```

`state`, `tiling`, `fetch` and `navigation` import neither vispy nor Qt
and are unit-tested headless (`src/compass/tests/test_viewer_*.py`).
`canvas`, `inset` and `app` need vispy, PySide6 and OpenGL.

Dependency direction follows the project rule: the viewer imports
`compass.core` (and in future `compass.processing`), never
`compass.connector`.

# Coordinates

- **World coordinates are level-0 pixels, y down**, the same as the
  annotation store. A tile of level *L* is an image visual scaled by the
  level's downsample (`2**L` for a ×2 pyramid) and translated to its
  level-0 position, so overlays need no extra transform.
- A **view** is `View(cx, cy, upp)`: center in level-0 pixels and zoom as
  **level-0 pixels per physical screen pixel** (`upp`). `upp = 1` is level
  0 at 1:1, `upp = 2**n` is level *n* at 1:1. A view stores no rectangle,
  so it survives window resizes; the canvas converts it to a camera
  rectangle for the current window size (`View.rect(screen_w, screen_h)`).
- "Physical" pixels are device pixels (HiDPI-aware):
  `SlideCanvas._screen_size()` is the view box size × `pixel_scale`. Mouse
  event positions are logical pixels and are converted with the scene
  transform.

# Rendering pipeline

```
camera change (pan / zoom / resize)
  └─> _update_tiles
        ├─ TileGrid.level_for_scale ............ which level
        ├─ TileGrid.tiles_in_viewport .......... which tiles (center-first, +1 ring)
        ├─ cached?  ──yes──> Image visual
        │           └─no──> TileFetcher (worker threads) ─> TileCache + queue
        │                        queue ─(timer, 30 Hz, GUI thread)─> Image visual
        └─ _prune_visuals ...................... drop other levels once covered
```

1. **Trigger.** The camera is `_NotifyingPanZoomCamera`, whose
   `view_changed()` calls `SlideCanvas._update_tiles`. It runs on every
   pan, zoom and window resize.
2. **Visible rectangle.** `_visible_rect()` maps the view box corners
   back to world coordinates. This is *not* `camera.rect`: with a locked
   aspect ratio vispy widens that rect to the window's aspect when
   drawing.
3. **Level.** `scale = screen_w / visible_w` (screen px per level-0 px);
   `TileGrid.level_for_scale` picks the coarsest level that is still at
   least screen resolution.
4. **Tiles.** `tiles_in_viewport` lists the tiles covering the visible
   rect plus one ring of prefetch (`prefetch_margin`), sorted
   center-first. Cached tiles become visuals at once. Missing ones go to
   the fetcher, in that order. Queued reads no longer wanted are
   cancelled.
5. **Reads.** `TileFetcher` worker threads call
   `raster.get_region_px(...)`. Zarr shard reads have no global lock, so
   they run in parallel (architecture §2). A finished tile is put in the
   cache and its key in a queue. The worker thread never touches the
   scene graph.
6. **Upload.** A vispy timer on the GUI thread drains the queue (30 Hz).
   Each tile still wanted becomes a `scene.visuals.Image`.
7. **Stacking.** Tiles are stacked by draw order: `order = -level`, so
   fine draws over coarse. The coarsest-level backdrop has
   `order = -1000`. Tiles from the previous level stay as backdrop until
   the current level covers the viewport, then they are pruned. Zooming
   never shows holes.
8. **State.** `viewport`, `canvas_size` and `current_level` are written
   to `ViewerState`, and the thumbnail outline is updated.

# Navigation model

`navigation.Navigator` holds all view arithmetic for one slide, as pure
functions of a `View` and the screen size. Every method returns a
**clamped** view (zoom limits, centering rules from
[Zoom limits and history](#zoom-limits-and-history)).

| Method | Used by |
|---|---|
| `fit(sw, sh)` | `Home`/`f`, menu, zoom-out limit |
| `zoom(view, factor, sw, sh)` | `+`/`-` (about the center) |
| `zoom_at(view, factor, px, py, sw, sh)` | wheel, double-click (point `(px, py)` stays put on screen) |
| `to_box(x0, y0, x1, y1, sw, sh)` | Shift+drag box |
| `at_level(view, level, sw, sh)` | number keys, menu (`None` if no such level) |
| `pan(view, dx, dy, sw, sh)` | arrows, PgUp/PgDn (`dx`, `dy` in view sizes) |
| `center_on(view, x, y, sw, sh)` | menu, thumbnail |
| `clamp(view, sw, sh)`, `upp_limits(sw, sh)` | everything above |

`ViewHistory.record(view, kind)` stores the view *before* a navigation.
Consecutive records with the same non-`None` `kind` are merged (that is
how a run of pan keys becomes one step). `kind=None` always makes a new
step. The canvas uses `"pan"`/`"zoom"` for keys, a distinct kind per jump,
and `None` for mouse gestures.

## Input routing

vispy's `PanZoomCamera` has its own key and mouse bindings. Both are
switched off in `_NotifyingPanZoomCamera`, except the trackpad pinch.
All input goes through `SlideCanvas`:

- **Keys** → `_on_key` → `navigation.action_for_key(name, text,
  modifiers)` → an action tuple → `SlideCanvas.navigate(action)`.
  Printable keys match on the typed character (so `+` works with or
  without Shift, on any layout); other keys match on the vispy key name.
- **Mouse** → `_on_mouse_press/move/release/double_click/wheel`. A press
  starts a `_Drag` with a mode decided by button, modifiers and
  position: `"inset"` (thumbnail), `"box"` (Shift+left), `"pan"`
  (left/middle), `"menu"` (right). A press only counts as a drag after
  4 px of movement, so a right-click with a slight wobble still opens the
  menu. Panning is computed from the press position and the view at
  press time, so rounding doesn't accumulate.
- **App keys** (quit, fullscreen, info, help) are `QAction` shortcuts on
  the main window (`app.py`), not canvas keys. An embedded canvas
  shouldn't be able to quit its host application.

The key map `navigation.CANVAS_KEYS` (and `MOUSE_HELP`, `app.APP_KEYS`)
is the single source for both dispatch and the help dialog.

# Shared state

`ViewerState` fields written by the canvas:

| Field | Type | Meaning |
|---|---|---|
| `viewport` | `(x0, y0, w, h)` | Visible rectangle, level-0 px |
| `canvas_size` | `(w, h)` | View size, physical px |
| `current_level` | `int` | Level being drawn |
| `cursor` | `(x, y)` or `None` | Mouse position, level-0 px |
| `cursor_value` | `tuple[float, ...]` or `None` | Pixel value under the cursor, one entry per channel |
| `cursor_value_level` | `int` or `None` | Level `cursor_value` came from |

Fields written by panels (not used by the canvas yet): `visible_groups`,
`selected_ids`, `hovered_id`. Derived: `state.scale` = `canvas_size[0] /
viewport[2]`. Each field has a signal, `state.events.<field>`. Assign
new (immutable) values; never mutate in place.

# Embedding `SlideCanvas`

```python
from vispy import app as vispy_app
vispy_app.use_app("pyside6")              # before creating any canvas
from compass.viewer.canvas import SlideCanvas

canvas = SlideCanvas(raster, state=None, tile_size=512,
                     cache_bytes=768 * 2**20, prefetch_margin=1,
                     max_workers=2, **vispy_canvas_kwargs)
```

| Member | Purpose |
|---|---|
| `canvas.native` | The Qt widget to put in a layout |
| `canvas.state` | The `ViewerState` (pass your own to share it between widgets) |
| `canvas.view` | vispy view box; annotation overlays go under `canvas.view.scene` (level-0 coordinates) |
| `canvas.navigator` | The `Navigator` for this slide |
| `canvas.get_view()` / `set_view(view)` | Read / move the camera. `set_view` doesn't touch the history |
| `canvas.go_to(view, kind=None)` | Move and record a history step |
| `canvas.navigate(action)` | Run a key-map action, e.g. `("level", 2)`, `("fit",)`, `("pan", 0.25, 0)`, `("back",)`. Returns `False` if it doesn't apply |
| `canvas.overview_visible` | Show/hide the thumbnail |
| `canvas.close()` | Stop the timer and the tile fetcher. Call it before the process exits, or reads still running will fail noisily |

Example: a panel that jumps to a given object. Here `bbox` would come
from `AnnotationStore.get_bboxes`:

```python
def show_object(bbox, pad=1.2):
    x0, y0, x1, y1 = bbox
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * pad, (y1 - y0) * pad
    sw, sh = canvas.state.canvas_size     # physical px
    canvas.go_to(canvas.navigator.to_box(cx - w / 2, cy - h / 2,
                                         cx + w / 2, cy + h / 2, sw, sh))
```

A full main-window example sharing one `ViewerState` with a dock panel is
in [`index.md`](index.md), *Embedding in your own window*.

## Adding a key binding

1. Add a row to `navigation.CANVAS_KEYS`:
   `("r", ("level", 2), "jump to level 2")`. A new action kind also needs
   a branch in `SlideCanvas.navigate`.
2. For window-level actions (dialogs, toggles of Qt widgets), add an
   `add_action(...)` call and a row in `APP_KEYS` in `app.py` instead.
3. The help dialog picks both up automatically. Add a case to
   `test_action_for_key` in `test_viewer_navigation.py`.

# Testing

**Headless (CI-able):**

```bash
uv run pytest src/compass/tests -q -k viewer
```

This covers tile geometry, caching, fetching, state signals, all view
arithmetic, history, key resolution, the help table and info-bar
formatting.

**GUI, on a machine with OpenGL and an X display.** These are scripts,
not part of the test suite:

- *Render frames*: build a `SlideCanvas`, move the camera from code,
  pump `QApplication.processEvents()`, then `canvas.render()` → numpy
  RGBA → PNG. This shows what is really drawn. Two of the bugs in
  [Pitfalls](#vispy-pitfalls-and-why-the-code-looks-like-this) were only
  visible this way.
- *Real input*: send X events with XTEST
  (`uv run --extra viewer --with python-xlib python script.py`):
  `xtest.fake_input(display, X.KeyPress, keycode)`,
  `X.ButtonPress` with buttons 1–3 (left, middle, right), 4/5 (wheel),
  8/9 (back/forward), `X.MotionNotify` for pointer moves. The window must
  be active (`win.activateWindow()` and pump events).
- **Don't trust `QTest.keyClick`** for this widget: it skips window
  shortcuts and depends on focus in ways real input doesn't, so `q`/`i`
  "failed" under QTest while working with a real keyboard.
- `QT_QPA_PLATFORM=offscreen` runs the tile/level logic but draws
  nothing.

# vispy pitfalls (and why the code looks like this)

Each of these caused a real bug while building the viewer. Keep them in
mind when touching `canvas.py`.

1. **`transform_change` fires only once.** `PanZoomCamera` mutates one
   `MatrixTransform` in place, and `Node.transform`'s setter ignores
   re-assigning the same object. A handler on
   `view.scene.events.transform_change` never runs after start-up, so the
   level never changes. Hook `camera.view_changed()` instead
   (`_NotifyingPanZoomCamera`).
2. **Images depth-test by default.** vispy's `Image` uses the
   `'translucent'` GL preset, which has `depth_test=True`. All layers sit
   at z = 0, so the *first-drawn* layer wins, regardless of `order`. The
   coarse backdrop hid every finer tile. Every slide layer calls
   `_layer_by_order()` (`depth_test=False`); do the same for new overlay
   visuals that should stack by `order`.
3. **The camera binds keys and mouse.** `BaseCamera` maps `Backspace` to
   `reset()`, back to the construction rect `(0,0)–(1,1)`, which looks
   like a jump to a tiny corner. The camera listens to the whole canvas,
   including the thumbnail's camera. Both cameras override
   `viewbox_key_event` / `viewbox_mouse_event`.
4. **`camera.rect` ≠ what's on screen.** With `aspect=1`, the drawn rect
   is widened to the window's aspect ratio. Use the visible rect
   (`_visible_rect()`) for anything that depends on what the user sees.
5. **`keys="interactive"` closes the canvas on Esc**, leaving an empty
   main window. The canvas is created with `keys=None`.
6. **Single-channel contrast is per tile.** With the default
   `clim="auto"`, vispy uses the data type's full range for 3- and
   4-channel images (so RGB is consistent across tiles), but computes the
   range from each image's own min/max for single-channel data. Each
   grayscale tile is then stretched differently. See the next section.

# Limitations and next steps

- **Annotation overlays** (architecture §7 step 4) are not built: live
  vispy markers/polygons below a count threshold, datashader texture
  above it. Overlays attach under `canvas.view.scene` in level-0
  coordinates and must disable depth testing (pitfall 2).
- **Selection and panels** (step 5): single left-click and `Esc` are kept
  free for selecting objects / clearing the selection.
  `selected_ids`, `hovered_id` and `visible_groups` exist in
  `ViewerState` but are not drawn yet.
- **Display of non-RGB data.** Tiles are uploaded as stored, with
  vispy's default `clim="auto"` (pitfall 6). 3-/4-channel slides display
  consistently over the full range of their data type: right for 8-bit
  RGB, dark for 16-bit data that doesn't use the full range.
  Single-channel slides (8- or 16-bit) are contrast-stretched **per
  tile**, which shows as tile seams. Only 1, 3 or 4 channels can be
  uploaded as one texture at all. Multiplex slides need a channel → color
  mapping and a fixed, slide-wide contrast window; the fix for
  single-channel slides is to pass that window as `clim` to every tile
  visual. The info bar already reports per-channel values for such
  slides.
- **Thumbnail placement** is fixed at the top-right corner, at most
  220 px. It shrinks on small windows.
- **Settings** (wheel sensitivity, zoom limit, thumbnail size) are
  module constants (`navigation.MAX_ZOOM`, `WHEEL_NOTCHES_PER_LEVEL`,
  `FIT_MARGIN`, `inset_rect(max_size=…)`), not user preferences yet.
