# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Keyboard navigation for the slide canvas — pure logic, no GUI
dependencies, fully testable headless.

A view is described by its center and its zoom (:class:`View`), not by a
rectangle, so it survives window resizes unchanged. The canvas converts a
:class:`View` to a camera rectangle for the current screen size.

The key map (:data:`CANVAS_KEYS`) is the single source of truth for both
dispatch (:func:`action_for_key`) and the help dialog.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

#: zoom-in limit: screen pixels per level-0 pixel (8x past full resolution)
MAX_ZOOM = 8.0
#: empty border around the slide when fitting it to the window
FIT_MARGIN = 0.02
#: mouse-wheel notches per pyramid level (zoom x2)
WHEEL_NOTCHES_PER_LEVEL = 4


@dataclass(frozen=True, slots=True)
class View:
    """A camera position.

    Attributes:
        cx, cy: view center in level-0 pixels.
        upp: zoom, as level-0 pixels per (physical) screen pixel. 1.0 shows
            level 0 at 1:1; 2.0 shows level 1 at 1:1, etc.
    """
    cx: float
    cy: float
    upp: float

    def rect(self, screen_w: float, screen_h: float) -> tuple[float, float, float, float]:
        """(x0, y0, width, height) in level-0 pixels for a screen size."""
        w, h = self.upp * screen_w, self.upp * screen_h
        return self.cx - w / 2, self.cy - h / 2, w, h

    @classmethod
    def from_rect(cls, rect: tuple[float, float, float, float],
                  screen_w: float) -> "View":
        x0, y0, w, h = rect
        return cls(x0 + w / 2, y0 + h / 2, w / screen_w)
##


#####
class Navigator:
    """View arithmetic for one slide shown on one screen.

    Args:
        slide_w, slide_h: level-0 slide extent in pixels.
        downsample: ``level -> level-0 pixels per level pixel``
            (``TileGrid.downsample``).
        nlevels: number of pyramid levels.
        max_zoom: zoom-in limit, screen pixels per level-0 pixel.
    """

    def __init__(self, slide_w: float, slide_h: float,
                 downsample: Callable[[int], float], nlevels: int,
                 max_zoom: float = MAX_ZOOM):
        self.slide_w = float(slide_w)
        self.slide_h = float(slide_h)
        self._downsample = downsample
        self.nlevels = int(nlevels)
        self.max_zoom = float(max_zoom)

    # --- limits ---

    def fit(self, screen_w: float, screen_h: float) -> View:
        """Whole slide in the window, centered."""
        upp = max(self.slide_w / screen_w, self.slide_h / screen_h) * (1 + 2 * FIT_MARGIN)
        return View(self.slide_w / 2, self.slide_h / 2, upp)

    def upp_limits(self, screen_w: float, screen_h: float) -> tuple[float, float]:
        """(min, max) allowed ``upp``: MAX_ZOOM in, fit-to-window out."""
        lo = 1.0 / self.max_zoom
        return lo, max(lo, self.fit(screen_w, screen_h).upp)

    def clamp(self, view: View, screen_w: float, screen_h: float) -> View:
        """Clamp zoom to :meth:`upp_limits`; per axis, center the slide if
        the view is larger than it, else keep the view center on the slide."""
        lo, hi = self.upp_limits(screen_w, screen_h)
        upp = min(max(view.upp, lo), hi)

        def axis(c, slide, screen):
            if upp * screen >= slide:
                return slide / 2
            return min(max(c, 0.0), slide)

        return View(axis(view.cx, self.slide_w, screen_w),
                    axis(view.cy, self.slide_h, screen_h), upp)

    # --- actions (all return clamped views) ---

    def zoom(self, view: View, factor: float, screen_w: float, screen_h: float) -> View:
        """Zoom about the view center; ``factor > 1`` zooms in."""
        return self.clamp(replace(view, upp=view.upp / factor), screen_w, screen_h)

    def zoom_at(self, view: View, factor: float, px: float, py: float,
                screen_w: float, screen_h: float) -> View:
        """Zoom by ``factor`` (> 1 zooms in) keeping the level-0 point
        (px, py) at the same place on screen (wheel / double-click zoom)."""
        lo, hi = self.upp_limits(screen_w, screen_h)
        upp = min(max(view.upp / factor, lo), hi)
        r = upp / view.upp
        return self.clamp(View(px + (view.cx - px) * r,
                               py + (view.cy - py) * r, upp),
                          screen_w, screen_h)

    def to_box(self, x0: float, y0: float, x1: float, y1: float,
               screen_w: float, screen_h: float) -> View | None:
        """View fitting the level-0 box with corners (x0, y0), (x1, y1);
        None for a degenerate box."""
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w <= 0 or h <= 0:
            return None
        return self.clamp(View((x0 + x1) / 2, (y0 + y1) / 2,
                               max(w / screen_w, h / screen_h)),
                          screen_w, screen_h)

    def center_on(self, view: View, x: float, y: float,
                  screen_w: float, screen_h: float) -> View:
        return self.clamp(View(x, y, view.upp), screen_w, screen_h)

    def at_level(self, view: View, level: int,
                 screen_w: float, screen_h: float) -> View | None:
        """Show ``level`` at 1:1 around the current center, or None if the
        slide has no such level."""
        if not 0 <= level < self.nlevels:
            return None
        return self.clamp(replace(view, upp=self._downsample(level)), screen_w, screen_h)

    def pan(self, view: View, dx: float, dy: float,
            screen_w: float, screen_h: float) -> View:
        """Pan by (dx, dy) view sizes (e.g. 0.25 = a quarter of the view)."""
        return self.clamp(View(view.cx + dx * view.upp * screen_w,
                               view.cy + dy * view.upp * screen_h,
                               view.upp), screen_w, screen_h)
##


#####
class ViewHistory:
    """Back/forward history of views, browser-style.

    Call :meth:`record` with the view *before* each navigation. Repeats of
    the same kind of action (e.g. holding an arrow key) collapse into one
    history step; ``kind=None`` is never collapsed (one step per call).
    """

    def __init__(self, max_len: int = 100):
        self._back: list[View] = []
        self._forward: list[View] = []
        self._last_kind: str | None = None
        self._max_len = int(max_len)

    def record(self, current: View, kind: str | None = None) -> None:
        if kind is not None and kind == self._last_kind and self._back:
            return
        self._last_kind = kind
        if not self._back or self._back[-1] != current:
            self._back.append(current)
            del self._back[:-self._max_len]
        self._forward.clear()

    def back(self, current: View) -> View | None:
        if not self._back:
            return None
        self._forward.append(current)
        self._last_kind = None
        return self._back.pop()

    def forward(self, current: View) -> View | None:
        if not self._forward:
            return None
        self._back.append(current)
        self._last_kind = None
        return self._forward.pop()

    @property
    def can_back(self) -> bool:
        return bool(self._back)

    @property
    def can_forward(self) -> bool:
        return bool(self._forward)
##


# ---------------------------------------------------------------------------
# overview inset (thumbnail) geometry
# ---------------------------------------------------------------------------

def inset_rect(canvas_w: float, canvas_h: float,
               slide_w: float, slide_h: float,
               max_size: float = 220, margin: float = 10
               ) -> tuple[float, float, float, float]:
    """(x, y, w, h) of the overview thumbnail in canvas pixels: top-right
    corner, slide aspect, longest side ``max_size`` (shrunk to fit small
    canvases)."""
    size = max(16.0, min(max_size, (canvas_w - 2 * margin) / 2,
                         (canvas_h - 2 * margin) / 2))
    s = size / max(slide_w, slide_h)
    w, h = slide_w * s, slide_h * s
    return canvas_w - margin - w, margin, w, h


def inset_to_slide(px: float, py: float,
                   inset: tuple[float, float, float, float],
                   slide_w: float, slide_h: float) -> tuple[float, float] | None:
    """Canvas pixel -> level-0 slide point, if (px, py) is inside the
    thumbnail (clamped to the slide); else None."""
    x, y, w, h = inset
    if not (x <= px <= x + w and y <= py <= y + h):
        return None
    return ((px - x) / w * slide_w, (py - y) / h * slide_h)


# ---------------------------------------------------------------------------
# key map
# ---------------------------------------------------------------------------

#: An action is a tuple: ("zoom", factor), ("level", n), ("fit",),
#: ("pan", dx, dy), ("back",), ("forward",).
Action = tuple

#: (keys shown in help, action, description). Printable keys match on the
#: typed character, others on the vispy key name; "Shift+"/"Alt+" prefixes
#: are modifiers.
CANVAS_KEYS: list[tuple[str, Action, str]] = [
    ("+", ("zoom", 2.0), "zoom in x2"),
    ("=", ("zoom", 2.0), "zoom in x2"),
    ("-", ("zoom", 0.5), "zoom out x2"),
    *[(str(n), ("level", n), f"pyramid level {n} at 1:1" + (" (full resolution)" if n == 0 else ""))
      for n in range(10)],
    ("Home", ("fit",), "fit slide to window"),
    ("f", ("fit",), "fit slide to window"),
    ("Left", ("pan", -0.25, 0.0), "pan left 1/4 view"),
    ("Right", ("pan", 0.25, 0.0), "pan right 1/4 view"),
    ("Up", ("pan", 0.0, -0.25), "pan up 1/4 view"),
    ("Down", ("pan", 0.0, 0.25), "pan down 1/4 view"),
    ("Shift+Left", ("pan", -1.0, 0.0), "pan left one screen"),
    ("Shift+Right", ("pan", 1.0, 0.0), "pan right one screen"),
    ("Shift+Up", ("pan", 0.0, -1.0), "pan up one screen"),
    ("Shift+Down", ("pan", 0.0, 1.0), "pan down one screen"),
    ("PageUp", ("pan", 0.0, -1.0), "pan up one screen"),
    ("PageDown", ("pan", 0.0, 1.0), "pan down one screen"),
    ("Backspace", ("back",), "previous view"),
    ("Alt+Left", ("back",), "previous view"),
    ("Alt+Right", ("forward",), "next view"),
]

#: mouse bindings, for the help dialog only (handled in SlideCanvas)
MOUSE_HELP: list[tuple[str, str]] = [
    ("Left-drag, middle-drag", "pan"),
    ("Wheel / two-finger scroll", "zoom at cursor (4 notches = 1 level)"),
    ("Double-click", "zoom in x2 at cursor"),
    ("Shift+double-click", "zoom out x2 at cursor"),
    ("Shift+left-drag", "zoom to box"),
    ("Mouse back / forward", "previous / next view"),
    ("Right-click", "menu: center here, zoom to level, fit, copy coordinates"),
    ("Click / drag in overview", "jump to that part of the slide"),
]

_MODIFIERS = ("Shift", "Alt", "Control", "Meta")


def _parse(spec: str) -> tuple[frozenset[str], str]:
    if spec == "+":
        return frozenset(), "+"
    *mods, key = spec.split("+")
    return frozenset(mods), key


_LOOKUP: dict[tuple[frozenset[str], str], Action] = {}
for _spec, _action, _ in CANVAS_KEYS:
    _LOOKUP[_parse(_spec)] = _action


def action_for_key(name: str | None, text: str,
                   modifiers: frozenset[str] | set[str] = frozenset()) -> Action | None:
    """Resolve a key press to an action.

    Args:
        name: vispy key name (``event.key.name``), e.g. "Left", "PageUp".
        text: typed character (``event.text``), e.g. "+", "0".
        modifiers: names of held modifiers ("Shift", "Alt", "Control", "Meta").
    """
    mods = frozenset(m for m in modifiers if m in _MODIFIERS)
    # typed characters already include Shift (e.g. "+" on many layouts)
    if text and text.isprintable() and not text.isspace():
        hit = _LOOKUP.get((mods - {"Shift"}, text))
        if hit is not None:
            return hit
    if name:
        return _LOOKUP.get((mods, name))
    return None


def help_rows() -> list[tuple[str, str]]:
    """(keys, description) rows for the help dialog, in key-map order;
    keys with the same description share a row, digits 1-9 are collapsed."""
    rows: dict[str, list[str]] = {}
    for spec, action, desc in CANVAS_KEYS:
        if action[0] == "level" and action[1] > 0:
            spec, desc = "1 ... 9", "pyramid level n at 1:1 (if the slide has it)"
        keys = rows.setdefault(desc, [])
        if spec not in keys:
            keys.append(spec)
    return [(", ".join(keys), desc) for desc, keys in rows.items()]
