#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022-2023 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------
__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.2

__all__ = ['WindowSampler', 'SlidingWindowSampler', 'RandomWindowSampler', 'Tiler']

from abc import ABC, abstractmethod
import shapely.geometry as shg
import numpy as np
from collections.abc import Sequence
from functools import cached_property, lru_cache

from .core import ImageShape


class WindowSampler(ABC, Sequence):
    """
    Defines an interface for an image sampler that returns rectangular
    regions from the image.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(self):
        """Reset the exploration, next call to next() will start from the
        initial conditions.
        """
        pass

    @abstractmethod
    def last(self):
        """Go to the last position and return it."""
        pass

    @abstractmethod
    def next(self):
        """Go to the next position."""
        pass

    @abstractmethod
    def prev(self):
        """Go to the previous position."""
        pass

    @abstractmethod
    def here(self):
        """Returns the current position, does not change it."""
        pass

    @property
    @abstractmethod
    def total_steps(self):
        """Returns the total number of steps to iterate over all positions
        in the image, according to the specific schedule.
        """
        pass

    @property
    @abstractmethod
    def bounding_box(self):
        """Return the bounding box for all windows."""
        pass

    @staticmethod
    def _corners_to_poly(x0, y0, x1, y1):
        """Returns a Shapely Polygon with all four vertices of the window
        defined by (x0,y0) -> (x1,y1).
        """
        return shg.Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])

    @staticmethod
    def _check_window(x0: int, y0: int, x1: int, y1: int,
                      width: int, height: int, clip: bool = True) -> tuple:
        """Checks whether the coordinates of the window are valid and, eventually
        (only if clip is True), truncates the window to fit the image extent
        given by width and height.

        Args:
            x0, y0 : int
                Top-left corner of the window.
            x1, y1 : int
                Bottom-right corner of the window.
            width, height : int
                Image shape.
            clip : bool
                Whether the window should be clipped to image boundary.

        Return:
            a tuple (x0, y0, x1, y1) of window vertices or None if the
            window is not valid (e.g. negative coordinates).
        """
        if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
            return None, None, None, None
        if x0 >= width or y0 >= height:
            return None, None, None, None

        if clip:
            x1 = min(x1, width)
            y1 = min(y1, height)

        return x0, y0, x1, y1

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __prev__(self):
        return self.prev()

    def __len__(self):
        return self.total_steps()


class SlidingWindowSampler(WindowSampler):
    """
    A sliding window image sampler. It successively returns the coordinates
    of the sliding window as a tuple (x0, y0, x1, y1).

    Args:
        image_shape : ImageShape ('width', 'height')
            Image shape (img.shape).
        w_size : ImageShape ('width', 'height')
            Window size as a pair of width and height values.
        start : tuple (x0, y0)
            Top left corner of the first window. Defaults to (0,0).
        step : tuple (x_step, y_step)
            Step size for the sliding window, as a pair of horizontal
            and vertical steps. Defaults to (1,1).
        poly : shapely.geometry.Polygon
            (if not None) Defines the region within which the windows will be generated.
            Note that this may be a union of polygons, allowing for rgeater flexibility.
        nv_inside : int
            number of corners/vertices of the the window required to be inside the
            polygon defining the region. This relaxes the constraint that whole window
            must lie within the polygon. Must be between 1 and 4.
        only_full_windows: bool
            if True, only fully-sized windows will be returned, no clipping at the end
            of the rows/columns will be applied._
    """

    def __init__(self, image_shape: ImageShape, w_size: ImageShape,
                 start: tuple = (0, 0), step=(1, 1), poly: shg.Polygon = None,
                 nv_inside: int = 4, only_full_windows: bool=True):
        self._image_shape = (image_shape.width, image_shape.height)
        self._w_size = (w_size.width, w_size.height)
        self._start = start
        self._step = step
        self._k = 0
        self._poly = poly

        nv_inside = max(1, min(nv_inside, 4))  # >=1 && <=4

        img_w, img_h = self._image_shape

        if self._w_size[0] < 2 or self._w_size[1] < 2:
            raise ValueError('Window size too small.')

        if img_w < start[0] + self._w_size[0] or img_h < start[1] + self._w_size[1]:
            raise ValueError('Start position and/or window size out of image.')

        x, y = np.meshgrid(np.arange(start[0], img_w - self._w_size[0] + 1, step[0]),
                           np.arange(start[1], img_h - self._w_size[1] + 1, step[1]))

        tmp_top_left_corners = [p for p in zip(x.reshape((-1,)).tolist(),
                                               y.reshape((-1,)).tolist())]

        #print(f"Sampler: total possible windows: {len(tmp_top_left_corners)}")
        if self._poly is None:
            self._top_left_corners = tmp_top_left_corners
        else:
            # need to filter out regions outside the Polygon
            self._top_left_corners = []
            for x0, y0 in tmp_top_left_corners:
                x1 = x0 + self._w_size[0]
                y1 = y0 + self._w_size[1]
                t = WindowSampler._check_window(x0, y0, x1, y1,
                                                img_w, img_h, clip=not only_full_windows)
                if t is None:
                    continue
                x0, y0, x1, y1 = t
                w = [int(shg.Point(p).within(self._poly)) for p in [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]]
                if np.array(w).sum() >= nv_inside:
                    self._top_left_corners.append((x0, y0))
        #print(f"Sampler: no. of kept windows: {len(self._top_left_corners)}")

        super().__init__()

        return

    @cached_property
    def total_steps(self):
        return len(self._top_left_corners)

    @cached_property
    def bounding_box(self):
        if len(self._top_left_corners) == 0:
            return 0, 0, 0, 0

        x_min, y_min = self._top_left_corners[0]
        x_max, y_max = x_min + self._w_size[0], y_min + self._w_size[1]

        for k in range(len(self._top_left_corners)):
            p = self._top_left_corners[k]
            x_min = min(x_min, p[0])
            y_min = min(y_min, p[1])
            x_max = max(x_max, p[0] + self._w_size[0])
            y_max = max(y_max, p[1] + self._w_size[1])

        return x_min, y_min, x_max, y_max

    def reset(self):
        self._k = 0

    def here(self):
        if 0 <= self._k < self.total_steps:
            x0, y0 = self._top_left_corners[self._k]
            x1 = min(x0 + self._w_size[0], self._image_shape[0])
            y1 = min(y0 + self._w_size[1], self._image_shape[1])

            return x0, y0, x1, y1
        raise RuntimeError("Position outside bounds")

    def last(self):
        if self.total_steps > 0:
            self._k = self.total_steps - 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise RuntimeError("Empty iterator")

    def next(self):
        if self._k < self.total_steps:
            x0, y0, x1, y1 = self.here()
            self._k += 1
            return x0, y0, x1, y1
        else:
            raise StopIteration()

    def prev(self):
        if self._k >= 1:
            self._k -= 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise StopIteration()

    def __getitem__(self, item):
        if 0 <= item < self.total_steps:
            x0, y0 = self._top_left_corners[item]
            x1 = min(x0 + self._w_size[0], self._image_shape[0])
            y1 = min(y0 + self._w_size[1], self._image_shape[1])

            return x0, y0, x1, y1
        raise RuntimeError("Position outside bounds")
##-

####
class Tiler(WindowSampler):
    """
    Returns tiles from an image. In addition to a SlidingWindowSampler interface,
    it also provides a method to return a tile referenced by its (row, column)
    coordinates. Further, the tiler does not allow masking the image and its
    default step equals the size of the tile.
    """
    def __init__(self, image_shape: ImageShape, tile_size: ImageShape, overlap: int = 0):
        """A tiler returns tiles within the bounds given by <image_shape>. The size of
        the tiles is given by <tile_size>, while the overlap between neighboring tiles is
        <overlap> pixels.

        Args:
            image_shape (ImageShape): Image shape/bounds for the tiles.
            tile_size (ImageShape): Size of the tiles.
            overlap (int): Overlap between neighboring tiles.
        """
        self._image_shape = image_shape
        self._tile_size = tile_size
        self._overlap = overlap

        # top-left corners (in pixels) for each tile
        self._t_x0 = np.arange(0, image_shape.width + 1, tile_size.width - overlap)
        self._t_y0 = np.arange(0, image_shape.height + 1, tile_size.height - overlap)
        self._n_columns = len(self._t_x0)
        self._n_rows = len(self._t_y0)

        self._k = 0

    @cached_property
    def bounding_box(self):
        return 0, 0, self._image_shape.width, self._image_shape.height

    def reset(self):
        self._k = 0

    @cached_property
    def n_horiz_tiles(self):
        return self._n_columns

    @cached_property
    def n_vert_tiles(self):
        return self._n_rows

    @cached_property
    def total_steps(self):
        return self._n_columns * self._n_rows

    @lru_cache(maxsize=128)
    def tile_at(self, i: int, j: int):
        """Returns the coordinates of the tile at row i and column j."""
        if i < 0 or i >= self._n_rows or j < 0 or j >= self._n_columns:
            raise RuntimeError("Tile index out of bounds")
        x0 = self._t_x0[j]
        y0 = self._t_y0[i]
        x1 = min(self._image_shape.width, x0 + self._tile_size.width)
        y1 = min(self._image_shape.height, y0 + self._tile_size.height)

        return x0, y0, x1, y1

    def here(self):
        i = self._k // self._n_columns
        j = self._k % self._n_columns

        return self.tile_at(i, j)

    def last(self):
        if self.total_steps > 0:
            self._k = self.total_steps - 1
            return self.tile_at(self._n_rows - 1, self._n_columns - 1)
        raise RuntimeError("Empty iterator")

    def next(self):
        if self._k < self.total_steps:
            x0, y0, x1, y1 = self.here()
            self._k += 1
            return x0, y0, x1, y1
        else:
            raise StopIteration()

    def prev(self):
        if self._k >= 1:
            self._k -= 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise StopIteration()

    def __getitem__(self, item):
        if 0 <= item < self.total_steps:
            return self.tile_at(item // self._n_columns, item % self._n_columns)
        raise RuntimeError("Position outside bounds")


##-
class RandomWindowSampler(WindowSampler):
    """
    A random window image sampler. It returns a sequence of random window coordinates
    (x0, y0, x1, y1) within the image.

    Args:
        image_shape : ImageShape ('width', 'height')
            Image shape (img.shape).
        w_size : ImageShape (width, height)
            Window size as a pair of width and height values.
        n : int
            Number of windows to return.
        poly : shapely.geometry.Polygon
            (if not None) Defines the region within which the windows will be generated.
            Note that this may be a union of polygons, allowing for rgeater flexibility.
        rng_seed : int or None
            random number generator seed for initialization in a known state. If None,
            the seed is set by the system.
        nv_inside : int
            number of corners/vertices of the the window required to be inside the
            polygon defining the region. This relaxes the constraint that whole window
            must lie within the polygon. Must be between 1 and 4.
        only_full_windows: bool
            if True, only fully-sized windows will be returned, no clipping at the end
            of the rows/columns will be applied._
    """

    def __init__(self, image_shape: ImageShape, w_size: ImageShape, n: int,
                 poly: shg.Polygon = None, rng_seed: int = 42, nv_inside: int = 4,
                 only_full_windows: bool=True):
        self._image_shape = (image_shape.width, image_shape.height)
        self._w_size = (w_size.width, w_size.height)
        self._poly = poly
        self._n = n
        self._k = 0
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(rng_seed)

        nv_inside = max(1, min(nv_inside, 4))  # >=1 && <=4
        img_w, img_h = self._image_shape

        if self._w_size[0] < 2 or self._w_size[1] < 2:
            raise ValueError('Window size too small.')

        if img_w < self._w_size[0] or img_h < self._w_size[1]:
            raise ValueError('Window size larger than image.')

        if self._poly is None:
            self._top_left_corners = []
            k = 0
            while k < self._n:
                x0 = self._rng.integers(low=0, high=img_w - self._w_size[0], size=1)[0]
                y0 = self._rng.integers(low=0, high=img_h - self._w_size[1], size=1)[0]
                x1 = x0 + self._w_size[0]
                y1 = y0 + self._w_size[1]
                t = WindowSampler._check_window(x0, y0, x1, y1, img_w, img_h,
                                                clip=not only_full_windows)
                if t is None:
                    continue
                x0, y0, x1, y1 = t
                # finally, a valid window
                self._top_left_corners.append((x0, y0))
                k += 1
        else:
            # need to filter out regions outside the Polygon
            k = 0
            self._top_left_corners = []
            while k < self._n:
                x0 = self._rng.integers(low=0, high=img_w - self._w_size[0], size=1)[0]
                y0 = self._rng.integers(low=0, high=img_h - self._w_size[1], size=1)[0]
                x1 = x0 + self._w_size[0]
                y1 = y0 + self._w_size[1]
                t = WindowSampler._check_window(x0, y0, x1, y1, img_w, img_h, clip=True)
                if t is None:
                    continue
                x0, y0, x1, y1 = t
                w = [int(shg.Point(p).within(self._poly)) for p in [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]]
                if np.array(w).sum() >= nv_inside:
                    self._top_left_corners.append((x0, y0))
                    k += 1

        super().__init__()

        return

    @cached_property
    def total_steps(self):
        return self._n

    @cached_property
    def bounding_box(self):
        if len(self._top_left_corners) == 0:
            return 0, 0, 0, 0

        x_min, y_min = self._top_left_corners[0]
        x_max, y_max = x_min + self._w_size[0], y_min + self._w_size[1]

        for k in range(len(self._top_left_corners)):
            p = self._top_left_corners[k]
            x_min = min(x_min, p[0])
            y_min = min(y_min, p[1])
            x_max = max(x_max, p[0] + self._w_size[0])
            y_max = max(y_max, p[1] + self._w_size[1])

        return x_min, y_min, x_max, y_max

    def reset(self):
        self._k = 0

    def here(self):
        if 0 <= self._k < self.total_steps:
            x0, y0 = self._top_left_corners[self._k]
            # bounds where checked in the constructor
            x1 = x0 + self._w_size[0]
            y1 = y0 + self._w_size[1]

            return x0, y0, x1, y1

        raise RuntimeError("Position outside bounds")

    def last(self):
        if self.total_steps > 0:
            self._k = self.total_steps - 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise RuntimeError("Empty iterator")

    def next(self):
        if self._k < self.total_steps:
            x0, y0, x1, y1 = self.here()
            self._k += 1
            return x0, y0, x1, y1
        else:
            raise StopIteration()

    def prev(self):
        if self._k >= 1:
            self._k -= 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise StopIteration()

    def __getitem__(self, item):
        if 0 <= item < self.total_steps:
            x0, y0 = self._top_left_corners[item]
            x1 = x0 + self._w_size[0]
            y1 = y0 + self._w_size[1]

            return x0, y0, x1, y1
        raise RuntimeError("Position outside bounds")
##
