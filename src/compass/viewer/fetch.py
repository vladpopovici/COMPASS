# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Background tile fetching — pure threading, no GUI dependencies.

Zarr shard reads are independent byte-range reads with no global lock
(the reason Zarr was chosen over HDF5, docs/architecture.md section 2),
so a small thread pool genuinely parallelizes tile decode. The fetcher
only reads and reports: the ``on_tile`` callback runs on a worker thread,
and the GUI layer is responsible for marshalling back to its own thread
(the vispy canvas drains a queue on a timer).
"""

import logging
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from .tiling import TileKey

logger = logging.getLogger(__name__)


#####
class TileFetcher:
    """Deduplicating background reader of tiles.

    Args:
        read_tile: blocking function ``TileKey -> ndarray`` (typically a
            closure over ``TileGrid.tile_bounds_px`` + ``raster.get_region_px``).
        on_tile: called as ``on_tile(key, array)`` from a worker thread for
            every successfully read tile.
        max_workers: reader thread count.
    """

    def __init__(self,
                 read_tile: Callable[[TileKey], np.ndarray],
                 on_tile: Callable[[TileKey, np.ndarray], None],
                 max_workers: int = 2):
        self._read_tile = read_tile
        self._on_tile = on_tile
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="tile-fetch")
        self._inflight: dict[TileKey, Future] = {}
        self._lock = threading.Lock()
        self._closed = False

    def request(self, keys: Iterable[TileKey]) -> int:
        """Queue reads for ``keys`` (in the given order — pass them
        priority-first). Keys already queued or running are skipped.
        Returns the number of newly queued reads."""
        n = 0
        with self._lock:
            if self._closed:
                return 0
            for key in keys:
                if key in self._inflight:
                    continue
                self._inflight[key] = self._pool.submit(self._work, key)
                n += 1
        return n

    def _work(self, key: TileKey) -> None:
        try:
            arr = self._read_tile(key)
            self._on_tile(key, arr)
        except Exception:
            logger.exception("tile read failed: %s", key)
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def cancel_except(self, keep: Iterable[TileKey]) -> int:
        """Cancel queued (not yet running) reads whose key is not in
        ``keep`` — call on viewport change to drop stale prefetches.
        Returns the number of cancelled reads."""
        keep = set(keep)
        n = 0
        with self._lock:
            for key, fut in list(self._inflight.items()):
                if key not in keep and fut.cancel():
                    self._inflight.pop(key)
                    n += 1
        return n

    @property
    def pending(self) -> int:
        """Number of reads queued or running."""
        with self._lock:
            return len(self._inflight)

    def close(self) -> None:
        """Cancel queued reads and stop the pool (running reads finish)."""
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
##
