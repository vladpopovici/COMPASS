# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

import threading
import time

import numpy as np
import pytest

from compass.viewer import TileFetcher, TileKey

K = [TileKey(0, i, 0) for i in range(8)]


def wait_until(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            pytest.fail("timeout waiting for condition")
        time.sleep(0.005)


def test_fetch_delivers_all_tiles():
    reads, done = [], []

    def read(key):
        reads.append(key)
        return np.full((2, 2), key.ix, np.uint8)

    fetcher = TileFetcher(read, lambda k, a: done.append((k, int(a[0, 0]))),
                          max_workers=2)
    try:
        assert fetcher.request(K[:4]) == 4
        wait_until(lambda: len(done) == 4)
        assert {k for k, _ in done} == set(K[:4])
        assert all(v == k.ix for k, v in done)
        wait_until(lambda: fetcher.pending == 0)
    finally:
        fetcher.close()


def test_fetch_deduplicates_inflight():
    gate = threading.Event()
    done = []

    def read(key):
        gate.wait(5)
        return np.zeros(1, np.uint8)

    fetcher = TileFetcher(read, lambda k, a: done.append(k), max_workers=1)
    try:
        assert fetcher.request([K[0], K[0], K[1]]) == 2
        assert fetcher.request([K[0], K[1]]) == 0  # already in flight
        gate.set()
        wait_until(lambda: len(done) == 2)
    finally:
        fetcher.close()


def test_cancel_except_drops_queued_only():
    started, done = [], []
    gate = threading.Event()

    def read(key):
        started.append(key)
        gate.wait(5)
        return np.zeros(1, np.uint8)

    fetcher = TileFetcher(read, lambda k, a: done.append(k), max_workers=1)
    try:
        fetcher.request(K[:4])                  # K0 starts, K1-K3 queued
        wait_until(lambda: len(started) == 1)
        n = fetcher.cancel_except([K[0], K[2]])  # K1, K3 cancelled
        assert n == 2
        gate.set()
        wait_until(lambda: set(done) == {K[0], K[2]})
        time.sleep(0.05)
        assert K[1] not in done and K[3] not in done
    finally:
        fetcher.close()


def test_read_errors_are_logged_not_raised(caplog):
    done = []

    def read(key):
        if key == K[0]:
            raise RuntimeError("boom")
        return np.zeros(1, np.uint8)

    fetcher = TileFetcher(read, lambda k, a: done.append(k), max_workers=1)
    try:
        fetcher.request(K[:2])
        wait_until(lambda: fetcher.pending == 0)
        assert done == [K[1]]
        assert any("tile read failed" in r.message for r in caplog.records)
    finally:
        fetcher.close()


def test_closed_fetcher_refuses_requests():
    fetcher = TileFetcher(lambda k: np.zeros(1), lambda k, a: None)
    fetcher.close()
    assert fetcher.request(K[:2]) == 0
    fetcher.close()  # idempotent
