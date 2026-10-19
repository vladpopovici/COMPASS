# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Tests for partial-pyramid imports (start level) and the import record:
compass.connector.plan_import / import_record / crop_scale_transform, the
start_level of wsi2zarr (fake pyvips/WSI, see test_connector_ingest) and
ometiff2zarr (real OME-TIFF), and the src/tools/wsi_import.py CLI
(--start-level/--start-mpp, --show-meta, --meta-format).
"""

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from compass.connector import (TRANSFORM_TYPES, box_downsample,
                               closest_level_for_mpp, crop_scale_transform,
                               plan_import, wsi2zarr)
from compass.connector.ingest import _pick_source_level
from compass.core import ImageShape, ZarrRaster
from compass.tests.test_connector_ingest import FakeVipsImage, FakeWSI, fake_slide  # noqa: F401

TOOL = Path(__file__).resolve().parents[2] / "tools" / "wsi_import.py"


def _apply(matrix, x, y):
    (a, b, c), (d, e, f) = matrix
    return a * x + b * y + c, d * x + e * y + f


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------

def test_plan_import_level0_is_plain_plan():
    p = plan_import(1500, 1200, (8, 16, 1400, 1100), 0)
    assert p["downsample"] == 1
    assert p["crop"] == (8, 16, 1400, 1100)
    assert [(lv["w"], lv["h"]) for lv in p["levels"]] == [(1400, 1100), (700, 550), (350, 275)]
    assert p["levels"] == p["planned"]


def test_plan_import_start_level_snaps_crop():
    p = plan_import(1500, 1200, (9, 17, 1400, 1100), 1)
    assert p["downsample"] == 2
    # origin snapped down to the level-1 grid; extent = planned level 1 x 2
    assert p["crop"] == (8, 16, 1400, 1100)
    assert p["requested_crop"] == (9, 17, 1400, 1100)
    assert [(lv["w"], lv["h"]) for lv in p["levels"]] == [(700, 550), (350, 275)]
    assert len(p["planned"]) == 3


def test_plan_import_rejects_missing_level_and_bad_crop():
    with pytest.raises(ValueError, match="levels 0..2"):
        plan_import(1500, 1200, (0, 0, 1400, 1100), 3)
    with pytest.raises(ValueError):
        plan_import(1500, 1200, (0, 0, 1400, 1100), -1)
    with pytest.raises(ValueError, match="outside"):
        plan_import(1000, 1000, (500, 0, 600, 100), 0)


def test_closest_level_for_mpp():
    # levels: 0.25, 0.5, 1, 2 um/px
    assert closest_level_for_mpp(0.25, 1.0, 4) == 2
    assert closest_level_for_mpp(0.25, 0.9, 4) == 2
    assert closest_level_for_mpp(0.25, 0.1, 4) == 0
    assert closest_level_for_mpp(0.25, 100, 4) == 3
    # sqrt(0.5 * 1): equally far by ratio -> the finer level
    assert closest_level_for_mpp(0.25, np.sqrt(0.5), 4) == 1
    # factor 3: 0.25, 0.75, 2.25
    assert closest_level_for_mpp(0.25, 2.0, 3, downscale_factor=3) == 2


@pytest.mark.parametrize("x0,y0,d,kind", [
    (0, 0, 1, "shift"), (10, 20, 1, "shift"), (0, 0, 4, "scale"), (8, 16, 2, "affine")])
def test_crop_scale_transform(x0, y0, d, kind):
    t = crop_scale_transform(x0, y0, d)
    assert t["type"] == kind and kind in TRANSFORM_TYPES
    assert (t["from"], t["to"]) == ("source_level0", "output_level0")
    # the matrix always implements x_out = (x_src - x0) / d
    assert _apply(t["matrix"], x0 + 6 * d, y0 + 2 * d) == pytest.approx((6, 2))
    if kind == "shift":
        assert (t["dx"], t["dy"]) == (-x0, -y0)
    if kind == "scale":
        assert (t["sx"], t["sy"]) == (1 / d, 1 / d)


def test_pick_source_level():
    ds = [1.0, 2.0, 4.0, 8.0]
    assert _pick_source_level(ds, 1) == (0, 1)
    assert _pick_source_level(ds, 4) == (2, 1)
    assert _pick_source_level(ds, 32) == (3, 4)
    # scanner levels that are not powers of 2: only level 0 divides 2
    assert _pick_source_level([1.0, 3.0, 9.0], 2) == (0, 2)
    assert _pick_source_level([1.0, 4.0001], 4) == (1, 1)  # rounding noise ok


# ----------------------------------------------------------------------
# wsi2zarr with a start level
# ----------------------------------------------------------------------

class LevelVipsImage(FakeVipsImage):
    """Fake whose loader honours ``level=`` like openslideload: stored
    level k is level 0 subsampled by 2**k."""

    @staticmethod
    def new_from_file(path, level=None, **kwargs):
        img = FakeVipsImage.new_from_file(path)
        if level:
            img = FakeVipsImage(img.arr[:: 2 ** level, :: 2 ** level])
        return LevelVipsImage(img.arr)


class LevelWSI(FakeWSI):
    def __init__(self, path):
        super().__init__(path)
        self.info["level_downsamples"] = [1.0, 2.0, 4.0]


@pytest.fixture
def fake_levels(fake_slide, monkeypatch):
    sys.modules["pyvips"].Image = LevelVipsImage
    monkeypatch.setattr(sys.modules["compass.connector.wsi"], "WSI", LevelWSI)
    return fake_slide


@pytest.mark.parametrize("use_stored_levels", [False, True])
def test_wsi2zarr_start_level(fake_slide, monkeypatch, tmp_path, use_stored_levels):
    slide_path, arr = fake_slide
    loads = []
    if use_stored_levels:
        sys.modules["pyvips"].Image = LevelVipsImage
        monkeypatch.setattr(sys.modules["compass.connector.wsi"], "WSI", LevelWSI)
        orig = LevelVipsImage.new_from_file
        monkeypatch.setattr(LevelVipsImage, "new_from_file", staticmethod(
            lambda p, **kw: (loads.append(kw.get("level")), orig(p, **kw))[1]))

    dst = tmp_path / "s.zarr"
    rec = wsi2zarr(slide_path, dst, crop=(9, 17, 1400, 1100), start_level=1,
                   min_size=256, chunk_size=64, shard_factor=2,
                   provenance={"tool": "test"})
    m = ZarrRaster(dst)
    assert m.nlevels == 2
    assert m.shape(0) == ImageShape(width=700, height=550)
    # fakes subsample: output level 0 = snapped crop, every 2nd pixel
    np.testing.assert_array_equal(m.get_plane(0), arr[16:16 + 1100:2, 8:8 + 1400:2])
    if use_stored_levels:
        assert loads[0] == 1  # read slide level 1 directly, no level-0 read

    # calibration follows the new level 0
    assert m.info["base_mpp"] == 0.5
    assert m.info["base_objective_power"] == 20.0

    # record: returned, stored, consistent
    assert m.info["import"] == json.loads(json.dumps(rec))
    assert rec["tool"] == "test"
    assert rec["source"]["name"] == "slide.mrxs" and rec["source"]["format"] == "wsi"
    assert (rec["source"]["width"], rec["source"]["height"]) == (1500, 1200)
    assert rec["crop"] == {"mode": "explicit", "x0": 8, "y0": 16, "width": 1400, "height": 1100,
                           "requested": {"x0": 9, "y0": 17, "width": 1400, "height": 1100}}
    assert rec["start"]["level"] == 1 and rec["start"]["downsample"] == 2
    out = rec["output"]
    assert (out["width"], out["height"], out["levels"]) == (700, 550, 2)
    assert (out["mpp_x"], out["objective_power"]) == (0.5, 20.0)
    assert rec["transform"]["type"] == "affine"
    # a source pixel lands where the stored pixels say it does
    x, y = _apply(rec["transform"]["matrix"], 8 + 2 * 37, 16 + 2 * 11)
    assert (x, y) == (37, 11)
    np.testing.assert_array_equal(m.get_plane(0)[11, 37], arr[16 + 22, 8 + 74])


def test_wsi2zarr_default_record_full_extent(fake_slide, tmp_path):
    slide_path, _ = fake_slide
    rec = wsi2zarr(slide_path, tmp_path / "s.zarr", min_size=512,
                   chunk_size=64, shard_factor=2)
    assert rec["crop"]["mode"] == "none"
    assert (rec["crop"]["x0"], rec["crop"]["width"]) == (0, 1500)
    assert rec["transform"]["type"] == "shift"
    assert (rec["transform"]["dx"], rec["transform"]["dy"]) == (0.0, 0.0)


def test_wsi2zarr_start_level_out_of_plan(fake_slide, tmp_path):
    slide_path, _ = fake_slide
    with pytest.raises(ValueError, match="start level 5"):
        wsi2zarr(slide_path, tmp_path / "s.zarr", start_level=5)
    assert not (tmp_path / "s.zarr").exists()


# ----------------------------------------------------------------------
# ometiff2zarr with a start level + the CLI
# ----------------------------------------------------------------------

tifffile = pytest.importorskip("tifffile")
from compass.connector import ometiff2zarr  # noqa: E402
from compass.tests.test_connector_ometiff import write_ome  # noqa: E402


@pytest.fixture
def ome_rgb(tmp_path):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, size=(1100, 1400, 3), dtype=np.uint8)
    src = tmp_path / "rgb.ome.tif"
    write_ome(src, img, "YXS", PhysicalSizeX=0.25, PhysicalSizeY=0.25)
    return src, img


def test_ometiff2zarr_start_level_box_average(ome_rgb, tmp_path, monkeypatch):
    src, img = ome_rgb
    monkeypatch.setattr("compass.connector.ingest._BASE_READ_EDGE", 128)  # many sub-reads
    dst = tmp_path / "o.zarr"
    rec = ometiff2zarr(src, dst, crop=(101, 51, 1200, 1000), start_level=2,
                       objective_power=40.0, min_size=64, chunk_size=64, shard_factor=2)
    m = ZarrRaster(dst)
    # planned 1200x1000 -> 600x500 -> 300x250 -> 150x125 (75x62 < min_size)
    assert [m.shape(k).width for k in range(m.nlevels)] == [300, 150]
    # origin snapped to the level-2 grid: 101 -> 100, 51 -> 48
    assert (rec["crop"]["x0"], rec["crop"]["y0"]) == (100, 48)
    ref = box_downsample(img[48:48 + 1000, 100:100 + 1200], 4)
    np.testing.assert_array_equal(m.get_plane(0), ref)
    assert m.info["base_mpp"] == 1.0 and m.info["base_objective_power"] == 10.0
    assert rec["source"]["series"] == 0 and rec["source"]["dtype"] == "uint8"
    assert rec["output"]["level_shapes"] == [[300, 250], [150, 125]]


def _run_tool(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


def _load_tool():
    spec = importlib.util.spec_from_file_location("wsi_import", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tool_info_show_meta_json_is_dry_run(ome_rgb, tmp_path):
    src, _ = ome_rgb
    r = _run_tool(src, tmp_path / "x.zarr", "--info", "--show-meta",
                  "--objective-power", "40", "--start-mpp", "0.6",
                  "--crop", "100", "50", "1200", "1000", "--min-size", "64")
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout)  # stdout is only the record
    assert rec["dry_run"] is True and rec["tool"] == "wsi_import"
    assert rec["start"] == {"level": 1, "downsample": 2, "mpp_x": 0.5, "mpp_y": 0.5,
                            "requested_level": None, "requested_mpp": 0.6}
    assert rec["output"]["width"] == 600 and rec["output"]["objective_power"] == 20.0
    assert "pyramid plan" in r.stderr and "> [1]" in r.stderr
    assert not (tmp_path / "x.zarr").exists()


def test_tool_convert_kv_is_eval_able(ome_rgb, tmp_path):
    src, _ = ome_rgb
    dst = tmp_path / "with space.zarr"
    r = _run_tool(src, dst, "--show-meta", "--meta-format", "kv",
                  "--objective-power", "40", "--start-level", "1",
                  "--min-size", "64", "--chunk-size", "64")
    assert r.returncode == 0, r.stderr
    assert "wrote" in r.stderr and "wrote" not in r.stdout
    script = r.stdout + '\nprintf "%s|%s|%s|%s|%s" "$OUTPUT_PATH" "$CROP_X0" ' \
                        '"$TRANSFORM_TYPE" "$TRANSFORM_MATRIX" "$START_REQUESTED_MPP"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    path, x0, kind, matrix, req_mpp = out.stdout.split("|")
    assert path == str(dst.resolve())
    assert (x0, kind, req_mpp) == ("0", "scale", "")
    assert [float(v) for v in matrix.split()] == [0.5, 0, 0, 0, 0.5, 0]
    # the same record is in the store
    assert ZarrRaster(dst).info["import"]["start"]["level"] == 1


def test_tool_start_level_out_of_plan_lists_levels(ome_rgb, tmp_path):
    src, _ = ome_rgb
    r = _run_tool(src, tmp_path / "x.zarr", "--objective-power", "40",
                  "--start-level", "9")
    assert r.returncode != 0
    assert "Planned levels" in r.stderr and "[2] 350 x 275" in r.stderr
    assert not (tmp_path / "x.zarr").exists()


def test_format_record_kv_quoting_and_flattening():
    tool = _load_tool()
    text = tool.format_record({"a": {"b": None, "c": [[1, 2], [3, 4]]},
                               "path": "it's here", "ok": True}, "kv")
    assert text.splitlines() == ["A_B=''", "A_C='1 2 3 4'",
                                 "PATH=" + shlex.quote("it's here"), "OK=true"]
