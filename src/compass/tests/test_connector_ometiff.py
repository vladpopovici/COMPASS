# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Tests for OME-TIFF ingestion (compass.connector.ometiff2zarr).

tifffile is pure Python, so unlike wsi2zarr these run end-to-end against
real pyramidal OME-TIFF files written into tmp_path — no fakes.
"""

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from compass.connector import box_downsample, ometiff2zarr
from compass.connector.ometiff import read_ome_metadata, yxc_selection
from compass.core import ImageShape, ZarrRaster


def write_ome(path, data, axes, *, levels=2, **metadata):
    """Write ``data`` as a tiled pyramidal OME-TIFF (SubIFD levels)."""
    meta = {"axes": axes, **metadata}
    rgb = axes.endswith("S")
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as tw:
        tw.write(data, subifds=levels - 1, tile=(128, 128), metadata=meta,
                 photometric="rgb" if rgb else "minisblack")
        sub = data
        for _ in range(levels - 1):
            sub = sub[..., ::2, ::2, :] if rgb else sub[..., ::2, ::2]
            tw.write(sub, subfiletype=1, tile=(128, 128),
                     photometric="rgb" if rgb else "minisblack")


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------

def test_box_downsample_mean_and_rounding():
    a = np.array([[0, 1, 4, 4],
                  [1, 1, 4, 5],
                  [9, 9, 2, 2]], dtype=np.uint8)
    # trailing row dropped (floor), 0.75 -> 1, 4.25 -> 4
    np.testing.assert_array_equal(box_downsample(a, 2), [[1, 4]])
    b = np.full((4, 4, 2), 65535, dtype=np.uint16)
    out = box_downsample(b, 2)
    assert out.dtype == np.uint16 and out.shape == (2, 2, 2) and (out == 65535).all()


def test_yxc_selection():
    assert yxc_selection("YXS", (10, 20, 3)) == ([slice(None)] * 3, [0, 1, 2], 3)
    idx, perm, nc = yxc_selection("TCZYX", (1, 4, 3, 10, 20), {"Z": 2, "T": 0})
    assert idx == [0, slice(None), 2, slice(None), slice(None)]
    assert perm == [1, 2, 0] and nc == 4
    # single-channel: C of size 1 is dropped
    assert yxc_selection("CYX", (1, 10, 20))[1:] == ([0, 1], 1)
    with pytest.raises(ValueError, match="select one plane"):
        yxc_selection("ZYX", (5, 10, 20))
    with pytest.raises(ValueError, match="more than one channel axis"):
        yxc_selection("CYXS", (2, 10, 20, 3))


OME_WITH_OBJECTIVE = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Instrument ID="Instrument:0">
    <Objective ID="Objective:0" NominalMagnification="20.0"/>
    <Objective ID="Objective:1" NominalMagnification="40.0"/>
  </Instrument>
  <Image ID="Image:0">
    <InstrumentRef ID="Instrument:0"/>
    <ObjectiveSettings ID="Objective:1"/>
    <Pixels ID="Pixels:0" PhysicalSizeX="0.00025" PhysicalSizeXUnit="mm"
            PhysicalSizeY="0.25">
      <Channel ID="Channel:0:0" Name="DAPI"/>
      <Channel ID="Channel:0:1"/>
    </Pixels>
  </Image>
</OME>"""


def test_read_ome_metadata():
    meta = read_ome_metadata(OME_WITH_OBJECTIVE)
    assert meta["mpp_x"] == pytest.approx(0.25)   # mm -> µm
    assert meta["mpp_y"] == pytest.approx(0.25)   # unit defaults to µm
    assert meta["objective_power"] == 40.0        # via ObjectiveSettings
    assert meta["channel_names"] == ["DAPI", None]
    with pytest.raises(IndexError):
        read_ome_metadata(OME_WITH_OBJECTIVE, image_index=1)


def test_read_ome_metadata_objective_via_instrument_ref():
    # the minimal OME-XML of raster2tiff: InstrumentRef, no ObjectiveSettings
    xml = OME_WITH_OBJECTIVE.replace('<ObjectiveSettings ID="Objective:1"/>', "")
    assert read_ome_metadata(xml)["objective_power"] is None  # ambiguous: 2 objectives
    xml = xml.replace('<Objective ID="Objective:1" NominalMagnification="40.0"/>', "")
    assert read_ome_metadata(xml)["objective_power"] == 20.0


# ----------------------------------------------------------------------
# ometiff2zarr end-to-end
# ----------------------------------------------------------------------

def reference_pyramid(base, factor, nlevels):
    lv = [base]
    for _ in range(nlevels - 1):
        lv.append(box_downsample(lv[-1], factor))
    return lv


def test_ometiff2zarr_rgb_uint8(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(1100, 1400, 3), dtype=np.uint8)
    src = tmp_path / "rgb.ome.tif"
    write_ome(src, img, "YXS", PhysicalSizeX=0.25, PhysicalSizeY=0.25)

    dst = tmp_path / "rgb.zarr"
    ometiff2zarr(src, dst, objective_power=40.0, min_size=256,
                 chunk_size=64, shard_factor=2)

    m = ZarrRaster(dst)
    assert m.nlevels == 3
    assert m.shape(2) == ImageShape(width=350, height=275)
    for lv, ref in enumerate(reference_pyramid(img, 2, 3)):
        np.testing.assert_array_equal(m.get_plane(lv), ref)
    assert m.channel_names == ["R", "G", "B"]
    assert m.info["base_mpp"] == 0.25
    assert m.info["base_objective_power"] == 40.0
    assert m.level_info(1)["mpp_x"] == 0.5
    assert m.get_level_for_mpp(1.0) == 2


def test_ometiff2zarr_planar_uint16_named_channels(tmp_path):
    rng = np.random.default_rng(2)
    img = rng.integers(0, 65535, size=(3, 600, 900), dtype=np.uint16)
    src = tmp_path / "mif.ome.tif"
    write_ome(src, img, "CYX",
              PhysicalSizeX=500, PhysicalSizeXUnit="nm",
              PhysicalSizeY=500, PhysicalSizeYUnit="nm",
              Channel={"Name": ["DAPI", "CD3", "CD8"]})

    dst = tmp_path / "mif.zarr"
    ometiff2zarr(src, dst, objective_power=20.0, crop=(100, 50, 640, 520),
                 min_size=128, chunk_size=64, shard_factor=2)

    m = ZarrRaster(dst)
    base = np.moveaxis(img, 0, -1)[50:570, 100:740]
    assert m.get_plane(0).dtype == np.uint16
    assert m.nlevels == 3   # 640x520, 320x260, 160x130
    for lv, ref in enumerate(reference_pyramid(base, 2, 3)):
        np.testing.assert_array_equal(m.get_plane(lv), ref)
    assert m.channel_names == ["DAPI", "CD3", "CD8"]
    assert m.info["base_mpp"] == pytest.approx(0.5)


def test_ometiff2zarr_single_channel_zstack_and_factor3(tmp_path):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, size=(4, 900, 800), dtype=np.uint8)
    src = tmp_path / "z.ome.tif"
    write_ome(src, img, "ZYX", levels=1, PhysicalSizeX=1.0, PhysicalSizeY=1.0)

    with pytest.raises(ValueError, match="select one plane"):
        ometiff2zarr(src, tmp_path / "bad.zarr", objective_power=10.0)
    # crop=True: no scan ROI in OME-TIFF -> whole image
    ometiff2zarr(src, tmp_path / "z.zarr", z=2, crop=True, objective_power=10.0,
                 downscale_factor=3, min_size=64, chunk_size=64, shard_factor=2)
    m = ZarrRaster(tmp_path / "z.zarr")
    assert m.nchannels == 1 and m.channel_names == ["C0"]
    assert m.shape(1) == ImageShape(width=266, height=300)
    for lv, ref in enumerate(reference_pyramid(img[2][..., None], 3, m.nlevels)):
        np.testing.assert_array_equal(m.get_plane(lv), ref)
    assert m.level_info(1)["objective_power"] == pytest.approx(10.0 / 3)


def test_ometiff2zarr_requires_calibration(tmp_path):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    src = tmp_path / "nocal.ome.tif"
    write_ome(src, img, "YXS", levels=1)

    with pytest.raises(ValueError, match="mpp="):
        ometiff2zarr(src, tmp_path / "a.zarr", objective_power=20.0)
    with pytest.raises(ValueError, match="objective_power="):
        ometiff2zarr(src, tmp_path / "b.zarr", mpp=0.5)
    # explicit overrides suffice, anisotropic mpp kept per axis
    ometiff2zarr(src, tmp_path / "c.zarr", mpp=(0.5, 0.6), objective_power=20.0,
                 channel_names=["r", "g", "b"])
    m = ZarrRaster(tmp_path / "c.zarr")
    assert m.info["base_mpp"] == pytest.approx(0.55)
    assert m.level_info(0)["mpp_y"] == 0.6
    assert m.channel_names == ["r", "g", "b"]


def test_ometiff2zarr_overwrite(tmp_path):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    src = tmp_path / "o.ome.tif"
    write_ome(src, img, "YXS", levels=1, PhysicalSizeX=0.5, PhysicalSizeY=0.5)
    dst = tmp_path / "o.zarr"
    ometiff2zarr(src, dst, objective_power=20.0)
    with pytest.raises(Exception):
        ometiff2zarr(src, dst, objective_power=20.0)
    ometiff2zarr(src, dst, objective_power=20.0, overwrite=True)
    assert ZarrRaster(dst).nlevels == 1


def test_ometiff2zarr_roundtrip_with_raster2tiff_metadata(tmp_path):
    # raster2tiff's minimal OME-XML (objective via InstrumentRef) must ingest
    # without overrides — export/ingest close the loop
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <Instrument ID="Instrument:0">
            <Objective ID="Objective:0" NominalMagnification="40.0"/>
        </Instrument>
        <Image ID="Image:0" Name="x">
            <InstrumentRef ID="Instrument:0"/>
            <Pixels DimensionOrder="XYCZT" ID="Pixels:0" SizeC="3" SizeT="1"
                    SizeX="400" SizeY="300" SizeZ="1" Type="uint8"
                    PhysicalSizeX="0.25" PhysicalSizeXUnit="µm"
                    PhysicalSizeY="0.25" PhysicalSizeYUnit="µm">
            </Pixels>
        </Image>
    </OME>"""
    img = np.random.default_rng(4).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    src = tmp_path / "exp.ome.tiff"
    tifffile.imwrite(src, img, photometric="rgb", tile=(128, 128),
                     description=xml.encode(), metadata=None)
    dst = tmp_path / "exp.zarr"
    ometiff2zarr(src, dst, min_size=128)
    m = ZarrRaster(dst)
    assert m.info["base_objective_power"] == 40.0 and m.info["base_mpp"] == 0.25
    np.testing.assert_array_equal(m.get_plane(0), img)
