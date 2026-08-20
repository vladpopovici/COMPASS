# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""On-demand export to OME-TIFF/BigTIFF for interchange with other
pathology tools (QuPath, ASAP, MIKAIA, clinical viewers). Zarr stays the
internal format; this is the interchange path (docs/architecture.md
section 2). Port of the legacy ``mri2tiff``/``build_omexml``
(``compass.legacy.core``), generalized from the HDF5-backed MRI to any
:class:`compass.core.PyramidalRaster`.
"""

import logging
from os import PathLike
from pathlib import Path

from ..core import ImageShape, PyramidalRaster

logger = logging.getLogger(__name__)


##-
def build_omexml(
        image_name: str = "noname",
        image_description: str = "no description",
        image_shape: ImageShape = ImageShape(width=1, height=1),
        image_type: str = "RGB",  # RGB, BGR, gray
        magnif: float = 1.0,
        pixel_type: str = "uint8",
        mpp: float = 1.0,
) -> str:
    """
    Builds a reduced OME XML file.

    Args:
        image_name: (optional) image name
        image_description: (optional) image description
        image_shape: image shape (width, height)
        image_type: image type RGB or BGR or gray
        magnif: objective native magnification
        pixel_type: pixel data type (uint8 or uint16)
        mpp: resolution in microns per pixel

    Returns:
        the OME XML as a string
    """
    if image_type == "RGB" or image_type == "BGR":
        n_channels = 3
    else:
        n_channels = 1

    omexml = \
    f"""
    <?xml version="1.0" encoding="UTF-8"?>
    <OME
    xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" UUID="urn:uuid:29a39710-33c5-4e40-8faf-c9d146496bd2">
    <Instrument ID="Instrument:0">
        <Microscope Manufacturer="virtual" Model="v1"/>
        <Objective ID="Objective:0:0" Manufacturer="virtual" Model="v1" NominalMagnification="{magnif}"/>
    </Instrument>
    <Image ID="Image:0" Name="{image_name}">
        <Description>{image_description}</Description>
        <InstrumentRef ID="Instrument:0"/>
        <ObjectiveSettings ID="Objective:0:0"/>
        <Pixels ID="Pixels:0"
            DimensionOrder="XYCZT"
            Type="{pixel_type}"
            SignificantBits="8"
            Interleaved="true"
            SizeX="{image_shape.width}"
            SizeY="{image_shape.height}"
            SizeZ="1"
            SizeC="{n_channels}"
            SizeT="1"
            PhysicalSizeX="{mpp}"
            PhysicalSizeXUnit="µm"
            PhysicalSizeY="{mpp}"
            PhysicalSizeYUnit="µm">
        </Pixels>
        <Channel ID="Channel:0:0" SamplesPerPixel="{n_channels}">
            <LightPath/>
        </Channel>
        <TiffData IFD="0" PlaneCount="1">
        </TiffData>
        <Plane TheZ="0" TheT="0" TheC="0" PositionX="0.0" PositionXUnit="nm" PositionY="0.0" PositionYUnit="nm" PositionZ="0.0" PositionZUnit="nm"/>
    </Image>
    </OME>
    """

    return omexml
##


##-
def raster2tiff(
        raster: PyramidalRaster,
        out_path: str | Path | PathLike,
        start_level: int = 0,
        overwrite: bool = True,
        tile_shape: tuple[int, int] = (1024, 1024),
        minimal_omexml: bool = True
) -> str:
    """
    Save a pyramidal raster in a pyramidal BigTiff file with the meta
    information properly set to follow OME TIFF specification.

    Args:
        raster: any PyramidalRaster (e.g. compass.core.ZarrRaster).
        out_path: path and filename (including the .tiff suffix)
        start_level: starting level of the output TIFF image
        overwrite: if True, overwrite existing file
        tile_shape: a tuple (width, height) of the output tile shape
        minimal_omexml: (optional) minimal OME xml version, best supported
            by various applications.
    Returns:
        the OME XML metadata written into the file

    Notes:
        For compatibility with MIKAIA, make sure to use minimal_omexml=True.
    """
    import pyvips

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        # refuse
        raise FileExistsError(out_path)
    if out_path.suffix != ".tiff":
        raise RuntimeError(f"Output path {out_path} is not a TIFF file")

    # read the smallest image to get its characteristics
    test_img = raster.get_plane(raster.nlevels - 1)
    n_channels = test_img.shape[-1]
    data_type = str(test_img.dtype)
    if data_type not in ["uint8", "uint16"]:
        raise TypeError(f"Data type {data_type} is not supported")
    im_shape = raster.shape(start_level)
    mpp = raster.get_mpp_for_level(start_level)

    meta = f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Instrument ID="Instrument:0">
            <Microscope Manufacturer="virtual" Model="v1"/>
            <Objective Manufacturer="virtual" Model="v1" ID="Objective:0" NominalMagnification="{raster.get_magnification_for_level(start_level)}"/>
        </Instrument>
        <Image ID="Image:0" Name="{out_path.stem}">
            <InstrumentRef ID="Instrument:0"/>
            <!-- Minimum required fields about image dimensions -->
            <Pixels DimensionOrder="XYCZT"
                    ID="Pixels:0"
                    SizeC="{n_channels}"
                    SizeT="1"
                    SizeX="{im_shape.width}"
                    SizeY="{im_shape.height}"
                    SizeZ="1"
                    Type="{data_type}"
                    PhysicalSizeX="{mpp}"
                    PhysicalSizeXUnit="µm"
                    PhysicalSizeY="{mpp}"
                    PhysicalSizeYUnit="µm">
            </Pixels>
        </Image>
    </OME>"""

    if not minimal_omexml:
        meta = build_omexml(out_path.stem, image_shape=im_shape, image_type="RGB",
                            magnif=raster.get_magnification_for_level(start_level),
                            pixel_type=data_type, mpp=mpp)

    if data_type == "uint8":
        out_img = pyvips.Image.new_from_array(raster.get_plane(start_level), interpretation="rgb")
    else:
        out_img = pyvips.Image.new_from_array(raster.get_plane(start_level), interpretation="rgb16")

    out_img.set_type(pyvips.GValue.gstr_type, "image-description", meta)

    out_img.write_to_file(
        out_path,
        compression='lzw',
        tile=True,
        tile_width=tile_shape[0],
        tile_height=tile_shape[1],
        pyramid=True,
        subifd=True,
        bigtiff=True,
        miniswhite=False,
        xres=10000.0 / mpp, yres=10000.0 / mpp, resunit="cm",
    )

    return meta
##
