# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from os import PathLike
from math import floor
from pathlib import Path
import pyvips
import h5py
import hdf5plugin
import numpy as np
import netCDF4 as cdf
import dask.array as da
import tempfile
import tqdm
#from dask_image.ndfilters import gaussian_filter

"""
COMPASS.CORE: core classes and functions.
"""

from _magnif import Magnification
from _pyr import PyramidalImage
from _wsi import WSI
from _mri import MRI
from _misc import (ImageShape, Px, NumpyImage, NumpyJSONEncoder, rgb2ycbcr, ycbcr2rgb, R_, G_, B_)

__all__ = [
    'ImageShape',
    'Px',
    'Magnification',
    'PyramidalImage',
    'WSI',
    'MRI',
    'NumpyImage',
    'rgb2ycbcr',
    'ycbcr2rgb',
    'R_', 'G_', 'B_',
    'NumpyJSONEncoder',
    'wsi2hdf5',
    'mri2tiff'
]



##-
def wsi2hdf5(
        wsi_path: str | Path | PathLike,
        dst_path: str | Path | PathLike,
        crop: tuple[int, int, int, int] | bool = False,
        band_size: int = 1528,
        downscale_factor: int = 2,
        min_size: int = 256,
        compression: str = "none",
) -> None:
    """
    Converts a WSI file to pyramidal format stored in HDF5. Highest resolution is at the first
    level ("/scale0").

    :param wsi_path: source file path.
    :param dst_path: destination file path (normally a .h5 file).
    :param crop: either bool to control auto-crop or (x0, y0, width, height) for the crop region
    :param band_size: band height for processed regions
    :param downscale_factor: downsampling factor between levels in the pyramid
    :param min_size: minimum size (either width or height) of the last level in the pyramid
    :param compression: one of "none", "lzf", "gzip", "blosc"
    :return: None
    """
    compression = compression.lower()
    if compression not in ['none', 'lzf', 'gzip', 'blosc']:
        raise RuntimeError('Unknown compression type')
    if compression == 'blosc':
        compression = hdf5plugin.Blosc2(cname='blosclz', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)
    elif compression == 'none':
        compression = None

    wsi_path = Path(wsi_path)
    dst_path = Path(dst_path).with_suffix('.h5')

    wsi = WSI(wsi_path)

    # initially, the whole image
    x0, y0, width, height = (0, 0, wsi.info["width"], wsi.info["height"])

    if isinstance(crop, bool):
        if crop and wsi.info['roi'] is not None:
            x0, y0, width, height = (wsi.info['roi']['x0'],
                                     wsi.info['roi']['y0'],
                                     wsi.info['roi']["width"],
                                     wsi.info['roi']["height"])
    else:
        if isinstance(crop, tuple):
            x0, y0, width, height = crop
            x0 = max(0, min(x0, wsi.info["width"]))
            y0 = max(0, min(y0, wsi.info["height"]))
            width = min(width, wsi.info["width"] - x0)
            height = min(height, wsi.info["height"] - y0)

    nlevels = 1
    levels = [{'w': width, 'h': height, 'x0': x0, 'y0': y0}]
    while True:
        w, h = int(floor(levels[-1]['w'] / downscale_factor)), int(floor(levels[-1]['h'] / downscale_factor))
        cx0, cy0 = int(floor(x0 / downscale_factor)), int(floor(y0 / downscale_factor))
        if w < min_size or h < min_size:
            break
        levels.append({'w': w, 'h': h, 'x0': cx0, 'y0': cy0})
    nlevels = len(levels)


    # with h5py.File(str(dst_path), 'w') as root:
    #     for i in range(wsi.level_count):
    #         # copy levels from WSI, band by band...
    #         # -level i crop region:
    #         cx0 = int(floor(x0 / wsi.downsample_factor(i)))
    #         cy0 = int(floor(y0 / wsi.downsample_factor(i)))
    #         cw = int(floor(width / wsi.downsample_factor(i)))
    #         ch = int(floor(height / wsi.downsample_factor(i)))
    #
    #         im = pyvips.Image.new_from_file(str(wsi_path), level=i, autocrop=False)
    #         im = im.crop(cx0, cy0, cw, ch)
    #         im = im.flatten()
    #
    #         shape = (ch, cw, 3)  # YXC axes
    #         levels[:, i] = (cw, ch)
    #
    #         current_level = root.create_group(str(i))
    #         arr = current_level.create_dataset(
    #             "data", shape=shape,
    #             chunks=(min(512, ch), min(512, cw), 3),
    #             dtype="uint8",
    #             # compression="lzf"
    #             compression=hdf5plugin.Blosc2(cname='blosclz', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)
    #         )
    #
    #         n_bands = ch // band_size
    #         incomplete_band = shape[0] % band_size
    #         for j in range(n_bands):  # by horizontal bands
    #             buf = im.crop(0, j * band_size, cw, band_size).numpy()
    #             arr[j * band_size: (j + 1) * band_size] = buf
    #
    #         if incomplete_band > 0:
    #             buf = im.crop(0, n_bands * band_size, cw, incomplete_band).numpy()
    #             arr[n_bands * band_size: n_bands * band_size + incomplete_band] = buf
    #     root.attrs["max_level"] = wsi.level_count
    #     root.attrs["channel_names"] = ["R", "G", "B"]
    #     root.attrs["dimension_names"] = ["y", "x", "c"]
    #     root.attrs["mpp_x"] = wsi.info['mpp_x']
    #     root.attrs["mpp_y"] = wsi.info["mpp_y"]
    #     root.attrs["mag_step"] = int(wsi.info['magnification_step'])
    #     root.attrs["objective_power"] = wsi.info['objective_power']
    #     root.attrs["extent"] = levels.tolist()
    #     root.attrs["numpy_ordering"] = 1  # data is stored as NumPy array - might be important in other languages

    # Approach:
    # -copy the highest resolution image to destination
    # -each new level is generated by downscaling the original one
    #
    # This allows a different pyramid structure than the one stored in WSI and
    # gives the control on how lower resolutions as generated.

    level = 0
    with h5py.File(str(dst_path), 'w') as root:
        def im_to_arr(im: pyvips.Image, arr: h5py.Dataset):
            n_bands = ch // band_size
            incomplete_band = shape[0] % band_size
            for j in range(n_bands):  # by horizontal bands
                buf = im.crop(0, j * band_size, cw, band_size).numpy()
                arr[j * band_size: (j + 1) * band_size] = buf

            if incomplete_band > 0:
                buf = im.crop(0, n_bands * band_size, cw, incomplete_band).numpy()
                arr[n_bands * band_size: n_bands * band_size + incomplete_band] = buf

        # level 0: copy
        level = 0

        im = pyvips.Image.new_from_file(str(wsi_path), level=level, autocrop=False)
        cx0, cy0, cw, ch = levels[level]['x0'], levels[level]['y0'], levels[level]['w'], levels[level]['h']
        im = im.crop(cx0, cy0, cw, ch)
        im = im.flatten()

        print("Copying level 0...")
        shape = (ch, cw, 3)  # YXC axes
        current_level = root.create_group(f"scale{level}")
        arr = current_level.create_dataset(
            "image",
            shape=shape,
            chunks=(min(8192, ch), min(8192, cw), 3),
            dtype="uint8",
            compression=compression,
        )
        print("...OK")
        # n_bands = ch // band_size
        # incomplete_band = shape[0] % band_size
        # for j in range(n_bands):  # by horizontal bands
        #     buf = im.crop(0, j * band_size, cw, band_size).numpy()
        #     arr[j * band_size: (j + 1) * band_size] = buf
        #
        # if incomplete_band > 0:
        #     buf = im.crop(0, n_bands * band_size, cw, incomplete_band).numpy()
        #     arr[n_bands * band_size: n_bands * band_size + incomplete_band] = buf

        im_to_arr(im, arr)

        # array-specific metadata
        arr.attrs["channel_names"] = ["R", "G", "B"]
        arr.attrs["dimension_names"] = ["y", "x", "c"]

        # scale-specific metadata
        current_level.attrs["mpp_x"] = downscale_factor**level * wsi.info['mpp_x']
        current_level.attrs["mpp_y"] = downscale_factor**level * wsi.info['mpp_y']
        current_level.attrs["objective_power"] = wsi.info['objective_power'] / (downscale_factor**level)
        current_level.attrs["scale_factor"] = downscale_factor**level

        for level in range(1, nlevels):
            cw, ch = levels[level]['w'], levels[level]['h']  # current extent
            sf = downscale_factor ** level                   # scale factor
            im_scaled = im.resize(sf, pyvips.enums.Kernel.CUBIC)

            # write to file
            shape = (ch, cw, 3)  # YXC axes
            current_level = root.create_group(f"scale{level}")
            arr = current_level.create_dataset(
                "image",
                shape=shape,
                chunks=(min(8192, ch), min(8192, cw), 3),
                dtype="uint8",
                compression=compression,
            )
            print(f"Generating level {level}...")
            im_to_arr(im, arr)
            print("...OK")
            # array-specific metadata
            arr.attrs["channel_names"] = ["R", "G", "B"]
            arr.attrs["dimension_names"] = ["y", "x", "c"]

            # scale-specific metadata
            current_level.attrs["mpp_x"] = sf * wsi.info['mpp_x']
            current_level.attrs["mpp_y"] = sf * wsi.info['mpp_y']
            current_level.attrs["objective_power"] = wsi.info['objective_power'] / sf
            current_level.attrs["scale_factor"] = sf

        # # Use dask-image to generate the levels of the pyramid
    # with h5py.File(str(dst_path), 'a') as root:
    #     lv0 = da.array(root["/scale0/image"])
    #     for level in range(1, nlevels):
    #         cw, ch = levels[level]['w'], levels[level]['h']  # current extent
    #         sf = downscale_factor ** level                   # scale factor
    #         sigma = sf / 2
    #         lvk_blurred = gaussian_filter(lv0, sigma=[sigma, sigma, 0])
    #         lvk_blurred = lvk_blurred[:ch, :cw, :]
    #         lvk = lvk_blurred[::sf, ::sf, :].astype(lv0.dtype)  # keep type
    #
    #         # write to file
    #         current_level = root.create_group(f"scale{level}")
    #         arr = current_level.create_dataset(
    #             "image",
    #             data=lvk,
    #         #    shape=(ch, cw, 3),
    #             chunks=(min(8192, ch), min(8192, cw), 3),
    #         #    dtype="uint8",
    #             compression=compression,
    #         )
    #         # array-specific metadata
    #         arr.attrs["channel_names"] = ["R", "G", "B"]
    #         arr.attrs["dimension_names"] = ["y", "x", "c"]
    #
    #         # scale-specific metadata
    #         current_level.attrs["mpp_x"] = sf * wsi.info['mpp_x']
    #         current_level.attrs["mpp_y"] = sf * wsi.info['mpp_y']
    #         current_level.attrs["objective_power"] = wsi.info['objective_power'] / sf
    #         current_level.attrs["scale_factor"] = sf

        root.attrs["max_level"] = nlevels
        root.attrs["base_mpp_x"] = wsi.info['mpp_x']
        root.attrs["base_mpp_y"] = wsi.info["mpp_y"]
        root.attrs["base_mag_step"] = downscale_factor
        root.attrs["base_objective_power"] = wsi.info['objective_power']
        root.attrs["extent"] = [[d['w'], d['h']] for d in levels]
        root.attrs["numpy_ordering"] = 1  # data is stored as NumPy array - might be important in other languages

    return


##

##-
def build_omexml(
        image_name: str = "noname",
        image_description: str = "no description",
        image_shape: ImageShape = (1, 1),
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
def mri2tiff(mri: MRI, out_path: str | Path | PathLike, overwrite: bool = True,
             tile_shape: tuple[int, int] = (1024, 1024),
             minimal_omexml: bool = True) -> str:
    """
    Save a multi-resolution image in a pyramidal BigTiff file with the meta information
    properly set to follow OME TIFF specification.

    Args:
        mri: an MRI object.
        out_path: path and filename (including the .tiff suffix)
        overwrite: if True, overwrite existing file
        tile_shape: a tuple (width, height) of the output tile shape
        minimal_omexml: (optional) minimal OME xml version, best supported by various applications.
    Returns:
        None

    Notes:
        For compatibility with MIKAIA, make sure to use minimal_omexml=True.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        # refuse
        raise FileExistsError(out_path)
    if out_path.suffix != ".tiff":
        raise RuntimeError(f"Output path {out_path} is not a TIFF file")
    test_img = mri.get_plane(mri.nlevels - 1)
    n_channels = test_img.shape[-1]
    data_type = str(test_img.dtype)
    if data_type not in ["uint8", "uint16"]:
        raise TypeError(f"Data type {data_type} is not supported")
    im_shape = mri.shape(0)
    mpp = mri.get_native_resolution

    meta = f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Instrument ID="Instrument:0">
            <Microscope Manufacturer="virtual" Model="v1"/>
            <Objective Manufacturer="virtual" Model="v1" ID="Objective:0" NominalMagnification="{mri.get_native_magnification}"/>
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
                            magnif=mri.get_native_magnification, pixel_type=data_type,
                            mpp=mpp)

    if data_type == "uint8":
        out_img = pyvips.Image.new_from_array(mri.get_plane(0), interpretation="rgb")
    else:
        out_img = pyvips.Image.new_from_array(mri.get_plane(0), interpretation="rgb16")

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

