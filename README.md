COMPASS: Computational Pathology and Spatial Statistics
---

This package contains both a rich library of functions for computational
pathology and a collection of applications dealing with various problems
including spatial transcriptomics. This broad coverage comes with the 
price of installing many dependencies.

To avoid having to dealing with various image (and data) formats (and their
evolving specifications), we store the data in ZARR format, with images
saved in pyramidal structures easily opened in [NAPARI](https://napari.org).
An image pyramid is stored as a ZARR group and has a simple structure 
(see wsitk-utils for a converter):
```
pyramid_<k>
   +--- 0 <- highest resolution (largest image)
   +--- 1
   ...
   +--- n <- lowest resolution 
```
In addition, "pyramid_<k>" has a few attributed (meta-data) guaranteed:
* `max_level`: number of levels in the pyramid
* `channel_names`: normally R, G, B, but others may be possible
* `dimension_names`: normally y, x, c
* `mpp_x` and `mpp_y`: the original resolution (microns-per-pixel) for x and y dimensions
* `mag_step`: magnification step (usually 2.0): scaling factor between pyramid levels
* `objective_power`: native objective used to acquire the image (e.g. 20.0 or 40.0)
* `extent`: a `2 x max_level` array with `extent[0,i]` and `extent[1,i]` indicating
   the width and height of level `i`, respectively

A tissue section may have several pyramids (`pyramid_0`, ..., `pyramid_n`) as long as they
refer to the exact same specimen and they are aligned (registered). They may represent
different modalities (e.g. H&E and IHC), or different versions of the same image (e.g.
gray-scale or stain-normalized image). `pyramid_0` is taken as the reference and it defines
the space in which the annotations are produced/saved, and all the analyses carried out.
