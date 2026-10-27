# -*- coding: utf-8 -*-
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""Tissue (foreground) detection in whole-slide images.

numpy in, numpy out: every detector takes an H x W x 3 RGB image (usually a
low-resolution level of the pyramid) and returns a boolean mask of the same
height/width, ``True`` on tissue. Ported from ``compass.legacy.tissue``.

Typical use, e.g. to find the region worth keeping on a slide::

    mask, _ = detect_foreground(thumbnail, method="simple-he")
    x0, y0, w, h = foreground_bbox(mask)   # in thumbnail pixels
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

__all__ = ["detect_foreground", "foreground_bbox", "mask_to_polygons", "tissue_regions",
           "METHODS"]

from collections.abc import Callable

import cv2
import numpy as np
import skimage.morphology as skm
from scipy import ndimage, signal
from scipy.optimize import fmin_slsqp
from scipy.stats import norm
from shapely import affinity
from shapely.geometry import Polygon, box
from skimage import color
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KernelDensity

from .stain import rgb2he

# A detector returns (mask, threshold); threshold is None for the methods
# that do not use a single intensity threshold.
DetectorResult = tuple[np.ndarray, float | None]


def detect_foreground(img: np.ndarray, method: str = "fast-he",
                      **kwargs) -> DetectorResult:
    """Detect the foreground (tissue) in an RGB slide image.

    Args:
        img: H x W x 3 RGB image (uint8). The ``simple`` method also accepts
            a single-channel H x W image.
        method: one of

            - ``"fast-he"``: fixed threshold on the G plane plus morphology;
              H&E slides. See :func:`he_fast_foreground`.
            - ``"simple-he"``: G-plane threshold estimated by 2-means
              clustering, refined with H/E stain rules; H&E slides. See
              :func:`he_simple_foreground`.
            - ``"simple"``: robust threshold on the L plane (Lab space),
              assuming a bimodal distribution. See :func:`simple_foreground`.
            - ``"htk"``: Gaussian-mixture model of the grey-level histogram,
              from HistomicsTK. See :func:`htk_foreground`.
            - ``"fesi"``: after Bug et al., "Foreground Extraction for
              Histopathological Whole Slide Imaging", BVM 2015, pp. 419-424.
              See :func:`fesi_foreground`.
        kwargs: method-specific parameters, passed through to the detector
            (see its docstring). Unknown names raise ``TypeError``.

    Returns:
        ``(mask, threshold)``: boolean H x W mask (``True`` = tissue) and the
        threshold used (``simple``, ``simple-he``) or ``None``.
    """
    try:
        detector = METHODS[method.lower()]
    except KeyError:
        raise ValueError(f"unknown foreground detection method {method!r}; "
                         f"expected one of {sorted(METHODS)}") from None
    return detector(img, **kwargs)


def foreground_bbox(mask: np.ndarray,
                    min_fraction: float = 0.0) -> tuple[int, int, int, int] | None:
    """Bounding box of the ``True`` pixels of a 2D mask, as
    ``(x0, y0, width, height)``; ``None`` if the mask is empty.

    Args:
        mask: 2D boolean mask.
        min_fraction: ignore connected components smaller than this
            fraction of the largest one (dust, debris, pen specks that would
            otherwise stretch the box to the slide edges); 0 keeps all.
    """
    if min_fraction > 0:
        labels, n = ndimage.label(mask)
        if n > 1:
            areas = np.bincount(labels.ravel())[1:]
            keep = np.flatnonzero(areas >= min_fraction * areas.max()) + 1
            mask = np.isin(labels, keep)
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _check_rgb(img: np.ndarray) -> None:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 RGB image, got {img.shape}")


def _kmeans_threshold(values: np.ndarray, max_samples: int = 100_000) -> int:
    """95% of the brighter of two k-means centres: separates a bright
    background mode from the darker foreground. Fitted on a random
    subsample of at most ``max_samples`` values (fitting all pixels of a
    multi-megapixel overview takes minutes for the same answer)."""
    values = values.reshape(-1, 1)
    if values.shape[0] > max_samples:
        values = np.random.default_rng(0).choice(values, max_samples, replace=False)
    vq = MiniBatchKMeans(n_clusters=2, random_state=0)
    centres = vq.fit(values).cluster_centers_
    return int(np.round(0.95 * centres.max()))


def he_fast_foreground(img: np.ndarray, g_th: int = 220, ker_size: int = 33,
                       min_area: int = 150) -> DetectorResult:
    """Fast foreground detection in H&E slides: fixed threshold on the G(reen)
    plane, hole filling, erosion, small-object removal.

    Args:
        img: H x W x 3 RGB image.
        g_th: G-plane values above this are background.
        ker_size: edge of the square structuring element for the erosion.
        min_area: smallest object (pixels) to keep.
    """
    _check_rgb(img)
    g = img[..., 1]
    mask = (g > 0) & (g <= g_th)
    mask = ndimage.binary_fill_holes(mask)
    kr = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ker_size), int(ker_size)))
    mask = cv2.erode(mask.astype(np.uint8), kr).astype(bool)
    mask = skm.remove_small_objects(mask, min_size=min_area)
    return mask, None


def he_simple_foreground(img: np.ndarray, g_th: float | None = None,
                         min_area: int = 150, fill_holes: bool = True) -> DetectorResult:
    """Foreground detection in H&E slides: G-plane threshold (estimated by
    2-means clustering unless given) plus a few hand-picked stain rules — a
    tissue pixel needs some H or E signal (above their 5th percentile) and
    a residual stain component below its median.

    Args:
        img: H x W x 3 RGB image.
        g_th: G-plane threshold; ``None`` or 0 estimates it from the image.
        min_area: smallest object (pixels) to keep.
        fill_holes: fill all holes of the final mask (default); pass False to
            keep them, e.g. when the caller applies its own hole policy.
    """
    _check_rgb(img)
    if not g_th:
        g_th = _kmeans_threshold(img[..., 1])

    mask = img[..., 1] < g_th
    mask = skm.binary_closing(mask, skm.disk(3))
    mask = skm.remove_small_objects(mask, min_size=min_area)

    h, e, b = rgb2he(img)
    mask &= (h > np.percentile(h, 5)) | (e > np.percentile(e, 5))
    mask &= b < np.percentile(b, 50)

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    return mask, g_th


def simple_foreground(img: np.ndarray, th: float | None = None,
                      min_area: int = 150,
                      background_is_white: bool = True) -> DetectorResult:
    """Foreground detection on the L plane (Lab space) of an RGB image, or
    directly on a single-channel image, with a robust threshold.

    Args:
        img: H x W x 3 RGB or H x W grey-level image.
        th: threshold; ``None`` or 0 estimates it from the image.
        min_area: smallest object (pixels) to keep.
        background_is_white: if True the tissue is darker than the
            background, otherwise brighter.
    """
    if img.ndim > 2:
        img = color.rgb2lab(img)[..., 0]
    if not th:
        th = _kmeans_threshold(img)

    mask = img < th if background_is_white else img > th
    mask = skm.binary_closing(mask, skm.disk(3))
    mask = skm.remove_small_objects(mask, min_size=min_area)
    mask = ndimage.binary_fill_holes(mask)
    mask = skm.binary_opening(mask, skm.disk(5))
    return mask, th


def fesi_foreground(img: np.ndarray, laplace_ker: int = 5, gauss_ker: int = 15,
                    gauss_sigma: float = 4.0, morph_open_ker: int = 9,
                    morph_open_iter: int = 7, morph_blur: int = 15,
                    min_area: int = 150) -> DetectorResult:
    """FESI foreground detection (Bug et al., BVM 2015): thresholded,
    smoothed Laplacian of the grey-level image, then morphological cleanup.

    Args:
        img: H x W x 3 RGB image.
        laplace_ker: aperture of the Laplacian.
        gauss_ker: Gaussian kernel size.
        gauss_sigma: Gaussian sigma.
        morph_open_ker: edge of the square kernel for the opening.
        morph_open_iter: number of opening iterations.
        morph_blur: median blur kernel size.
        min_area: smallest object (pixels) to keep.
    """
    _check_rgb(img)
    grey = cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.convertScaleAbs(cv2.Laplacian(grey, cv2.CV_16S, ksize=laplace_ker))
    smooth = cv2.GaussianBlur(lap, (gauss_ker, gauss_ker), gauss_sigma)
    _, mk = cv2.threshold(smooth, smooth.mean(), 255, cv2.THRESH_BINARY)
    mk = cv2.medianBlur(mk, morph_blur)
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN,
                          np.ones((morph_open_ker, morph_open_ker), np.uint8),
                          iterations=morph_open_iter)

    mask = skm.remove_small_objects(mk > 0, min_size=min_area)
    mask = ndimage.binary_fill_holes(mask)
    mask = skm.binary_erosion(mask, skm.disk(5))
    mask = skm.binary_opening(mask, skm.disk(5))
    return mask, None


def htk_foreground(img: np.ndarray, bandwidth: float = 2, bgnd_std: float = 2.5,
                   tissue_std: float = 30, min_peak_width: float = 10,
                   max_peak_width: float = 25, fraction: float = 0.10,
                   min_tissue_prob: float = 0.05,
                   rng: np.random.Generator | int | None = None) -> DetectorResult:
    """Foreground detection by a two-Gaussian mixture model of the
    grey-level histogram. Adapted from HistomicsTK's ``simple_mask``
    (https://github.com/DigitalSlideArchive/HistomicsTK).

    Args:
        img: H x W x 3 RGB image (uint8).
        bandwidth: KDE bandwidth for smoothing the grey-level histogram.
        bgnd_std: background Gaussian std, used if its estimation fails.
        tissue_std: tissue Gaussian std, used if its estimation fails.
        min_peak_width, max_peak_width: peak widths searched in the KDE
            histogram to initialize the fit.
        fraction: fraction of pixels sampled to build the model.
        min_tissue_prob: minimum probability for a tissue pixel.
        rng: seed or generator for the pixel sampling.
    """
    _check_rgb(img)
    rng = np.random.default_rng(rng)

    grey = (255 * color.rgb2gray(img)).astype(np.uint8)
    n = int(fraction * grey.size)
    sample = rng.choice(grey.ravel(), n)[:, np.newaxis]

    # kernel-density smoothed histogram, flipped so that the (bright)
    # background mode is on the left
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(sample)
    x_hist = np.linspace(0, 255, 256)
    y_hist = np.exp(kde.score_samples(x_hist[:, np.newaxis]))
    y_hist = np.flipud(y_hist / y_hist.sum())

    # initial means: highest peak is background, highest of the rest tissue
    peaks = signal.find_peaks_cwt(y_hist, np.arange(min_peak_width, max_peak_width))
    bg_peak = peaks[0]
    if len(peaks) > 1:
        tissue_peak = peaks[y_hist[peaks[1:]].argmax() + 1]
    else:  # no second peak: initial guess at 2/3 of the range
        tissue_peak = x_hist[int(np.round(0.66 * x_hist.size))].item()

    bg_scale = _estimate_variance(x_hist, y_hist, bg_peak)
    if bg_scale == -1:
        bg_scale = bgnd_std
    tissue_scale = _estimate_variance(x_hist, y_hist, tissue_peak)
    if tissue_scale == -1:
        tissue_scale = tissue_std

    mix = y_hist[bg_peak] * (bg_scale * (2 * np.pi) ** 0.5)

    def gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p):
        return (p * norm(loc=mu1, scale=sigma1).pdf(x)
                + (1 - p) * norm(loc=mu2, scale=sigma2).pdf(x))

    def residuals(params, y, x):
        return np.sum((y - gaussian_mixture(x, *params)) ** 2)

    mu_bg, mu_tissue, sigma_bg, sigma_tissue, p = fmin_slsqp(
        residuals, [bg_peak, tissue_peak, bg_scale, tissue_scale, mix],
        args=(y_hist, x_hist),
        bounds=[(0, 255), (0, 255), (np.spacing(1), 10), (np.spacing(1), 50), (0, 1)],
        iprint=0)

    background = norm(loc=mu_bg, scale=sigma_bg)
    tissue = norm(loc=mu_tissue, scale=sigma_tissue)

    # maximum-likelihood threshold, beyond the background mean
    diff = (1 - p) * tissue.pdf(x_hist) - p * background.pdf(x_hist)
    candidates = np.flatnonzero(diff >= 0)
    ml = x_hist[candidates[x_hist[candidates] > mu_bg][0]]

    # tissue limits (min_prob, 1 - min_prob), back in un-flipped grey levels
    endpoints = np.sort(255 - np.asarray(tissue.interval(1 - min_tissue_prob / 2)))
    ml = 255 - ml

    mask = (grey <= ml) & (grey >= endpoints[0]) & (grey <= endpoints[1])
    return mask, None


def _estimate_variance(x: np.ndarray, y: np.ndarray, peak: float) -> float:
    """Std of a histogram peak from its full width at half maximum (FWHM),
    tracing down both sides of ``peak``; if one side cannot be traced the
    FWHM is taken as twice the other half-width. Returns -1 if neither side
    can be traced. Adapted from HistomicsTK.
    """
    peak = int(peak)
    half = y[peak] / 2

    left = peak
    while left >= 0 and y[left] > half:
        left -= 1
    right = peak
    while right < y.size and y[right] > half:
        right += 1

    def left_crossing():
        slope = (y[left + 1] - y[left]) / (x[left + 1] - x[left])
        return (half - y[left]) / slope + x[left]

    def right_crossing():
        slope = (y[right] - y[right - 1]) / (x[right] - x[right - 1])
        return (half - y[right]) / slope + x[right]

    left_ok, right_ok = left != -1, right != y.size
    if left_ok and right_ok:
        return (right_crossing() - left_crossing()) / 2.355
    if right_ok:
        return 2 * (right_crossing() - x[peak]) / 2.355
    if left_ok:
        return 2 * (x[peak] - left_crossing()) / 2.355
    return -1


METHODS: dict[str, Callable[..., DetectorResult]] = {
    "fast-he": he_fast_foreground,
    "simple-he": he_simple_foreground,
    "simple": simple_foreground,
    "fesi": fesi_foreground,
    "htk": htk_foreground,
}


# ----------------------------------------------------------------------
# Tissue regions as polygons
# ----------------------------------------------------------------------

def mask_to_polygons(mask: np.ndarray, sx: float = 1.0, sy: float = 1.0, *,
                     min_area: float = 0.0, hole_area: float = 0.0,
                     tolerance: float = 0.0) -> list[Polygon]:
    """Polygons of the connected components of a boolean mask.

    Contours are traced with OpenCV, widened by half a pixel and shifted by
    (0.5, 0.5), so that a polygon follows the pixel *edges* (pixel ``i`` spans
    ``[i, i + 1]``). The polygons are scaled by ``(sx, sy)``; ``min_area``,
    ``hole_area`` (square units) and ``tolerance`` (Douglas-Peucker
    simplification) are all in these scaled coordinates. Holes smaller than
    ``hole_area`` are filled, larger ones kept; components smaller than
    ``min_area`` are dropped.

    Returns:
        polygons sorted by decreasing area, in scaled coordinates.
    """
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    xy = lambda c: c[:, 0, :].astype(float)
    out: list[Polygon] = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1 or len(c) < 3:           # not an outer contour
            continue
        holes = []
        j = hier[i][2]
        while j != -1:
            if len(contours[j]) >= 3:
                h = Polygon(xy(contours[j]))
                if h.area * sx * sy >= hole_area:
                    holes.append(xy(contours[j]))
            j = hier[j][0]
        poly = Polygon(xy(c), holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = affinity.translate(poly.buffer(0.5, join_style="mitre"), 0.5, 0.5)
        polys = [poly] if poly.geom_type == "Polygon" else list(getattr(poly, "geoms", []))
        for q in polys:
            q = affinity.scale(q, sx, sy, origin=(0, 0))
            if q.is_empty or q.area < min_area:
                continue
            if tolerance > 0:
                q = q.simplify(tolerance, preserve_topology=True)
            if not q.is_empty and q.geom_type == "Polygon":
                out.append(q)
    return sorted(out, key=lambda q: -q.area)


def tissue_regions(raster, *, method: str = "simple-he", work_mpp: float = 8.0,
                   tolerance_um: float = 10.0, min_area_mm2: float = 0.25,
                   hole_area_mm2: float = 0.1, **detector_kwargs) -> dict:
    """Tissue pieces of an imported pyramid: an outline polygon and a
    bounding box per piece, in level-0 pixel coordinates.

    The tissue is detected on the pyramid level whose resolution is closest to
    ``work_mpp`` (microns per pixel), see :func:`detect_foreground`
    (``simple-he`` keeps holes here so that ``hole_area_mm2`` decides). Holes
    below ``hole_area_mm2`` are filled, pieces below ``min_area_mm2`` dropped,
    the outlines simplified with ``tolerance_um`` (Douglas-Peucker).

    Returns:
        ``{"pieces": [{"outline": Polygon, "bbox": Polygon, "area_um2": float}],
        "info": {...}}``, pieces sorted by decreasing area; ``info`` has the
        working level and resolution, the threshold and the parameters.
    """
    mpp0 = float(raster.get_mpp_for_level(0))
    levels = range(raster.nlevels)
    k = min(levels, key=lambda lv: abs(np.log(raster.get_mpp_for_level(lv) / work_mpp)))
    mpp_k = float(raster.get_mpp_for_level(k))
    w0, h0 = raster.shape(0).width, raster.shape(0).height
    wk, hk = raster.shape(k).width, raster.shape(k).height
    sx, sy = w0 / wk, h0 / hk

    img = np.asarray(raster.get_plane(k))[..., :3]
    kwargs = dict(detector_kwargs)
    if method == "simple-he":
        kwargs.setdefault("fill_holes", False)
    mask, th = detect_foreground(img, method=method, **kwargs)

    um2 = lambda mm2: mm2 * 1e6 / (mpp0 * mpp0)       # mm^2 -> level-0 px^2
    polys = mask_to_polygons(mask, sx, sy, min_area=um2(min_area_mm2),
                             hole_area=um2(hole_area_mm2),
                             tolerance=tolerance_um / mpp0)
    pieces = []
    for p in polys:
        x0, y0, x1, y1 = p.bounds
        pieces.append({"outline": p, "bbox": box(x0, y0, x1, y1),
                       "area_um2": float(p.area * mpp0 * mpp0)})
    info = {"method": method, "threshold": None if th is None else float(th),
            "work_level": int(k), "work_mpp": mpp_k, "mask_size": [int(wk), int(hk)],
            "tolerance_um": tolerance_um, "min_area_mm2": min_area_mm2,
            "hole_area_mm2": hole_area_mm2}
    return {"pieces": pieces, "info": info}
