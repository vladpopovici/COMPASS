#!/usr/bin/env python3

# Run in a properly configured environment. For example, using the spatial-omics:1.0 Docker image.

"""
Normalize 10x Visium data with SpaNorm (spatially-aware normalization).

- Within-sample: run SpaNorm on each slide separately.
- Between-sample: concatenate SpaNorm-normalized slides into one AnnData with a 'batch' column.

OUTPUT:
  - For each input slide directory: <sample_id>.spanorm.h5ad
  - If multiple slides: combined.spanorm.h5ad (concatenated object)

REQUIREMENTS (Python):
  pip install scanpy anndata numpy pandas scipy rpy2

REQUIREMENTS (R; install once in your R environment):
  install.packages("BiocManager")
  BiocManager::install(c("SpaNorm","SpatialExperiment","Matrix"))

USAGE:
  python visium_spanorm.py --slides path/to/slide1 path/to/slide2 ... \
                           --sample-ids S1 S2 ... \
                           --outdir results_spanorm \
                           [--adj-method logpac] [--sample-p 0.25] [--threads 4]

EXAMPLE:
    python visium_spanorm.py \
    --slides /workspace/sample1/outs/spatial \
            /workspace/sample2/outs/spatial \
    --sample-ids S1 S2 \
    --outdir /workspace/results_spanorm \
    --adj-method logpac \
    --sample-p 0.25 \
    --threads 4

Notes:
- adj-method: 'logpac' (default), 'pearson', 'meanbio', or 'medbio' (per SpaNorm vignette).
- sample-p: fraction of spots sampled to fit the model (SpaNorm speed/accuracy tradeoff).
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# --- rpy2 / R interop ---
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter

# Activate automatic converters
numpy2ri.activate()
pandas2ri.activate()

# Preload R libraries once
rpackages = ro.packages
utils = ro.packages.importr("utils")

# Silently try to import R packages (raises if missing)
SpaNorm = rpackages.importr("SpaNorm")
SpatialExperiment = rpackages.importr("SpatialExperiment")
Matrix = rpackages.importr("Matrix")

# R helpers
r = ro.r

def _to_spatial_experiment(counts_csr, obs_df, var_df, spatial_xy):
    """
    Build a SpatialExperiment object in R from Python AnnData parts.
    counts_csr: scipy.sparse CSR matrix of raw counts (genes x spots or spots x genes? -> AnnData is spots x genes)
    We need R assays as genes x spots, i.e., transpose.
    """
    from scipy import sparse

    if not sparse.issparse(counts_csr):
        # ensure sparse to keep memory manageable
        counts_csr = sparse.csr_matrix(counts_csr)

    # AnnData: n_obs x n_vars (spots x genes). SpaNorm expects counts as genes x spots in R (dgCMatrix).
    counts_csc = counts_csr.tocsc()
    X = counts_csc.transpose().tocsr()  # genes x spots
    X = X.astype(np.int32)

    # Build dgCMatrix in R
    i = ro.vectors.IntVector(X.indices + 1)     # 1-based
    p = ro.vectors.IntVector(X.indptr)
    x = ro.vectors.FloatVector(X.data.astype(np.float64))
    dim = ro.vectors.IntVector(list(X.shape))
    r_dgCMatrix = r["new"]("dgCMatrix", i=i, p=p, x=x, Dim=dim)

    # Spatial coords: ensure columns named 'pxl_col_in_fullres' and 'pxl_row_in_fullres' as in SpatialExperiment docs
    coords_df = pd.DataFrame(spatial_xy, columns=["pxl_col_in_fullres", "pxl_row_in_fullres"])
    with localconverter(default_converter + pandas2ri.converter):
        r_coords = ro.conversion.py2rpy(coords_df)

    # Row/col names
    with localconverter(default_converter + pandas2ri.converter):
        r_rownames = ro.conversion.py2rpy(var_df.index.astype(str))
        r_colnames = ro.conversion.py2rpy(obs_df.index.astype(str))
    r["rownames"](r_dgCMatrix, r_rownames)
    r["colnames"](r_dgCMatrix, r_colnames)

    # Create SpatialExperiment
    spe = SpatialExperiment.SpatialExperiment(assays=r.list(counts=r_dgCMatrix),
                                              spatialCoords=r_coords)
    return spe

def run_spanorm_on_adata(adata: ad.AnnData, adj_method="logpac", sample_p=0.25, threads=1) -> ad.AnnData:
    """
    Run SpaNorm on a single AnnData (Visium slide).
    Returns a *copy* with normalized values placed in:
      - .layers['spanorm_log']  (log-scale adjusted assay from SpaNorm; same scale as 'logcounts' in R)
      - .uns['spanorm'] metadata dict with parameters
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is missing. This script expects Visium data with spatial coords.")

    # Make sure counts are raw counts (integers)
    X = adata.X
    if not np.issubdtype(X.dtype, np.integer):
        # Best-effort: cast if they are counts stored as float
        X = X.astype(np.int32)

    # Build SpatialExperiment in R
    spe = _to_spatial_experiment(X, adata.obs, adata.var, adata.obsm["spatial"])

    # Call SpaNorm; returns SpatialExperiment with normalized assay in 'logcounts'
    r:contentReference[oaicite:0]{index=0}ps://www.geeksforgeeks.org/python/python-normal-distribution-in-statistics/)
    spanorm_fun = ro.r("SpaNorm::SpaNorm")
    normalized_spe = spanorm_fun(spe,
                                 adj_method=adj_method,
                                 sample_p=sample_p,
                                 BPPARAM=r("BiocParallel::SerialParam(workers=%d)" % int(threads)) if threads == 1
                                         else r("BiocParallel::MulticoreParam(workers=%d)" % int(threads)))

    # Extract logcounts (genes x spots)
    logcounts = ro.r("as.matrix")(ro.r("SummarizedExperiment::assay")(normalized_spe, "logcounts"))
    with localconverter(default_converter + numpy2ri.converter):
        logcounts_np = np.array(logcounts, dtype=np.float32)

    # Transpose back to spots x genes
    logcounts_spotsxgenes = logcounts_np.T

    out = adata.copy()
    out.layers["spanorm_log"] = logcounts_spotsxgenes

    # Keep a minimal record of settings
    out.uns.setdefault("spanorm", {})
    out.uns["spanorm"].update({"adj_method": adj_method, "sample_p": float(sample_p), "threads": int(threads)})

    return out

def read_visium_smart(slidedir: str):
    """
    Robustly read a Visium slide directory with scanpy.
    """
    slidedir = Path(slidedir)
    # Try standard read_visium first
    adata = sc.read_visium(slidedir.as_posix())
    # Ensure obs_names are unique and preserve barcodes
    adata.obs_names_make_unique()
    # Ensure spatial coords present
    if "spatial" not in adata.obsm or adata.obsm["spatial"] is None:
        raise RuntimeError(f"No spatial coordinates found in {slidedir}.")
    return adata

def main():
    ap = argparse.ArgumentParser(description="SpaNorm normalization for 10x Visium (within- & between-sample).")
    ap.add_argument("--slides", nargs="+", required=True, help="Paths to Visium slide folders.")
    ap.add_argument("--sample-ids", nargs="+", required=False, help="Optional sample IDs matching --slides.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--adj-method", default="logpac", choices=["logpac","pearson","meanbio","medbio"],
                    help="SpaNorm adjustment method (default: logpac).")
    ap.add_argument("--sample-p", type=float, default=0.25, help="Fraction of spots to fit SpaNorm model (speed/accuracy).")
    ap.add_argument("--threads", type=int, default=1, help="Workers for SpaNorm's BiocParallel backend.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    slides = [Path(s) for s in args.slides]
    if args.sample_ids:
        if len(args.sample_ids) != len(slides):
            raise SystemExit("Number of --sample-ids must match number of --slides.")
        sample_ids = args.sample_ids
    else:
        # Derive IDs from folder names
        sample_ids = [s.name for s in slides]

    normed = []
    for slidedir, sid in zip(slides, sample_ids):
        print(f"[SpaNorm] Reading {slidedir} ...")
        adata = read_visium_smart(slidedir)
        adata.obs["sample_id"] = sid

        print(f"[SpaNorm] Normalizing {sid} (spots={adata.n_obs}, genes={adata.n_vars}) ...")
        adata_sn = run_spanorm_on_adata(adata,
                                        adj_method=args.adj_method,
                                        sample_p=args.sample_p,
                                        threads=args.threads)
        # Save per-sample
        out_path = outdir / f"{sid}.spanorm.h5ad"
        print(f"[SpaNorm] Writing {out_path}")
        adata_sn.write(out_path)
        normed.append(adata_sn)

    if len(normed) > 1:
        print("[SpaNorm] Concatenating normalized slides for between-sample work ...")
        combined = ad.concat(normed, label="batch", keys=sample_ids, index_unique=None, join="outer")
        # Keep raw counts in .X (as-is from inputs), SpaNorm normalized layer in .layers['spanorm_log']
        # Downstream users can choose: e.g., combined.layers['spanorm_log'] for PCA/clustering
        combined_path = outdir / "combined.spanorm.h5ad"
        print(f"[SpaNorm] Writing {combined_path}")
        combined.write(combined_path)
        print("[SpaNorm] Done.")

if __name__ == "__main__":
    main()
