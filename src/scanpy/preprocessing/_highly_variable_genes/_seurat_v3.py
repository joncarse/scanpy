from __future__ import annotations

import os
from functools import singledispatch
from typing import TYPE_CHECKING

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from fast_array_utils import stats

from ... import logging as logg
from ..._compat import CSBase, CSCBase, CSRBase, DaskArray, warn
from ..._utils import (
    check_nonnegative_integers,
    raise_if_dask_feature_axis_chunked,
)
from ...get import _get_obs_rep, aggregate
from .._distributed import materialize_as_ndarray
from ._lesson7_narrate import narrate

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


def _raise_if_unsupported_dask_chunking(data) -> None:
    """Reject dask chunkings that seurat_v3 cannot handle.

    Row-chunked (or unchunked-feature) dask arrays are supported, as is
    column-chunked ``csc``-in-dask (the observation axis whole, feature axis
    chunked). Everything else that chunks the feature axis - dense or ``csr``
    feature-chunked, or ``csc`` chunked on both axes - is rejected with the
    standard message.
    """
    # Non-dask inputs need no chunking checks.
    if not isinstance(data, DaskArray):
        narrate("scanpy", "chunking check: not a DaskArray, accept", type=type(data))
        return
    # Feature axis unchunked (one column chunk spanning all genes).
    if data.chunksize[1] == data.shape[1]:
        narrate(
            "scanpy",
            "chunking check: feature axis unchunked, accept",
            shape=data.shape,
            chunksize=data.chunksize,
        )
        return
    # Lesson-7 path: CSC meta + full-height column chunks.
    if isinstance(data._meta, CSCBase) and data.chunksize[0] == data.shape[0]:
        narrate(
            "scanpy",
            "chunking check: column-chunked CSC dask, accept",
            shape=data.shape,
            chunksize=data.chunksize,
            numblocks=data.numblocks,
            meta_type=type(data._meta).__name__,
        )
        return
    narrate(
        "scanpy",
        "chunking check: unsupported feature-axis chunking, will raise",
        shape=data.shape,
        chunksize=data.chunksize,
        meta_type=type(data._meta).__name__,
    )
    raise_if_dask_feature_axis_chunked(data)


@singledispatch
def clip_square_sum(
    data_batch: np.ndarray, clip_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[DaskArray, DaskArray]:
    """Clip data_batch by clip_val.

    Parameters
    ----------
    data_batch
        The data to be clipped
    clip_val
        Clip by these values (must be broadcastable to the input data)

    Returns
    -------
        The clipeed data
    """
    narrate(
        "scanpy",
        "clip_square_sum: dense/ndarray path",
        data_shape=getattr(data_batch, "shape", None),
        clip_val_shape=clip_val.shape,
    )
    batch_counts = data_batch.astype(np.float64).copy()
    clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
    np.putmask(
        batch_counts,
        batch_counts > clip_val_broad,
        clip_val_broad,
    )

    squared_batch_counts_sum = np.square(batch_counts).sum(axis=0)
    batch_counts_sum = batch_counts.sum(axis=0)
    narrate(
        "scanpy",
        "clip_square_sum ndarray: sums ready",
        squared_len=len(squared_batch_counts_sum),
        sum_len=len(batch_counts_sum),
    )
    return squared_batch_counts_sum, batch_counts_sum


@clip_square_sum.register(DaskArray)
def _(data_batch: DaskArray, clip_val: np.ndarray) -> tuple[DaskArray, DaskArray]:
    narrate(
        "scanpy",
        "clip_square_sum: DaskArray dispatch",
        shape=data_batch.shape,
        chunksize=data_batch.chunksize,
        numblocks=data_batch.numblocks,
        clip_val_shape=clip_val.shape,
    )
    # Column-chunked: each block is final for its genes (Lesson 7 path).
    if data_batch.chunksize[1] != data_batch.shape[1]:
        narrate(
            "scanpy",
            "clip_square_sum: feature-chunked → _clip_square_sum_feature_chunked",
            gene_chunksize=data_batch.chunksize[1],
            n_genes=data_batch.shape[1],
        )
        return _clip_square_sum_feature_chunked(data_batch, clip_val)

    # Row-chunked: sum clipped contributions across observation blocks.
    n_blocks = data_batch.blocks.size
    narrate(
        "scanpy",
        "clip_square_sum: row-chunked map_blocks + sum",
        n_blocks=n_blocks,
    )

    def sum_and_sum_squares_clipped_from_block(block):
        return np.vstack(clip_square_sum(block, clip_val))[None, ...]

    squared_batch_counts_sum, batch_counts_sum = data_batch.map_blocks(
        sum_and_sum_squares_clipped_from_block,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (2,), (data_batch.shape[1],)),
        meta=np.array([]),
        dtype=np.float64,
    ).sum(axis=0)
    narrate(
        "scanpy",
        "clip_square_sum row-chunked: lazy sums built",
        squared_type=type(squared_batch_counts_sum).__name__,
    )
    return squared_batch_counts_sum, batch_counts_sum


def _clip_square_sum_feature_chunked(
    data_batch: DaskArray, clip_val: np.ndarray
) -> tuple[DaskArray, DaskArray]:
    narrate(
        "scanpy",
        "_clip_square_sum_feature_chunked: enter",
        shape=data_batch.shape,
        numblocks=data_batch.numblocks,
        chunks=data_batch.chunks,
    )
    if data_batch.numblocks[0] != 1:
        msg = "clip_square_sum requires the observation axis to be unchunked for feature-chunked dask arrays."
        raise ValueError(msg)

    def per_block(block, block_info: dict | None = None) -> np.ndarray:
        # Worker task: one gene-block of CSC counts.
        col_subset = slice(*block_info[0]["array-location"][1])
        narrate(
            "worker",
            "per_block: gene column slice for this chunk",
            pid=os.getpid(),
            col_subset=col_subset,
            block_shape=getattr(block, "shape", None),
            block_type=type(block).__name__,
            clip_slice_len=len(clip_val[col_subset]),
        )
        # Clip + sum on this block (often CSBase → numba path).
        squared_sum, total = clip_square_sum(block, clip_val[col_subset])
        narrate(
            "worker",
            "per_block: clipped sums for gene chunk",
            pid=os.getpid(),
            squared_sum_len=len(np.asarray(squared_sum)),
            total_len=len(np.asarray(total)),
        )
        stacked = np.vstack([np.asarray(squared_sum), np.asarray(total)])
        narrate(
            "worker",
            "per_block: stack rows [squared_sum, total] for map_blocks",
            pid=os.getpid(),
            stacked_shape=stacked.shape,
        )
        return stacked

    # Build lazy graph only — workers run per_block later at da.compute.
    combined = data_batch.map_blocks(
        per_block,
        chunks=((2,), data_batch.chunks[1]),
        meta=np.array([], dtype=np.float64),
    )
    narrate(
        "scanpy",
        "_clip_square_sum_feature_chunked: map_blocks graph built (still lazy)",
        combined_shape=combined.shape,
        combined_numblocks=combined.numblocks,
    )
    squared = combined[0]
    totals = combined[1]
    narrate(
        "scanpy",
        "_clip_square_sum_feature_chunked: split combined[0]/combined[1]",
        squared_shape=squared.shape,
        totals_shape=totals.shape,
    )
    return squared, totals


@clip_square_sum.register(CSBase)
def _(data_batch: CSBase, clip_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    narrate(
        "scanpy",
        "clip_square_sum: sparse CSBase path (→ numba)",
        shape=data_batch.shape,
        format=getattr(data_batch, "format", None),
        nnz=data_batch.nnz,
        clip_val_shape=clip_val.shape,
    )
    batch_counts = data_batch if isinstance(data_batch, CSRBase) else data_batch.tocsr()
    narrate(
        "scanpy",
        "clip_square_sum: ensure CSR for numba kernel",
        format=getattr(batch_counts, "format", None),
        nnz=batch_counts.nnz,
    )

    # Numba body cannot narrate; log around the call.
    narrate(
        "scanpy",
        "calling numba _sum_and_sum_squares_clipped",
        n_cols=batch_counts.shape[1],
        nnz=batch_counts.nnz,
    )
    out = _sum_and_sum_squares_clipped(
        batch_counts.indices,
        batch_counts.data,
        n_cols=batch_counts.shape[1],
        clip_val=clip_val,
        nnz=batch_counts.nnz,
    )
    narrate(
        "scanpy",
        "numba _sum_and_sum_squares_clipped returned",
        squared_len=len(out[0]),
        sum_len=len(out[1]),
    )
    return out


# parallel=False needed for accuracy
@numba.njit(cache=True, parallel=False)  # noqa: TID251
def _sum_and_sum_squares_clipped(
    indices: NDArray[np.integer],
    data: NDArray[np.floating],
    *,
    n_cols: int,
    clip_val: NDArray[np.float64],
    nnz: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    squared_batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    for i in numba.prange(nnz):
        idx = indices[i]
        element = min(np.float64(data[i]), clip_val[idx])
        squared_batch_counts_sum[idx] += element**2
        batch_counts_sum[idx] += element

    return squared_batch_counts_sum, batch_counts_sum


def _highly_variable_genes_seurat_v3(  # noqa: PLR0912, PLR0915
    adata: AnnData,
    *,
    flavor: Literal["seurat_v3", "seurat_v3_paper"] = "seurat_v3",
    layer: str | None = None,
    n_top_genes: int = 2000,
    batch_key: str | None = None,
    check_values: bool = True,
    span: float = 0.3,
    subset: bool = False,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """See `highly_variable_genes`.

    For further implementation details see https://www.overleaf.com/read/ckptrbgzzzpg

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields:

    highly_variable : :class:`bool`
        boolean indicator of highly-variable genes.
    **means**
        means per gene.
    **variances**
        variance per gene.
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches.
    highly_variable_rank : :class:`float`
        Rank of the gene according to normalized variance, median rank in the case of multiple batches.
    highly_variable_nbatches : :class:`int`
        If batch_key is given, this denotes in how many batches genes are detected as HVG.

    """
    narrate(
        "scanpy",
        "seurat_v3: enter",
        flavor=flavor,
        layer=layer,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        span=span,
        subset=subset,
        inplace=inplace,
        adata_shape=adata.shape,
    )
    try:
        from skmisc.loess import loess
    except ImportError as e:
        e.add_note("Please install `scikit-misc` and try again.")
        raise
    narrate("scanpy", "seurat_v3: skmisc.loess import ok")

    # Per-gene results table (index = gene names).
    df = pd.DataFrame(index=adata.var_names)
    narrate("scanpy", "seurat_v3: empty result frame", n_genes=len(df))

    # Expression matrix from X or the requested layer (may be dask).
    data = _get_obs_rep(adata, layer=layer)
    narrate(
        "scanpy",
        "seurat_v3: got expression matrix",
        layer=layer,
        data_type=type(data).__name__,
        data_shape=getattr(data, "shape", None),
        is_dask=isinstance(data, DaskArray),
        numblocks=getattr(data, "numblocks", None),
        chunksize=getattr(data, "chunksize", None),
    )

    _raise_if_unsupported_dask_chunking(data)

    if check_values and not check_nonnegative_integers(data):
        msg = f"`{flavor=!r}` expects raw count data, but non-integers were found."
        warn(msg, UserWarning)
        narrate("scanpy", "seurat_v3: non-integer warning emitted")
    else:
        narrate("scanpy", "seurat_v3: value check passed or skipped", check_values=check_values)

    # May trigger distributed compute for dask inputs (workers).
    narrate("scanpy", "seurat_v3: computing mean/var (may hit workers)", axis=0)
    df["means"], df["variances"] = stats.mean_var(data, axis=0, correction=1)
    narrate(
        "scanpy",
        "seurat_v3: mean/var done",
        mean_min=float(np.min(df["means"])),
        mean_max=float(np.max(df["means"])),
        var_min=float(np.min(df["variances"])),
        var_max=float(np.max(df["variances"])),
    )

    batch_info = (
        pd.Categorical(np.zeros(adata.shape[0], dtype=int))
        if batch_key is None
        else adata.obs[batch_key]
    )
    narrate(
        "scanpy",
        "seurat_v3: batch labels",
        batch_key=batch_key,
        n_batches=len(np.unique(batch_info)),
    )
    norm_gene_vars = []

    adata_agg = AnnData(
        X=data,
        var=pd.DataFrame(index=adata.var_names),
        obs=pd.DataFrame(
            index=adata.obs_names, data={"__hvg_v3_batch_info__": batch_info}
        ),
    )
    narrate("scanpy", "seurat_v3: built adata_agg shell for batch means/vars")

    if batch_key is not None:
        narrate("scanpy", "seurat_v3: multi-batch aggregate mean/var")
        aggregated_mean_var = aggregate(
            adata_agg, by="__hvg_v3_batch_info__", func=["mean", "var"]
        )
        aggregated_mean_var.layers["mean"], aggregated_mean_var.layers["var"] = (
            materialize_as_ndarray(
                tuple(aggregated_mean_var.layers[l] for l in ["mean", "var"])
            )
        )
        narrate("scanpy", "seurat_v3: aggregated mean/var materialized")
    else:
        # Single logical batch: reuse global means/variances.
        aggregated_mean_var = AnnData(
            var=pd.DataFrame(index=adata.var_names),
            obs=pd.DataFrame(
                index=np.array(["one"]), data={"__hvg_v3_batch_info__": np.array([0])}
            ),
            layers={
                "mean": df["means"].to_numpy().reshape((1, -1)),
                "var": df["variances"].to_numpy().reshape((1, -1)),
            },
        )
        narrate(
            "scanpy",
            "seurat_v3: single-batch aggregated_mean_var from df",
            mean_layer_shape=aggregated_mean_var.layers["mean"].shape,
        )

    batch_info = batch_info.to_numpy()
    unique_batches = np.unique(batch_info)
    narrate("scanpy", "seurat_v3: start per-batch loess/clip loop", batches=list(unique_batches))

    for b in unique_batches:
        narrate("scanpy", "seurat_v3: batch iteration", batch=b)
        data_batch = data[batch_info == b]
        narrate(
            "scanpy",
            "seurat_v3: data_batch selected",
            batch=b,
            batch_shape=getattr(data_batch, "shape", None),
            batch_type=type(data_batch).__name__,
        )
        mean, var = (
            aggregated_mean_var[
                aggregated_mean_var.obs["__hvg_v3_batch_info__"] == b
            ].layers[l]
            for l in ["mean", "var"]
        )
        if isinstance(mean, CSBase):
            mean = mean.toarray()
            narrate("scanpy", "seurat_v3: mean densified from sparse", batch=b)
        mean = mean.ravel()
        if isinstance(var, CSBase):
            var = var.toarray()
            narrate("scanpy", "seurat_v3: var densified from sparse", batch=b)
        var = var.ravel()
        narrate(
            "scanpy",
            "seurat_v3: mean/var vectors for loess",
            batch=b,
            mean_len=len(mean),
            var_len=len(var),
        )

        estimat_var = np.zeros(data.shape[1], dtype=np.float64)
        not_const = var > 0
        narrate(
            "scanpy",
            "seurat_v3: genes with positive variance",
            batch=b,
            n_not_const=int(not_const.sum()),
            n_genes=len(var),
        )
        if not_const.any():
            # Client-side loess fit of log10(var) ~ log10(mean).
            y = np.log10(var[not_const])
            x = np.log10(mean[not_const])
            narrate(
                "scanpy",
                "seurat_v3: loess fit on client",
                batch=b,
                span=span,
                n_points=len(x),
            )
            model = loess(x, y, span=span, degree=2)
            model.fit()
            estimat_var[not_const] = model.outputs.fitted_values
            narrate(
                "scanpy",
                "seurat_v3: loess fitted_values written",
                batch=b,
                estimat_var_nonzero=int(np.count_nonzero(estimat_var)),
            )

        reg_std = np.sqrt(10**estimat_var)
        narrate(
            "scanpy",
            "seurat_v3: regularized std from loess",
            batch=b,
            reg_std_min=float(np.min(reg_std)),
            reg_std_max=float(np.max(reg_std)),
        )

        # Clip thresholds as in Seurat VST.
        n_obs = data_batch.shape[0]
        clip_val = reg_std * np.sqrt(n_obs) + mean
        narrate(
            "scanpy",
            "seurat_v3: clip_val built on client",
            batch=b,
            n_obs=n_obs,
            clip_val_shape=clip_val.shape,
            clip_val_head=clip_val[:3].tolist(),
        )

        # Lazy dask graph if data_batch is dask; compute happens later.
        squared_batch_counts_sum, batch_counts_sum = clip_square_sum(
            data_batch, clip_val
        )
        narrate(
            "scanpy",
            "seurat_v3: clip_square_sum returned (maybe still lazy)",
            batch=b,
            squared_type=type(squared_batch_counts_sum).__name__,
            sum_type=type(batch_counts_sum).__name__,
            squared_is_dask=isinstance(squared_batch_counts_sum, DaskArray),
        )

        norm_gene_var = (1 / ((n_obs - 1) * np.square(reg_std))) * (
            (n_obs * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        narrate(
            "scanpy",
            "seurat_v3: norm_gene_var expression built",
            batch=b,
            norm_type=type(norm_gene_var).__name__,
            norm_is_dask=isinstance(norm_gene_var, DaskArray),
        )
        norm_gene_vars.append(norm_gene_var)
        narrate(
            "scanpy",
            "seurat_v3: appended norm_gene_var for batch",
            batch=b,
            n_collected=len(norm_gene_vars),
        )

    if any(isinstance(e, DaskArray) for e in norm_gene_vars):
        import dask.array as da

        narrate(
            "scanpy",
            "seurat_v3: da.compute — submit gene-chunk tasks; client waits",
            n_arrays=len(norm_gene_vars),
            types=[type(e).__name__ for e in norm_gene_vars],
        )
        norm_gene_vars = da.compute(*norm_gene_vars)
        narrate(
            "scanpy",
            "seurat_v3: da.compute finished",
            n_arrays=len(norm_gene_vars),
            shapes=[getattr(e, "shape", None) for e in norm_gene_vars],
        )
    else:
        narrate("scanpy", "seurat_v3: no dask arrays — skip da.compute")

    norm_gene_vars = [ngv.reshape(1, -1) for ngv in norm_gene_vars]
    narrate(
        "scanpy",
        "seurat_v3: reshaped norm_gene_vars to (1, n_genes)",
        shapes=[ngv.shape for ngv in norm_gene_vars],
    )
    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    narrate(
        "scanpy",
        "seurat_v3: concatenated batch × gene norm matrix",
        shape=norm_gene_vars.shape,
    )

    # argsort twice gives ranks; small rank = most variable.
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)
    narrate(
        "scanpy",
        "seurat_v3: per-batch ranks from -norm_gene_vars",
        ranked_shape=ranked_norm_gene_vars.shape,
    )

    # SelectIntegrationFeatures-style bookkeeping.
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    narrate(
        "scanpy",
        "seurat_v3: count batches where gene rank < n_top_genes",
        n_top_genes=n_top_genes,
        nbatches_min=int(num_batches_high_var.min()),
        nbatches_max=int(num_batches_high_var.max()),
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)
    narrate(
        "scanpy",
        "seurat_v3: median rank across batches",
        median_ranked_finite=int(np.isfinite(median_ranked).sum()),
    )

    df = df.assign(
        gene_name=df.index,
        highly_variable_nbatches=num_batches_high_var,
        highly_variable_rank=median_ranked,
        variances_norm=np.mean(norm_gene_vars, axis=0),
    )
    narrate(
        "scanpy",
        "seurat_v3: assigned rank / nbatches / variances_norm columns",
        columns=list(df.columns),
    )

    if flavor == "seurat_v3":
        sort_cols = ["highly_variable_rank", "highly_variable_nbatches"]
        sort_ascending = [True, False]
    elif flavor == "seurat_v3_paper":
        sort_cols = ["highly_variable_nbatches", "highly_variable_rank"]
        sort_ascending = [False, True]
    else:
        msg = f"Did not recognize flavor {flavor}"
        raise ValueError(msg)
    narrate(
        "scanpy",
        "seurat_v3: sort order for top genes",
        flavor=flavor,
        sort_cols=sort_cols,
        sort_ascending=sort_ascending,
    )

    sorted_index = (
        df[sort_cols]
        .sort_values(sort_cols, ascending=sort_ascending, na_position="last")
        .index
    )
    df["highly_variable"] = False
    df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True
    narrate(
        "scanpy",
        "seurat_v3: marked highly_variable top-N",
        n_top_genes=int(n_top_genes),
        n_hvg=int(df["highly_variable"].sum()),
    )

    if inplace:
        narrate("scanpy", "seurat_v3: inplace write into adata.var")
        adata.uns["hvg"] = {"flavor": flavor}
        logg.hint(
            "added\n"
            "    'highly_variable', boolean vector (adata.var)\n"
            "    'highly_variable_rank', float vector (adata.var)\n"
            "    'means', float vector (adata.var)\n"
            "    'variances', float vector (adata.var)\n"
            "    'variances_norm', float vector (adata.var)"
        )
        for to_numpy_key in [
            "highly_variable",
            "highly_variable_rank",
            "means",
            "variances",
        ]:
            adata.var[to_numpy_key] = df[to_numpy_key].to_numpy()
        adata.var["variances_norm"] = (
            df["variances_norm"].to_numpy().astype("float64", copy=False)
        )
        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df[
                "highly_variable_nbatches"
            ].to_numpy()
        if subset:
            adata._inplace_subset_var(df["highly_variable"].to_numpy())
            narrate("scanpy", "seurat_v3: subset adata to HVGs inplace")
        narrate("scanpy", "seurat_v3: return None (inplace)")
        return None

    if batch_key is None:
        df = df.drop(["highly_variable_nbatches"], axis=1)
        narrate("scanpy", "seurat_v3: drop nbatches column (no batch_key)")
    if subset:
        df = df.iloc[df["highly_variable"].to_numpy(), :]
        narrate("scanpy", "seurat_v3: subset returned frame to HVGs", nrows=len(df))

    narrate(
        "scanpy",
        "seurat_v3: return DataFrame",
        nrows=len(df),
        columns=list(df.columns),
    )
    return df
