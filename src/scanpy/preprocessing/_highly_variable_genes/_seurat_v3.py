from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from fast_array_utils import stats

from ... import logging as logg
from ..._compat import CSBase, CSRBase, DaskArray, warn
from ..._utils import (
    check_nonnegative_integers,
    raise_if_dask_feature_axis_chunked,
)
from ...get import _get_obs_rep, aggregate
from .._distributed import materialize_as_ndarray

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


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
    batch_counts = data_batch.astype(np.float64).copy()
    clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
    np.putmask(
        batch_counts,
        batch_counts > clip_val_broad,
        clip_val_broad,
    )

    squared_batch_counts_sum = np.square(batch_counts).sum(axis=0)
    batch_counts_sum = batch_counts.sum(axis=0)
    return squared_batch_counts_sum, batch_counts_sum


@clip_square_sum.register(DaskArray)
def _(data_batch: DaskArray, clip_val: np.ndarray) -> tuple[DaskArray, DaskArray]:
    if data_batch.chunksize[1] != data_batch.shape[1]:
        # feature-axis (column) chunked: each block holds every observation for a
        # subset of genes, so the per-block clipped sums are already final and are
        # concatenated along the gene axis rather than summed across blocks.
        return _clip_square_sum_feature_chunked(data_batch, clip_val)

    n_blocks = data_batch.blocks.size

    def sum_and_sum_squares_clipped_from_block(block):
        return np.vstack(clip_square_sum(block, clip_val))[None, ...]

    squared_batch_counts_sum, batch_counts_sum = data_batch.map_blocks(
        sum_and_sum_squares_clipped_from_block,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (2,), (data_batch.shape[1],)),
        meta=np.array([]),
        dtype=np.float64,
    ).sum(axis=0)
    return squared_batch_counts_sum, batch_counts_sum


def _clip_square_sum_feature_chunked(
    data_batch: DaskArray, clip_val: np.ndarray
) -> tuple[DaskArray, DaskArray]:
    if data_batch.numblocks[0] != 1:
        msg = "clip_square_sum requires the observation axis to be unchunked for feature-chunked dask arrays."
        raise ValueError(msg)

    def per_block(block, block_info: dict | None = None) -> np.ndarray:
        col_subset = slice(*block_info[0]["array-location"][1])
        squared_sum, total = clip_square_sum(block, clip_val[col_subset])
        return np.vstack([np.asarray(squared_sum), np.asarray(total)])

    combined = data_batch.map_blocks(
        per_block,
        chunks=((2,), data_batch.chunks[1]),
        meta=np.array([], dtype=np.float64),
    )
    return combined[0], combined[1]


@clip_square_sum.register(CSBase)
def _(data_batch: CSBase, clip_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch_counts = data_batch if isinstance(data_batch, CSRBase) else data_batch.tocsr()

    return _sum_and_sum_squares_clipped(
        batch_counts.indices,
        batch_counts.data,
        n_cols=batch_counts.shape[1],
        clip_val=clip_val,
        nnz=batch_counts.nnz,
    )


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
    try:
        from skmisc.loess import loess
    except ImportError as e:
        e.add_note("Please install `scikit-misc` and try again.")
        raise
    df = pd.DataFrame(index=adata.var_names)
    data = _get_obs_rep(adata, layer=layer)
    raise_if_dask_feature_axis_chunked(data)

    if check_values and not check_nonnegative_integers(data):
        msg = f"`{flavor=!r}` expects raw count data, but non-integers were found."
        warn(msg, UserWarning)

    df["means"], df["variances"] = stats.mean_var(data, axis=0, correction=1)

    batch_info = (
        pd.Categorical(np.zeros(adata.shape[0], dtype=int))
        if batch_key is None
        else adata.obs[batch_key]
    )
    norm_gene_vars = []

    adata_agg = AnnData(
        X=data,
        var=pd.DataFrame(index=adata.var_names),
        obs=pd.DataFrame(
            index=adata.obs_names, data={"__hvg_v3_batch_info__": batch_info}
        ),
    )
    if batch_key is not None:
        aggregated_mean_var = aggregate(
            adata_agg, by="__hvg_v3_batch_info__", func=["mean", "var"]
        )
        aggregated_mean_var.layers["mean"], aggregated_mean_var.layers["var"] = (
            materialize_as_ndarray(
                tuple(aggregated_mean_var.layers[l] for l in ["mean", "var"])
            )
        )
    else:
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
    batch_info = batch_info.to_numpy()
    for b in np.unique(batch_info):
        data_batch = data[batch_info == b]
        mean, var = (
            aggregated_mean_var[
                aggregated_mean_var.obs["__hvg_v3_batch_info__"] == b
            ].layers[l]
            for l in ["mean", "var"]
        )
        if isinstance(mean, CSBase):
            mean = mean.toarray()
        mean = mean.ravel()
        if isinstance(var, CSBase):
            var = var.toarray()
        var = var.ravel()
        estimat_var = np.zeros(data.shape[1], dtype=np.float64)
        if (not_const := var > 0).any():
            y = np.log10(var[not_const])
            x = np.log10(mean[not_const])
            model = loess(x, y, span=span, degree=2)
            model.fit()
            estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10**estimat_var)

        # clip large values as in Seurat
        n_obs = data_batch.shape[0]
        clip_val = reg_std * np.sqrt(n_obs) + mean
        squared_batch_counts_sum, batch_counts_sum = clip_square_sum(
            data_batch, clip_val
        )

        norm_gene_var = (1 / ((n_obs - 1) * np.square(reg_std))) * (
            (n_obs * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        norm_gene_vars.append(norm_gene_var)
    if any(isinstance(e, DaskArray) for e in norm_gene_vars):
        import dask.array as da

        norm_gene_vars = da.compute(*norm_gene_vars)

    norm_gene_vars = [ngv.reshape(1, -1) for ngv in norm_gene_vars]
    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)

    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)

    df = df.assign(
        gene_name=df.index,
        highly_variable_nbatches=num_batches_high_var,
        highly_variable_rank=median_ranked,
        variances_norm=np.mean(norm_gene_vars, axis=0),
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
    sorted_index = (
        df[sort_cols]
        .sort_values(sort_cols, ascending=sort_ascending, na_position="last")
        .index
    )
    df["highly_variable"] = False
    df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True

    if inplace:
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
    else:
        if batch_key is None:
            df = df.drop(["highly_variable_nbatches"], axis=1)
        if subset:
            df = df.iloc[df["highly_variable"].to_numpy(), :]

        return df
    return None
