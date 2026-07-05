from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from fast_array_utils import stats

from ... import logging as logg
from ..._compat import DaskArray, warn
from ..._settings import Verbosity, settings
from ..._utils import sanitize_anndata
from ...get import _get_obs_rep
from .._distributed import materialize_as_ndarray
from .._simple import filter_genes
from ._cutoffs import _Cutoffs

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Concatenate, Literal, Unpack

    from anndata import AnnData
    from numpy.typing import NDArray

    from ._cutoffs import HvgArgs


def _highly_variable_genes_single_batch(
    adata: AnnData,
    *,
    layer: str | None = None,
    filter_unexpressed_genes: bool = False,
    **kwargs: Unpack[HvgArgs],
) -> pd.DataFrame:
    """See `highly_variable_genes`.

    Returns
    -------
    A DataFrame that contains the columns
    `highly_variable`, `means`, `dispersions`, and `dispersions_norm`.

    """
    cutoff = kwargs["cutoff"]
    flavor = kwargs["flavor"]
    n_bins = kwargs["n_bins"]

    x = _get_obs_rep(adata, layer=layer)

    # Filter to genes that are expressed
    if filter_unexpressed_genes:
        with settings.override(verbosity=Verbosity.error):
            # TODO use groupby or so instead of materialize_as_ndarray
            filt, _ = materialize_as_ndarray(
                filter_genes(x, min_cells=1, inplace=False)
            )
    else:
        filt = np.ones(x.shape[1], dtype=bool)

    n_removed = np.sum(~filt)
    if n_removed:
        x = x[:, filt].copy()

    if flavor == "seurat":
        x = x.copy()
        if (base := adata.uns.get("log1p", {}).get("base")) is not None:
            x *= np.log(base)
        # use out if possible. only possible since we copy the data matrix
        if isinstance(x, np.ndarray):
            np.expm1(x, out=x)
        else:
            x = np.expm1(x)

    mean, var = materialize_as_ndarray(stats.mean_var(x, axis=0, correction=1))
    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    if flavor == "seurat":  # logarithmized mean as in Seurat
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)

    # all of the following quantities are "per-gene" here
    df = pd.DataFrame(
        dict(zip(["means", "dispersions"], (mean, dispersion), strict=True))
    )
    df["mean_bin"] = _get_mean_bins(df["means"], flavor, n_bins)
    disp_stats = _get_disp_stats(df, flavor)

    # actually do the normalization
    df["dispersions_norm"] = (df["dispersions"] - disp_stats["avg"]) / disp_stats["dev"]
    df["highly_variable"] = _subset_genes(
        adata[:, filt],
        mean=mean,
        dispersion_norm=df["dispersions_norm"].to_numpy(),
        cutoff=cutoff,
    )

    df.index = adata[:, filt].var_names

    if n_removed > 0:
        # df.reset_index(drop=False, inplace=True, names=["gene"])
        # Add 0 values for genes that were filtered out
        missing_hvg = pd.DataFrame(
            np.zeros((n_removed, len(df.columns))),
            columns=df.columns,
        )
        missing_hvg["highly_variable"] = missing_hvg["highly_variable"].astype(bool)
        missing_hvg.index = adata.var_names[~filt]
        df = pd.concat([df, missing_hvg]).loc[adata.var_names]

    return df


def _get_mean_bins(
    means: pd.Series, flavor: Literal["seurat", "cell_ranger"], n_bins: int
) -> pd.Series:
    if flavor == "seurat":
        bins = n_bins
    elif flavor == "cell_ranger":
        bins = np.r_[-np.inf, np.percentile(means, np.arange(10, 105, 5)), np.inf]
    else:
        msg = '`flavor` needs to be "seurat" or "cell_ranger"'
        raise ValueError(msg)

    rv = pd.cut(means, bins=bins)
    # pandas converts Categoricals of Interval to string anyway: https://github.com/pandas-dev/pandas/issues/61928
    # As long as it does, doing it manually is more efficient
    return rv.cat.set_categories(rv.cat.categories.astype("string"), rename=True)


def _get_disp_stats(
    df: pd.DataFrame, flavor: Literal["seurat", "cell_ranger"]
) -> pd.DataFrame:
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    if flavor == "seurat":
        disp_bin_stats = disp_grouped.agg(avg="mean", dev="std")
        _postprocess_dispersions_seurat(disp_bin_stats, df["mean_bin"])
    elif flavor == "cell_ranger":
        disp_bin_stats = disp_grouped.agg(avg="median", dev=_mad)
    else:
        msg = '`flavor` needs to be "seurat" or "cell_ranger"'
        raise ValueError(msg)
    return disp_bin_stats.loc[df["mean_bin"]].set_index(df.index)


def _postprocess_dispersions_seurat(
    disp_bin_stats: pd.DataFrame, mean_bin: pd.Series
) -> None:
    # retrieve those genes that have nan std, these are the ones where
    # only a single gene fell in the bin and implicitly set them to have
    # a normalized disperion of 1
    one_gene_per_bin = disp_bin_stats["dev"].isna()
    gen_indices = np.flatnonzero(one_gene_per_bin.loc[mean_bin])
    if len(gen_indices) == 0:
        return
    logg.debug(
        f"Gene indices {gen_indices} fell into a single bin: their "
        "normalized dispersion was set to 1.\n    "
        "Decreasing `n_bins` will likely avoid this effect."
    )
    disp_bin_stats.loc[one_gene_per_bin, "dev"] = disp_bin_stats.loc[
        one_gene_per_bin, "avg"
    ]
    disp_bin_stats.loc[one_gene_per_bin, "avg"] = 0


def _mad(a):
    from statsmodels.robust import mad

    with warnings.catch_warnings():
        # MAD calculation raises the warning: "Mean of empty slice"
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return mad(a)


def _subset_genes(
    adata: AnnData,
    *,
    mean: NDArray[np.float64] | DaskArray,
    dispersion_norm: NDArray[np.float64] | DaskArray,
    cutoff: _Cutoffs | int,
) -> NDArray[np.bool] | DaskArray:
    """Get boolean mask of genes with normalized dispersion in bounds."""
    if isinstance(cutoff, _Cutoffs):
        dispersion_norm = np.nan_to_num(dispersion_norm)  # similar to Seurat
        return cutoff.in_bounds(mean, dispersion_norm)
    n_top_genes = cutoff
    del cutoff

    if n_top_genes > adata.n_vars:
        logg.info("`n_top_genes` > `adata.n_var`, returning all genes.")
        n_top_genes = adata.n_vars
    disp_cut_off = _nth_highest(dispersion_norm, n_top_genes)
    logg.debug(
        f"the {n_top_genes} top genes correspond to a "
        f"normalized dispersion cutoff of {disp_cut_off}"
    )
    return np.nan_to_num(dispersion_norm, nan=-np.inf) >= disp_cut_off


def _nth_highest(x: NDArray[np.float64] | DaskArray, n: int) -> float | DaskArray:
    x = x[~np.isnan(x)]
    if n > x.size:
        msg = (
            f"`n_top_genes` (={n}) > number of normalized dispersions (={x.size}), "
            "returning all genes with normalized dispersions."
        )
        warn(msg, UserWarning)
        n = x.size
    if isinstance(x, DaskArray):
        return x.topk(n)[-1]
    # interestingly, np.argpartition is slightly slower
    x[::-1].sort()
    return x[n - 1]


def _per_batch_func[R, **P](
    func: Callable[Concatenate[AnnData, P], R],
    adata: AnnData,
    batch_mask: pd.Series[bool],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    return func(adata[batch_mask].copy(), *args, **kwargs)


def _highly_variable_genes_batched(
    adata: AnnData, batch_key: str, *, layer: str | None, **kwargs: Unpack[HvgArgs]
) -> pd.DataFrame:
    cutoff = kwargs["cutoff"]
    sanitize_anndata(adata)
    batches = adata.obs[batch_key].cat.categories
    x = _get_obs_rep(adata, layer=layer)

    func = _per_batch_func
    if is_dask := isinstance(x, DaskArray):
        from dask import delayed

        func = delayed(_per_batch_func)

    dfs = (
        func(
            _highly_variable_genes_single_batch,
            adata=adata,
            batch_mask=adata.obs[batch_key] == batch,
            layer=layer,
            filter_unexpressed_genes=True,
            **kwargs,
        )
        for batch in batches
    )

    if is_dask:
        from dask import compute

        dfs = (compute(df)[0] for df in dfs)

    df = pd.concat(dfs, axis=0)

    df["highly_variable"] = df["highly_variable"].astype(int)
    df = df.groupby(df.index, observed=True).agg(
        dict(
            means="mean",
            dispersions="mean",
            dispersions_norm="mean",
            highly_variable="sum",
        )
    )
    df["highly_variable_nbatches"] = df["highly_variable"]
    df["highly_variable_intersection"] = df["highly_variable_nbatches"] == len(batches)

    if isinstance(cutoff, int):
        # sort genes by how often they selected as hvg within each batch and
        # break ties with normalized dispersion across batches

        df_orig_ind = adata.var.index.copy()
        df = df.sort_values(
            ["highly_variable_nbatches", "dispersions_norm"],
            ascending=False,
            na_position="last",
        )
        df["highly_variable"] = np.arange(df.shape[0]) < cutoff
        df = df.loc[df_orig_ind]
    else:
        df["dispersions_norm"] = df["dispersions_norm"].fillna(0)  # similar to Seurat
        df["highly_variable"] = cutoff.in_bounds(df["means"], df["dispersions_norm"])

    return df
