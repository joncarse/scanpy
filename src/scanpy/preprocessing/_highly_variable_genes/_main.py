from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, cast

import numpy as np
from anndata import AnnData

from ... import logging as logg
from ..._compat import warn
from ..._settings import Default
from ._cutoffs import _Cutoffs
from ._dispersion import (
    _highly_variable_genes_batched,
    _highly_variable_genes_single_batch,
)
from ._lesson7_narrate import narrate
from ._seurat_v3 import _highly_variable_genes_seurat_v3

if TYPE_CHECKING:
    import pandas as pd

    from ..._settings.presets import HVGFlavor


def highly_variable_genes(  # noqa: PLR0913
    adata: AnnData,
    *,
    layer: str | None = None,
    n_top_genes: int | None = None,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    min_mean: float = 0.0125,
    max_mean: float = 3,
    span: float = 0.3,
    n_bins: int = 20,
    flavor: HVGFlavor | Default = Default(preset=("highly_variable_genes", "flavor")),
    subset: bool = False,
    inplace: bool = True,
    batch_key: str | None = None,
    filter_unexpressed_genes: bool | None = None,
    check_values: bool = True,
) -> pd.DataFrame | None:
    """Annotate highly variable genes :cite:p:`Satija2015,Zheng2017,Stuart2019`.

    Expects logarithmized data, except when `flavor='seurat_v3'`/`'seurat_v3_paper'`, in which count
    data is expected.

    Depending on `flavor`, this reproduces the R-implementations of Seurat
    :cite:p:`Satija2015`, Cell Ranger :cite:p:`Zheng2017`, and Seurat v3 :cite:p:`Stuart2019`.

    `'seurat_v3'`/`'seurat_v3_paper'` requires `scikit-misc` package. If you plan to use this flavor, consider
    installing `scanpy` with this optional dependency: `scanpy[skmisc]`.

    For the dispersion-based methods (`flavor='seurat'` :cite:t:`Satija2015` and
    `flavor='cell_ranger'` :cite:t:`Zheng2017`), the normalized dispersion is obtained
    by scaling with the mean and standard deviation of the dispersions for genes
    falling into a given bin for mean expression of genes. This means that for each
    bin of mean expression, highly variable genes are selected.

    For `flavor='seurat_v3'`/`'seurat_v3_paper'` :cite:p:`Stuart2019`, a normalized variance for each gene
    is computed. First, the data are standardized (i.e., z-score normalization
    per feature) with a regularized standard deviation. Next, the normalized variance
    is computed as the variance of each gene after the transformation. Genes are ranked
    by the normalized variance.
    Only if `batch_key` is not `None`, the two flavors differ: For `flavor='seurat_v3'`, genes are first sorted by the median (across batches) rank, with ties broken by the number of batches a gene is a HVG.
    For `flavor='seurat_v3_paper'`, genes are first sorted by the number of batches a gene is a HVG, with ties broken by the median (across batches) rank.

    The following may help when comparing to Seurat's naming:
    If `batch_key=None` and `flavor='seurat'`, this mimics Seurat's `FindVariableFeatures(…, method='mean.var.plot')`.
    If `batch_key=None` and `flavor='seurat_v3'`/`flavor='seurat_v3_paper'`, this mimics Seurat's `FindVariableFeatures(..., method='vst')`.
    If `batch_key` is not `None` and `flavor='seurat_v3_paper'`, this mimics Seurat's `SelectIntegrationFeatures`.

    See also `scanpy.experimental.pp._highly_variable_genes` for additional flavors
    (e.g. Pearson residuals).

    .. array-support:: pp.highly_variable_genes

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    n_top_genes
        Number of highly-variable genes to keep. Mandatory if `flavor='seurat_v3'`.
    min_mean
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    max_mean
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    min_disp
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    max_disp
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored. Ignored if `flavor='seurat_v3'`.
    span
        The fraction of the data (cells) used when estimating the variance in the loess
        model fit if `flavor='seurat_v3'`.
    n_bins
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    flavor
        Choose the flavor for identifying highly variable genes
        (default depends on :attr:`scanpy.settings.preset` property :attr:`~scanpy.Preset.highly_variable_genes`).
        For the dispersion based methods in their default workflows,
        `'seurat'` passes the cutoffs whereas `'cell_ranger'` passes `n_top_genes`.
    subset
        Inplace subset to highly-variable genes if `True` otherwise merely indicate
        highly variable genes.
    inplace
        Whether to place calculated metrics in `.var` or return them.
    batch_key
        If specified, highly-variable genes are selected within each batch separately and merged.
        This simple process avoids the selection of batch-specific genes and acts as a
        lightweight batch correction method. For all flavors, except `seurat_v3`, genes are first sorted
        by how many batches they are a HVG. For dispersion-based flavors ties are broken
        by normalized dispersion. For `flavor = 'seurat_v3_paper'`, ties are broken by the median
        (across batches) rank based on within-batch normalized variance.
    filter_unexpressed_genes
        If `True`, remove genes that are not expressed in at least one cell from highly variable genes computation (does NOT remove the gene in-place).
        Disabled by default and ignored if `batch_key` is set, since filtering always enabled for batch-aware mode.
    check_values
        Check if counts in selected layer are integers. A Warning is returned if set to True.
        Only used if `flavor='seurat_v3'`/`'seurat_v3_paper'`.

    Returns
    -------
    Returns a :class:`pandas.DataFrame` with calculated metrics if `inplace=False`, else returns an `AnnData` object where it sets the following field:

    `adata.var['highly_variable']` : :class:`pandas.Series` (dtype `bool`)
        boolean indicator of highly-variable genes
    `adata.var['means']` : :class:`pandas.Series` (dtype `float`)
        means per gene
    `adata.var['dispersions']` : :class:`pandas.Series` (dtype `float`)
        For dispersion-based flavors, dispersions per gene
    `adata.var['dispersions_norm']` : :class:`pandas.Series` (dtype `float`)
        For dispersion-based flavors, normalized dispersions per gene
    `adata.var['variances']` : :class:`pandas.Series` (dtype `float`)
        For `flavor='seurat_v3'`/`'seurat_v3_paper'`, variance per gene
    `adata.var['variances_norm']`/`'seurat_v3_paper'` : :class:`pandas.Series` (dtype `float`)
        For `flavor='seurat_v3'`/`'seurat_v3_paper'`, normalized variance per gene, averaged in
        the case of multiple batches
    `adata.var['highly_variable_rank']` : :class:`pandas.Series` (dtype `float`)
        For `flavor='seurat_v3'`/`'seurat_v3_paper'`, rank of the gene according to normalized
        variance, in case of multiple batches description above
    `adata.var['highly_variable_nbatches']` : :class:`pandas.Series` (dtype `int`)
        If `batch_key` is given, this denotes in how many batches genes are detected as HVG
    `adata.var['highly_variable_intersection']` : :class:`pandas.Series` (dtype `bool`)
        If `batch_key` is given, this denotes the genes that are highly variable in all batches

    """
    if isinstance(flavor, Default):
        from ... import settings

        flavor = settings.preset.highly_variable_genes.flavor
        narrate("scanpy", "hvg: resolved Default flavor from settings", flavor=flavor)

    narrate(
        "scanpy",
        "hvg: enter highly_variable_genes",
        flavor=flavor,
        layer=layer,
        n_top_genes=n_top_genes,
        inplace=inplace,
        batch_key=batch_key,
        adata_shape=adata.shape,
    )
    start = logg.info("extracting highly variable genes")

    if not isinstance(adata, AnnData):
        msg = (
            "`pp.highly_variable_genes` expects an `AnnData` argument, "
            "pass `inplace=False` if you want to return a `pd.DataFrame`."
        )
        raise ValueError(msg)
    narrate("scanpy", "hvg: adata type check ok")

    if flavor in {"seurat_v3", "seurat_v3_paper"}:
        narrate("scanpy", "hvg: dispatch to seurat_v3 implementation", flavor=flavor)
        if n_top_genes is None:
            sig = signature(_highly_variable_genes_seurat_v3)
            n_top_genes = cast("int", sig.parameters["n_top_genes"].default)
            narrate("scanpy", "hvg: filled default n_top_genes", n_top_genes=n_top_genes)
        result = _highly_variable_genes_seurat_v3(
            adata,
            flavor=flavor,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
        )
        narrate(
            "scanpy",
            "hvg: seurat_v3 returned",
            result_type=type(result).__name__,
            result_shape=getattr(result, "shape", None),
        )
        return result

    cutoff = _Cutoffs.validate(
        n_top_genes=n_top_genes,
        min_disp=min_disp,
        max_disp=max_disp,
        min_mean=min_mean,
        max_mean=max_mean,
    )
    del min_disp, max_disp, min_mean, max_mean, n_top_genes

    if not batch_key:
        df = _highly_variable_genes_single_batch(
            adata,
            layer=layer,
            cutoff=cutoff,
            n_bins=n_bins,
            flavor=flavor,
            filter_unexpressed_genes=filter_unexpressed_genes or False,
        )
    else:
        if filter_unexpressed_genes is False:
            msg = f"filter_unexpressed_genes is set to False, but will ignored for batch-aware {flavor=!r} HVG computation"
            warn(msg, UserWarning)
        # filter_unexpressed_genes will not get passed to _highly_variable_genes_batched since it's always True for that function
        df = _highly_variable_genes_batched(
            adata, batch_key, layer=layer, cutoff=cutoff, n_bins=n_bins, flavor=flavor
        )

    logg.info("    finished", time=start)

    if not inplace:
        if subset:
            df = df.loc[df["highly_variable"]]

        return df

    adata.uns["hvg"] = {"flavor": flavor}
    logg.hint(
        "added\n"
        "    'highly_variable', boolean vector (adata.var)\n"
        "    'means', float vector (adata.var)\n"
        "    'dispersions', float vector (adata.var)\n"
        "    'dispersions_norm', float vector (adata.var)"
    )
    adata.var["highly_variable"] = df["highly_variable"]
    adata.var["means"] = df["means"]
    adata.var["dispersions"] = df["dispersions"]
    adata.var["dispersions_norm"] = df["dispersions_norm"].astype(np.float32)

    if batch_key is not None:
        adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"]
        adata.var["highly_variable_intersection"] = df["highly_variable_intersection"]
    if subset:
        adata._inplace_subset_var(df["highly_variable"])
