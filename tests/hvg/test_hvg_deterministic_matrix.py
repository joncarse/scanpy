"""Deterministic HVG anchors: memory CSC and mocked ooc CSC-dask × flavors × batch.

Out-of-core data is obtained by monkeypatching ``anndata.experimental.read_elem_lazy``
so the test load path matches production (lazy read → gene-axis rechunk) without
touching the filesystem.

Absolute checks use sparse Python anchors in ``_anchors.py`` (top genes, a handful
of means/metrics, HVG membership). Full-frame mem↔CSC-dask consistency across
ARRAY_TYPES chunk layouts is covered by ``test_hvg_csc_dask_matrix``.
"""

from __future__ import annotations

import importlib.util
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import scanpy as sc
from testing.scanpy._pytest.marks import needs

if TYPE_CHECKING:
    import pandas as pd
    from pytest import MonkeyPatch


def _import_sibling(name: str):
    """Load a same-directory helper under pytest ``--import-mode=importlib``."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"hvg_det_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


det = _import_sibling("_deterministic_counts")
anchors = _import_sibling("_anchors")

FLAVORS = ["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"]
STORAGES = ["memory", "ooc"]
ATOL = 1e-4
RTOL = 1e-4


def _cases() -> list[pytest.ParameterSet]:
    cases: list[pytest.ParameterSet] = []
    for flavor in FLAVORS:
        for batched in (False, True):
            for storage in STORAGES:
                marks = []
                if flavor in det.SEURAT_V3_FLAVORS:
                    marks.append(needs.skmisc)
                if storage == "ooc":
                    marks.append(needs.dask)
                cases.append(
                    pytest.param(
                        flavor,
                        batched,
                        storage,
                        marks=marks,
                        id=f"{flavor}-{'batched' if batched else 'single'}-{storage}",
                    )
                )
    return cases


@pytest.fixture
def mock_read_elem_lazy(monkeypatch: MonkeyPatch):
    """Patch AnnData's lazy reader to return deterministic CSC dask (no filesystem)."""
    import anndata.experimental as ad_experimental

    def _fake_read_elem_lazy(elem, *args, **kwargs):
        del elem, args, kwargs
        # Single column chunk here; make_hvg_adata rechunks along genes.
        import dask.array as da

        return da.from_array(det.deterministic_csc(), chunks=(-1, det.N_VARS))

    monkeypatch.setattr(ad_experimental, "read_elem_lazy", _fake_read_elem_lazy)
    return _fake_read_elem_lazy


@contextmanager
def _ignore_expected_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            r"`n_top_genes`.*normalized dispersions",
            category=UserWarning,
        )
        yield


def _run_hvg(adata, flavor: str, *, batched: bool, n_top_genes: int) -> pd.DataFrame:
    kwargs = dict(flavor=flavor, inplace=False, n_top_genes=n_top_genes)
    if batched:
        kwargs["batch_key"] = "batch"
    with _ignore_expected_warnings():
        result = sc.pp.highly_variable_genes(adata, **kwargs)
    assert result is not None
    return result


def _assert_anchors(result: pd.DataFrame, flavor: str, *, batched: bool) -> None:
    expected = anchors.ANCHORS[(flavor, batched)]
    assert int(result["highly_variable"].sum()) == expected["n_highly_variable"]
    assert len(result) == det.N_VARS

    top5 = expected["top5_genes"]
    for gene in top5:
        assert bool(result.loc[gene, "highly_variable"]), f"{gene} should be HVG"

    if "highly_variable_rank" in result.columns:
        ranked = list(result.nsmallest(5, "highly_variable_rank").index)
        assert ranked == top5
        for gene, rank in expected.get("top5_ranks", {}).items():
            assert result.loc[gene, "highly_variable_rank"] == rank
    else:
        ranked = list(result.nlargest(5, "dispersions_norm").index)
        assert ranked == top5

    for gene, mean in expected["means"].items():
        np.testing.assert_allclose(
            result.loc[gene, "means"], mean, atol=ATOL, rtol=RTOL, err_msg=gene
        )

    metric_key = (
        "variances_norm" if "variances_norm" in expected else "dispersions_norm"
    )
    for gene, value in expected[metric_key].items():
        np.testing.assert_allclose(
            result.loc[gene, metric_key],
            value,
            atol=ATOL,
            rtol=RTOL,
            err_msg=gene,
        )

    for gene, is_hvg in expected["sample_is_hvg"].items():
        assert bool(result.loc[gene, "highly_variable"]) is is_hvg, gene


@pytest.mark.parametrize(("flavor", "batched", "storage"), _cases())
def test_hvg_deterministic_matches_anchors(
    flavor: str,
    batched: bool,
    storage: str,
    mock_read_elem_lazy,
):
    """Absolute anchors for memory CSC and mocked ooc CSC-dask (same formula).

    Running both storages against the same anchors implies mem↔ooc agreement;
    ARRAY_TYPES-based mem↔CSC-dask coverage lives in ``test_hvg_csc_dask_matrix``.
    """
    _ = mock_read_elem_lazy
    adata = det.make_hvg_adata(
        flavor=flavor,
        storage=storage,  # type: ignore[arg-type]
        batched=batched,
    )
    result = _run_hvg(
        adata, flavor, batched=batched, n_top_genes=anchors.N_TOP_GENES
    )
    _assert_anchors(result, flavor, batched=batched)
