"""Build gene-block CSC zarr stores for Lesson 7A/7B.

Unlike ``build_fixtures.py`` (one CSC blob + read-time ``.rechunk``), this writer
splits the gene axis into on-disk CSC subgroups so each dask column-chunk maps to
a separate zarr group (``block_000``, ``block_001``, ...).

Usage
-----
    python build_geneblock_fixtures.py --dataset pbmc3k
    python build_geneblock_fixtures.py --dataset pbmc68k --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# Resolve local forks whether this suite lives at scverse/learn/... or scanpy/learn/...
_HERE = Path(__file__).resolve().parent
_SCVERSE = _HERE.parents[1]
_SCANPY = _HERE.parents[2]
if (_SCVERSE / "scanpy" / "src").is_dir():
    _SRCS = (_SCVERSE / "anndata" / "src", _SCVERSE / "scanpy" / "src")
else:
    _SRCS = (_SCANPY.parent / "anndata" / "src", _SCANPY / "src")
for _src in _SRCS:
    _src_str = str(_src)
    if _src.is_dir() and _src_str not in sys.path:
        sys.path.insert(0, _src_str)

import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
import zarr  # noqa: E402
from anndata import AnnData  # noqa: E402
from anndata.io import write_elem  # noqa: E402
from scipy.sparse import csc_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DOWNLOADS = DATA_DIR / "downloads"

DEFAULT_GENE_CHUNK = 2000
TARGET_SUM = 1e4

# 10x Fresh 68k PBMCs (Donor A), filtered gene-barcode matrices (CC BY 4.0).
PBMC68K_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "fresh_68k_pbmc_donor_a/"
    "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
)
PBMC68K_TARBALL = "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"

STORES = {
    "pbmc3k": DATA_DIR / "pbmc3k_geneblocks.zarr",
    "pbmc68k": DATA_DIR / "pbmc68k_geneblocks.zarr",
}


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (10x CDN requires a non-empty User-Agent)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "scverse-hvg-csc-dask-lesson7/1.0"},
    )
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)


def _ensure_pbmc68k_mtx() -> Path:
    """Download and extract the 10x filtered 68k matrix; return the mtx directory."""
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    tarball = DOWNLOADS / PBMC68K_TARBALL
    if not tarball.exists():
        print(f"[geneblocks] downloading {PBMC68K_URL} ...")
        _download(PBMC68K_URL, tarball)

    extract_root = DOWNLOADS / "fresh_68k_pbmc_donor_a"
    if not extract_root.exists() and not list(DOWNLOADS.rglob("matrix.mtx*")):
        print(f"[geneblocks] extracting {tarball.name} ...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(DOWNLOADS)

    # Typical layout after extract: filtered_gene_bc_matrices/hg19/
    candidates = list(DOWNLOADS.rglob("matrix.mtx")) + list(
        DOWNLOADS.rglob("matrix.mtx.gz")
    )
    if not candidates:
        msg = f"could not find matrix.mtx under {DOWNLOADS} after extract"
        raise FileNotFoundError(msg)
    mtx_dir = candidates[0].parent
    print(f"[geneblocks] 10x mtx dir: {mtx_dir}")
    return mtx_dir


def _load_dataset(name: str) -> AnnData:
    if name == "pbmc3k":
        print("[geneblocks] loading sc.datasets.pbmc3k() ...")
        return sc.datasets.pbmc3k().copy()
    if name == "pbmc68k":
        mtx_dir = _ensure_pbmc68k_mtx()
        print(f"[geneblocks] reading 10x mtx from {mtx_dir} ...")
        return sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
    msg = f"unknown dataset {name!r}"
    raise ValueError(msg)


def _prepare(adata: AnnData) -> tuple[csc_matrix, csc_matrix, AnnData]:
    """Filter, set batch labels; return (counts_csc, lognorm_csc, adata_meta)."""
    sc.pp.filter_genes(adata, min_cells=1)
    adata.var_names_make_unique()

    adata.obs["batch"] = np.tile(["a", "b"], adata.n_obs)[: adata.n_obs]
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    # Stay sparse: abs on .data only (pbmc68k must not densify).
    counts = adata.X.tocsc() if hasattr(adata.X, "tocsc") else csc_matrix(adata.X)
    counts = counts.copy()
    counts.data = np.abs(counts.data).astype(np.float32, copy=False)
    counts_csc = csc_matrix(counts)

    norm = AnnData(counts_csc.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    sc.pp.normalize_total(norm, target_sum=TARGET_SUM)
    sc.pp.log1p(norm)
    lognorm_csc = csc_matrix(norm.X)

    meta = AnnData(obs=adata.obs.copy(), var=adata.var.copy())
    return counts_csc, lognorm_csc, meta


def _write_gene_blocks(
    parent: zarr.Group, matrix: csc_matrix, *, gene_chunk: int
) -> int:
    """Write CSC slices as block_000, block_001, ... under ``parent``."""
    n_vars = matrix.shape[1]
    n_blocks = 0
    for start in range(0, n_vars, gene_chunk):
        end = min(start + gene_chunk, n_vars)
        block = matrix[:, start:end]
        if not isinstance(block, type(matrix)) or block.format != "csc":
            block = csc_matrix(block)
        name = f"block_{n_blocks:03d}"
        write_elem(parent, name, block)
        n_blocks += 1
    return n_blocks


def build(
    dataset: str,
    *,
    gene_chunk: int = DEFAULT_GENE_CHUNK,
    force: bool = False,
) -> Path:
    store = STORES[dataset]
    if store.exists() and not force:
        print(f"[geneblocks] reusing existing store: {store}")
        return store
    if store.exists():
        print(f"[geneblocks] --force: removing {store}")
        shutil.rmtree(store)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    adata = _load_dataset(dataset)
    counts_csc, lognorm_csc, meta = _prepare(adata)

    root = zarr.open_group(str(store), mode="w")
    write_elem(root, "obs", meta.obs)
    write_elem(root, "var", meta.var)

    counts_grp = root.require_group("layers").require_group("counts")
    x_grp = root.require_group("X")
    n_blocks = _write_gene_blocks(counts_grp, counts_csc, gene_chunk=gene_chunk)
    _write_gene_blocks(x_grp, lognorm_csc, gene_chunk=gene_chunk)

    root.attrs["fixture"] = {
        "dataset": dataset,
        "gene_chunk": gene_chunk,
        "n_blocks": n_blocks,
        "shape": [int(counts_csc.shape[0]), int(counts_csc.shape[1])],
        "layout": "CSC gene-block subgroups (block_000..)",
        "X": "log1p(normalize_total(1e4)), CSC gene blocks",
        "layers.counts": "raw counts, CSC gene blocks",
        "source_url": PBMC68K_URL if dataset == "pbmc68k" else "scanpy.datasets.pbmc3k()",
        "license": "CC BY 4.0 (10x) for pbmc68k; scanpy dataset for pbmc3k",
    }

    print("\n[geneblocks] manifest")
    print("-" * 60)
    print(f"  store     : {store}")
    print(f"  shape     : {counts_csc.shape}")
    print(f"  gene_chunk: {gene_chunk}")
    print(f"  n_blocks  : {n_blocks}")
    print(f"  counts    : layers/counts/block_000..{n_blocks - 1:03d}")
    print("-" * 60)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(STORES),
        required=True,
        help="pbmc3k (Lesson 7A) or pbmc68k (Lesson 7B)",
    )
    parser.add_argument("--gene-chunk", type=int, default=DEFAULT_GENE_CHUNK)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.dataset, gene_chunk=args.gene_chunk, force=args.force)


if __name__ == "__main__":
    main()
