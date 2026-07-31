"""Build gene-block CSC zarr stores for Lesson 7A/7B.

Unlike ``build_fixtures.py`` (one CSC blob + read-time ``.rechunk``), this writer
splits the gene axis into on-disk CSC subgroups so each dask column-chunk maps to
a separate zarr group (``block_000``, ``block_001``, ...).

Usage
-----
    python build_geneblock_fixtures.py --dataset pbmc3k
    python build_geneblock_fixtures.py --dataset pbmc68k --force

Lecture comments below use inspected values from a live Lesson 7B
(``pbmc68k``) rebuild.
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
    # L7-LECTURE (real Lesson 7B run)
    # topic: The 10x CDN rejects empty User-Agent strings, so we set one.
    # --- lecture ---
    # urllib's default opener often sends no User-Agent. The 10x CloudFront
    # endpoint then answers with an HTTP error instead of the tarball. A short
    # identifying User-Agent is enough; we are not pretending to be a browser.
    # --- facts at this step (Lesson 7B) ---
    #   url starts with https://cf.10xgenomics.com/samples/cell-exp/1.1.0/...
    #   dest name = fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "scverse-hvg-csc-dask-lesson7/1.0"},
    )
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)


def _ensure_pbmc68k_mtx() -> Path:
    """Download and extract the 10x filtered 68k matrix; return the mtx directory."""
    # L7-LECTURE (real Lesson 7B run)
    # topic: Before reading cells×genes, we need the extracted 10x mtx directory on disk.
    # --- lecture ---
    # Lesson 7B does not invent counts — it reuses the public 10x "Fresh 68k
    # PBMCs (Donor A)" filtered gene-barcode matrices (CC BY 4.0). The download
    # is a .tar.gz; after extract we hunt for matrix.mtx and treat its parent
    # directory as the 10x folder (barcodes + genes/features live beside it).
    #
    # On a machine that already ran the lesson once, the tarball is reused and
    # this step is nearly instant. First run downloads ~119 MiB.
    # --- facts at this step (Lesson 7B) ---
    #   url = https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
    #   tarball = data/downloads/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
    #   tarball_exists = True
    #   tarball_bytes = 124442812  (~119 MiB)
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
    # L7-LECTURE (real Lesson 7B run)
    # topic: We found the mtx directory that sc.read_10x_mtx understands.
    # --- lecture ---
    # Beside matrix.mtx sit barcodes.tsv and genes.tsv (older 10x layout) or
    # features.tsv (newer). scanpy's reader turns those into AnnData.obs index
    # (cell barcodes) and AnnData.var index (gene symbols) plus a gene_ids
    # column (Ensembl IDs).
    # --- facts at this step (Lesson 7B) ---
    #   mtx_dir = data/downloads/filtered_matrices_mex/hg19
    #   has_matrix = True
    #   has_barcodes = True
    #   has_genes = True
    print(f"[geneblocks] 10x mtx dir: {mtx_dir}")
    return mtx_dir


def _load_dataset(name: str) -> AnnData:
    if name == "pbmc3k":
        print("[geneblocks] loading sc.datasets.pbmc3k() ...")
        return sc.datasets.pbmc3k().copy()
    if name == "pbmc68k":
        mtx_dir = _ensure_pbmc68k_mtx()
        print(f"[geneblocks] reading 10x mtx from {mtx_dir} ...")
        # L7-LECTURE (real Lesson 7B run)
        # topic: sc.read_10x_mtx materializes the filtered 68k matrix as AnnData.
        # --- lecture ---
        # At this point we have NOT yet filtered genes or invented batch labels.
        # The raw 10x filtered matrix for this dataset is wider in genes than
        # the final Lesson 7B store: many genes are all-zero across cells and
        # will be dropped in _prepare.
        #
        # X arrives as CSR (cells-major), which is natural for 10x mtx. We will
        # convert to CSC later because gene-block writing and HVG want
        # gene-major slices.
        # --- facts at this step (Lesson 7B) ---
        #   adata.shape = (68579, 32738)   # cells × genes BEFORE gene filter
        #   X_type / format = csr_matrix / csr
        #   nnz = 37_323_295
        #   obs.shape = (68579, 0)         # barcode index only; no annotation columns yet
        #   obs.index head = AAACATACACCCAA-1, AAACATACCCCTCA-1, AAACATACCGGAGA-1
        #   var.shape = (32738, 1)         # columns: gene_ids (Ensembl)
        #   var.index head = MIR1302-10, FAM138A, OR4F5  (gene symbols)
        #   var['gene_ids'] head = ENSG00000243485, ENSG00000237613, ...
        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
        print(f"[geneblocks] adata.shape = {adata.shape}")
        print(f"[geneblocks] adata.X.shape = {adata.X.shape}")
        print(f"[geneblocks] adata.X.nnz = {adata.X.nnz}")
        print(f"[geneblocks] adata.obs.shape = {adata.obs.shape}")
        print(f"[geneblocks] adata.obs.index head = {adata.obs.index[:5]}")
        print(f"[geneblocks] adata.var.shape = {adata.var.shape}")
        print(f"[geneblocks] adata.var.index head = {adata.var.index[:5]}")
        print(f"[geneblocks] adata.var['gene_ids'] head = {adata.var['gene_ids'][:5]}")
        return adata
    msg = f"unknown dataset {name!r}"
    raise ValueError(msg)


def _prepare(adata: AnnData) -> tuple[csc_matrix, csc_matrix, AnnData]:
    """Filter, set batch labels; return (counts_csc, lognorm_csc, adata_meta)."""
    # L7-LECTURE (real Lesson 7B run)
    # topic: filter_genes(min_cells=1) drops genes never observed in any cell.
    # --- lecture ---
    # This is why Lesson 7B's on-disk gene count is ~20k, not the raw ~32k
    # 10x feature universe. filter_genes also writes var['n_cells'] = how many
    # cells had a nonzero count for that gene.
    sc.pp.filter_genes(adata, min_cells=1)
    adata.var_names_make_unique()
    # L7-LECTURE (real Lesson 7B run)
    # topic: After filtering, AnnData is (68579 × 20387) with richer var columns.
    # --- lecture ---
    # n_obs is unchanged (we did not filter cells). n_vars shrank from 32738
    # to 20387. var now has two columns:
    #   • gene_ids — Ensembl IDs carried from the 10x mtx
    #   • n_cells  — detection count written by filter_genes
    # The var index is still gene symbols (made unique).
    # --- facts at this step (Lesson 7B) ---
    #   adata.shape = (68579, 20387)
    #   var.columns = ['gene_ids', 'n_cells']
    #   var.index head = AL627309.1, AP006222.2, RP11-206L10.3
    #   var['n_cells'] head = 64, 4, 1, 91, ...

    # L7-LECTURE (real Lesson 7B run)
    # topic: We invent a synthetic two-level batch label on every cell.
    # --- lecture ---
    # This dataset has no real donor/batch column for our lesson purposes, but
    # scanpy's HVG APIs (and later multi-batch demos) expect obs['batch'] to
    # exist as a categorical. We tile the labels "a", "b", "a", "b", ... across
    # the cell axis so roughly half the cells are in each fake batch.
    #
    # IMPORTANT: this is teaching scaffolding, not biology. The alternating
    # a/b pattern is deterministic and visible when you inspect obs in the
    # debugger or in the written zarr.
    # --- facts at this step (Lesson 7B) ---
    #   obs.shape after = (68579, 1) with columns=['batch']
    #   batch dtype = category, categories=['a', 'b']
    #   batch value_counts ≈ a: 34290, b: 34289
    #   batch head = a, b, a, b, a, b, a, b
    adata.obs["batch"] = np.tile(["a", "b"], adata.n_obs)[: adata.n_obs]
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    # Stay sparse: touch .data only (pbmc68k must not densify).
    # L7-LECTURE (real Lesson 7B run)
    # topic: Convert X to CSC float32 counts without densifying.
    # --- lecture ---
    # CSC layout means "one column = one gene", which matches how we will slice
    # gene blocks and how seurat_v3's worker tasks prefer to see strips of genes.
    # We assert nonnegative values on the sparse .data buffer: raw 10x UMI counts
    # cannot be negative; silently taking abs() would hide a bad matrix. Then we
    # cast to float32 for a compact on-disk fixture.
    # --- facts at this step (Lesson 7B) ---
    #   counts_csc = csc_matrix shape (68579, 20387), nnz=37_323_295, dtype=float32
    counts = adata.X.tocsc() if hasattr(adata.X, "tocsc") else csc_matrix(adata.X)
    counts = counts.copy()
    if counts.nnz and np.any(counts.data < 0):
        msg = "expected nonnegative UMI counts from 10x/mtx (found negatives in X.data)"
        raise AssertionError(msg)
    counts.data = counts.data.astype(np.float32, copy=False)
    counts_csc = csc_matrix(counts)

    # L7-LECTURE (real Lesson 7B run)
    # topic: Build a parallel log-normalized matrix for layers stored under X/.
    # --- lecture ---
    # Lesson 7's HVG path uses raw counts (layers/counts). We still write a
    # log1p(normalize_total(1e4)) copy under X/ as gene blocks so the store is
    # a realistic AnnData-like fixture, not counts-only.
    # --- facts at this step (Lesson 7B) ---
    #   target_sum = 10000.0
    #   lognorm_csc = csc_matrix shape (68579, 20387), nnz=37_323_295, dtype=float32
    norm = AnnData(counts_csc.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    sc.pp.normalize_total(norm, target_sum=TARGET_SUM)
    sc.pp.log1p(norm)
    lognorm_csc = csc_matrix(norm.X)

    # L7-LECTURE (real Lesson 7B run)
    # topic: meta is an expression-free AnnData shell: obs + var only.
    # --- lecture ---
    # We will write these tables to the zarr root as obs/ and var/. That is
    # exactly what test_lesson7b later reloads in _meta_adata before attaching
    # lazy gene-block counts. So the batch column and gene_ids/n_cells columns
    # you invent here are the ones the test will see.
    # --- facts at this step (Lesson 7B) ---
    #   meta.shape = (68579, 20387)  (n_obs × n_vars; X is empty)
    #   meta.obs columns = ['batch']
    #   meta.var columns = ['gene_ids', 'n_cells']
    meta = AnnData(obs=adata.obs.copy(), var=adata.var.copy())
    return counts_csc, lognorm_csc, meta


def _write_gene_blocks(
    parent: zarr.Group, matrix: csc_matrix, *, gene_chunk: int
) -> int:
    """Write CSC slices as block_000, block_001, ... under ``parent``."""
    # L7-LECTURE (real Lesson 7B run)
    # topic: Slice the gene axis into on-disk CSC subgroups of width gene_chunk.
    # --- lecture ---
    # Each iteration writes one zarr child group via anndata.io.write_elem.
    # For Lesson 7B with gene_chunk=2000 and 20387 genes we expect
    # ceil(20387/2000)=11 blocks: block_000..block_009 are 2000 genes wide;
    # block_010 is the remainder (387 genes).
    #
    # This is the key layout difference vs Lessons 1–6: instead of one giant
    # CSC blob that we rechunk at read time, each future Dask column-chunk can
    # map 1:1 onto a separate on-disk group.
    # --- facts at this step (Lesson 7B, layers/counts pass) ---
    #   parent = /layers/counts  (second pass writes the same pattern under /X)
    #   matrix.shape = (68579, 20387), nnz=37_323_295, format=csc
    #   gene_chunk = 2000
    #   expected_n_blocks = 11
    n_vars = matrix.shape[1]
    n_blocks = 0
    for start in range(0, n_vars, gene_chunk):
        end = min(start + gene_chunk, n_vars)
        block = matrix[:, start:end]
        if not isinstance(block, type(matrix)) or block.format != "csc":
            block = csc_matrix(block)
        name = f"block_{n_blocks:03d}"
        # L7-LECTURE (real Lesson 7B run)
        # topic: write_elem stores one CSC strip as block_NNN under the parent group.
        # --- lecture ---
        # First strip (block_000) is a full-width chunk of 2000 genes across all
        # cells. Last strip (block_010) is the short remainder. nnz differs per
        # strip because genes vary in detection.
        # --- facts at this step (Lesson 7B, layers/counts) ---
        #   block_000 shape = (68579, 2000), nnz = 3_551_519
        #   block_010 shape = (68579, 387),  nnz = 1_041_421
        #   n_blocks total = 11
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
    # L7-LECTURE (real Lesson 7B run)
    # topic: build() either reuses an existing zarr or rebuilds from the download.
    # --- lecture ---
    # Lesson 7B's test looks for data/pbmc68k_geneblocks.zarr. If you renamed
    # that directory (e.g. to a backup), store.exists() is False and we rebuild
    # from the cached tarball / mtx. --force deletes an existing store first.
    # --- facts at this step (Lesson 7B rebuild) ---
    #   dataset = 'pbmc68k'
    #   store = .../learn/hvg_csc_dask/data/pbmc68k_geneblocks.zarr
    #   gene_chunk = 2000
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
    # L7-LECTURE (real Lesson 7B run)
    # topic: Persist obs/var metadata before writing the heavy gene-block matrices.
    # --- lecture ---
    # These two write_elem calls are what _meta_adata reads back in the test:
    #   obs — one column 'batch' (category a/b), index = cell barcodes
    #   var — columns gene_ids + n_cells, index = gene symbols
    # Expression is intentionally NOT in meta.X; it lives under layers/counts
    # and X as block_* groups.
    # --- facts at this step (Lesson 7B) ---
    #   meta.obs.shape = (68579, 1), columns=['batch']
    #   meta.var.shape = (20387, 2), columns=['gene_ids', 'n_cells']
    write_elem(root, "obs", meta.obs)
    write_elem(root, "var", meta.var)

    counts_grp = root.require_group("layers").require_group("counts")
    x_grp = root.require_group("X")
    n_blocks = _write_gene_blocks(counts_grp, counts_csc, gene_chunk=gene_chunk)
    _write_gene_blocks(x_grp, lognorm_csc, gene_chunk=gene_chunk)

    # L7-LECTURE (real Lesson 7B run)
    # topic: Record a small JSON-like manifest on the zarr root attrs.
    # --- lecture ---
    # Future readers (humans and tests) can open the store and see dataset,
    # gene_chunk, n_blocks, and shape without reconstructing them from the
    # directory listing. shape here is the filtered counts matrix.
    # --- facts at this step (Lesson 7B) ---
    #   fixture.dataset = pbmc68k
    #   fixture.gene_chunk = 2000
    #   fixture.n_blocks = 11
    #   fixture.shape = [68579, 20387]
    #   count blocks on disk = block_000 .. block_010
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
