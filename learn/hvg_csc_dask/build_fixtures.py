"""Generate the on-disk fixtures used by the HVG CSC/dask tutorial suite.

Why this script exists
----------------------
The scanpy test tree already ships a row-chunked, CSR-ish on-disk fixture
(``scanpy/tests/_data/10x-10k-subset.zarr``). Nothing on disk represents the
**CSC, column-chunked** layout our new HVG code targets, so we materialize it
here: a single AnnData written to zarr that carries both of the realistic HVG
inputs a user would have on disk.

    pbmc3k()[:1500, :1000]  (raw counts, CSR in memory)
        -> drop all-zero genes (realistic pp.filter_genes step)
        -> set deterministic var_names + obs["batch"]
        -> layers["counts"] = raw counts as CSC          (input for seurat_v3*)
        -> X              = log1p(normalize_total) as CSC (input for seurat/cell_ranger)
        -> write_zarr(data/pbmc3k_hvg.zarr)

"Column-chunked" is NOT a property of the file: on disk we just store CSC. The
gene-axis chunking is produced at *read* time by ``read_elem_lazy`` + a
``.rechunk((-1, gene_chunk))`` (see ``conftest.py``). This keeps the fixture a
faithful picture of what a real user stores.

Usage
-----
    python build_fixtures.py            # build if missing
    python build_fixtures.py --force    # always rebuild
"""

from __future__ import annotations

import argparse
import shutil
import sys
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
from anndata import AnnData  # noqa: E402
from scipy.sparse import csc_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
STORE = DATA_DIR / "pbmc3k_hvg.zarr"

# Match the seurat_v3 subset convention in
# scanpy/tests/test_highly_variable_genes.py::test_dask_consistency. This size
# keeps the loess fit well-conditioned for seurat_v3.
N_OBS = 1500
N_VARS = 1000
TARGET_SUM = 1e4


def build(*, force: bool = False) -> Path:
    """Build (or reuse) the tutorial zarr store and return its path."""
    if STORE.exists() and not force:
        print(f"[build_fixtures] reusing existing store: {STORE}")
        return STORE
    if STORE.exists():
        print(f"[build_fixtures] --force: removing {STORE}")
        shutil.rmtree(STORE)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[build_fixtures] loading pbmc3k() ...")
    adata = sc.datasets.pbmc3k()[:N_OBS, :N_VARS].copy()

    # Realistic preprocessing: drop genes expressed in no cell. This avoids
    # degenerate all-zero columns (which are meaningless for HVG anyway).
    sc.pp.filter_genes(adata, min_cells=1)

    # Deterministic, unique var_names so downstream comparisons are stable.
    adata.var_names = [f"gene_{i:04d}" for i in range(adata.n_vars)]
    adata.var_names_make_unique()

    # Two-batch layout: the realistic "one object, several samples" case.
    adata.obs["batch"] = np.tile(["a", "b"], adata.n_obs)[: adata.n_obs]
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    # Raw counts (integer-valued, stored as CSC) -> seurat_v3 / seurat_v3_paper.
    raw = np.asarray(adata.X.todense())
    raw = np.abs(raw).astype(np.float32)
    counts_csc = csc_matrix(raw)

    # Log-normalized values (stored as CSC) -> seurat / cell_ranger.
    norm = AnnData(counts_csc.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    sc.pp.normalize_total(norm, target_sum=TARGET_SUM)
    sc.pp.log1p(norm)
    lognorm_csc = csc_matrix(norm.X)

    out = AnnData(
        X=lognorm_csc,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        layers={"counts": counts_csc},
    )
    out.uns["fixture"] = {
        "source": "scanpy.datasets.pbmc3k()",
        "subset": f"[:{N_OBS}, :{N_VARS}] then filter_genes(min_cells=1)",
        "X": "log1p(normalize_total(target_sum=1e4)), CSC",
        "layers.counts": "raw counts, CSC",
        "target_sum": TARGET_SUM,
    }

    print(f"[build_fixtures] writing {STORE} ...")
    out.write_zarr(STORE)

    _print_manifest(out)
    return STORE


def _print_manifest(adata: AnnData) -> None:
    print("\n[build_fixtures] manifest")
    print("-" * 60)
    print(f"  store          : {STORE}")
    print(f"  shape          : {adata.shape} (obs x var)")
    print(f"  X              : {type(adata.X).__name__} (CSC, log-normalized)")
    counts = adata.layers["counts"]
    print(f"  layers[counts] : {type(counts).__name__} (CSC, raw counts)")
    print(f"  obs[batch]     : {sorted(adata.obs['batch'].cat.categories)}")
    print(f"  var_names[:3]  : {list(adata.var_names[:3])}")
    print("  read as dask   : read_elem_lazy(group[...]) -> CSC dask meta,")
    print("                   then .rechunk((-1, gene_chunk)) for column chunks")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild the store even if it already exists",
    )
    args = parser.parse_args()
    build(force=args.force)


if __name__ == "__main__":
    main()
