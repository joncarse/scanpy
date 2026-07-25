# HVG on column-chunked CSC dask: a guided debugging course

A graduated set of lessons for understanding how `scanpy.pp.highly_variable_genes`
(HVG) works on **column-chunked CSC dask arrays**, using the on-disk `pbmc3k`
fixtures in this folder. Each lesson has a matching entry in
[`../../.vscode/launch.json`](../../.vscode/launch.json) (group "HVG lessons").

Work through them in order. The point is not just to pass the tests — it is to
**Step Into** the local **scanpy** and **anndata** clones and see how the pieces
connect.

## Before you start

1. **Open the scverse workspace root** (`~/scverse`) so the root `.venv` and launch
   configs resolve. Interpreter should be
   [`../../.venv/bin/python`](../../.venv/bin/python) (Python 3.14.6).

   ```bash
   cd /path/to/scverse
   python3.14 -m venv .venv
   . .venv/bin/activate
   pip install -e ./scanpy -e ./anndata
   pip install -r learn/hvg_csc_dask/requirements.txt
   ```

2. **Confirm you are on the forks**, not site-packages. In any Debug Console:

   ```python
   import scanpy, anndata
   scanpy.__file__   # .../scverse/scanpy/src/scanpy/__init__.py
   anndata.__file__  # .../scverse/anndata/src/anndata/__init__.py
   ```

3. **`justMyCode` is `false`** on every lesson launch, and `PYTHONPATH` includes both
   `scanpy/src` and `anndata/src`. You can set breakpoints in either tree and F11
   (Step Into) will land there.

### How to step into scanpy / anndata

| Technique | When to use it |
|-----------|----------------|
| **F11 / Step Into** on a call | From a test or helper into library code |
| **Open file + set breakpoint** | Jump ahead to a known function before you hit Continue |
| **Call Stack** panel | See scanpy ↔ anndata ↔ dask frames while paused |
| **Debug Console** | Inspect `type(x)`, `x.format`, `x.chunks`, `x.__file__` |

Open library files via Go to File (`Ctrl+P` / `Cmd+P`) using the paths below
(relative to this folder → scverse root):

**scanpy (HVG / pp / aggregate)**

- [`../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_main.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_main.py) — public dispatcher
- [`../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_dispersion.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_dispersion.py) — `seurat` / `cell_ranger`
- [`../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_seurat_v3.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_seurat_v3.py) — `seurat_v3*` + new dask `clip_square_sum`
- [`../../scanpy/src/scanpy/get/_aggregated.py`](../../scanpy/src/scanpy/get/_aggregated.py) — `aggregate` / `aggregate_dask`
- [`../../scanpy/src/scanpy/preprocessing/_normalization.py`](../../scanpy/src/scanpy/preprocessing/_normalization.py) — `normalize_total`
- [`../../scanpy/src/scanpy/preprocessing/_simple.py`](../../scanpy/src/scanpy/preprocessing/_simple.py) — `log1p`

**anndata (zarr IO / lazy CSC)**

- [`../../anndata/src/anndata/_core/anndata.py`](../../anndata/src/anndata/_core/anndata.py) — `AnnData.write_zarr`
- [`../../anndata/src/anndata/_io/zarr.py`](../../anndata/src/anndata/_io/zarr.py) — `write_zarr` / `read_zarr`
- [`../../anndata/src/anndata/_io/specs/registry.py`](../../anndata/src/anndata/_io/specs/registry.py) — `read_elem_lazy`, `write_elem`
- [`../../anndata/src/anndata/_io/specs/methods.py`](../../anndata/src/anndata/_io/specs/methods.py) — `write_csc` / `read_sparse` (eager CSC)
- [`../../anndata/src/anndata/_io/specs/lazy_methods.py`](../../anndata/src/anndata/_io/specs/lazy_methods.py) — `read_sparse_as_dask` (lazy CSC → dask)

**This suite (glue)**

- [`_support.py`](_support.py) — `load_lazy_csc` (read + rechunk)
- [`build_fixtures.py`](build_fixtures.py) — build + `write_zarr`

---

## Lesson 0 - Build & inspect the fixtures (anndata write path)

**Launch:** `Lesson 0: Build & inspect fixtures` (runs `build_fixtures.py --force`).

**Goal:** see what gets **written** to disk, and step into **anndata’s** CSC zarr
writer. This launch does **not** call `load_lazy_csc` (that is Lesson 1+).

**Breakpoints (set these before you start):**

1. [`build_fixtures.py`](build_fixtures.py) — `out.write_zarr(STORE)`. Inspect `out`:
   `X` and `layers["counts"]` are both scipy CSC; `obs["batch"]` is categorical.
2. **Step Into** `write_zarr` →
   [`anndata/_core/anndata.py`](../../anndata/src/anndata/_core/anndata.py)
   `AnnData.write_zarr` →
   [`anndata/_io/zarr.py`](../../anndata/src/anndata/_io/zarr.py) `write_zarr`.
3. Optional deeper: when `X` / `layers/counts` are written, land in
   [`methods.py`](../../anndata/src/anndata/_io/specs/methods.py) `write_csc` /
   `write_sparse_compressed` — that is where the on-disk `data` / `indices` /
   `indptr` groups are created.

**Observe / ask yourself:**

- On disk under `data/pbmc3k_hvg.zarr/`, `X/` and `layers/counts/` each have
  `encoding-type: csc_matrix` and three arrays (`data`, `indices`, `indptr`). There
  are **no** gene-column chunk files.
- `X` = log-normalized (for `seurat` / `cell_ranger`); `layers["counts"]` = raw
  counts (for `seurat_v3*`). Why two matrices in one AnnData?
- Where do column chunks come from, then? Only at **read** time — Lesson 1.

---

## Lesson 1 - Dispersion flavor, single batch (`seurat`)

**Launch:** `Lesson 1: Dispersion (seurat, single batch)`
(`test_dispersion_matches_in_memory[single-seurat]`).

**Goal:** first **read** of the store (anndata lazy CSC → dask) then the simplest
**scanpy** HVG path on column-chunked `X`.

**Breakpoints (set in this order, then Continue / Step Into):**

1. [`_support.py`](_support.py) `load_lazy_csc` — suite glue.
2. **Step Into** `read_elem_lazy` →
   [`registry.py`](../../anndata/src/anndata/_io/specs/registry.py) `read_elem_lazy`
   → [`lazy_methods.py`](../../anndata/src/anndata/_io/specs/lazy_methods.py)
   `read_sparse_as_dask`. Watch `is_csc` become true and a dask array with CSC
   `_meta` return. Before rechunk, `lazy.chunks` is typically one block.
3. Back in `load_lazy_csc`, after `.rechunk((-1, gene_chunk))`: `chunks[1]` ≈
   `(200, 200, 164)` and `chunksize[1] != shape[1]`. That inequality is what the
   new HVG code keys off.
4. **scanpy:** [`_main.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_main.py)
   `highly_variable_genes` — watch `flavor` skip the `seurat_v3` branch and call
   `_highly_variable_genes_single_batch`.
5. [`_dispersion.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_dispersion.py)
   `_highly_variable_genes_single_batch`:
   - `expm1` (only for `seurat`; skipped for `cell_ranger` in Lesson 2);
   - `stats.mean_var(X, axis=0)` — dask reduction over gene blocks;
   - `_get_mean_bins` / `_get_disp_stats` / `_subset_genes` / `_nth_highest`.

**Observe / ask yourself:**

- After `read_elem_lazy`, where do column chunks come from — anndata or `.rechunk`?
- `mean_var` may return dask arrays; where do they become numpy?
- Why `n_top_genes=n_vars` in the test? (Compare numeric stats, not a flip-prone
  top-N boundary.)

---

## Lesson 2 - Dispersion flavor, batched (`cell_ranger`)

**Launch:** `Lesson 2: Dispersion (cell_ranger, batched)`
(`test_dispersion_matches_in_memory[batched-cell_ranger]`).

**Goal:** see how **scanpy** wraps the single-batch path for `batch_key`, and how
`cell_ranger` differs from `seurat`. Same anndata load path as Lesson 1 (`X`).

**Breakpoints:**

1. [`_support.py`](_support.py) `load_lazy_csc` (optional if you already trust the
   read path from Lesson 1).
2. [`_main.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_main.py)
   — with `batch_key` set, dispatch goes to `_highly_variable_genes_batched`.
3. [`_dispersion.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_dispersion.py)
   `_highly_variable_genes_batched` / `_per_batch_func` — per-batch loop, then
   combine.
4. Inside `_highly_variable_genes_single_batch`, confirm **`expm1` is not taken**
   for `cell_ranger`.

**Observe / ask yourself:**

- How is the per-batch subset taken from a column-chunked dask array (row mask on
  observations; gene axis still chunked)?
- What columns does batched mode add that single-batch does not?

---

## Lesson 3 - `seurat_v3`, single batch (the new scanpy dask code)

**Launch:** `Lesson 3: seurat_v3 (single batch, new clip branch)`
(`test_seurat_v3_matches_in_memory[single-seurat_v3]`).

**Goal:** heart of the feature in **scanpy**. Raw counts from `layers["counts"]`
(again via anndata lazy read + rechunk), then the new feature-chunked
`clip_square_sum` path.

**Breakpoints:**

1. [`_support.py`](_support.py) `load_lazy_csc(..., "layers/counts")` — same anndata
   `read_sparse_as_dask` path as Lesson 1, different zarr group.
2. [`_main.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_main.py)
   — `flavor in {"seurat_v3", ...}` → `_highly_variable_genes_seurat_v3`.
3. [`_seurat_v3.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_seurat_v3.py)
   `_raise_if_unsupported_dask_chunking` — why column-chunked **CSC** with a whole
   observation axis is allowed; what still raises.
4. Same file, `_highly_variable_genes_seurat_v3`: `stats.mean_var` → `loess` →
   `clip_square_sum(data_batch, clip_val)`.
5. `clip_square_sum` (`DaskArray` registration) — when
   `chunksize[1] != shape[1]`, branches to `_clip_square_sum_feature_chunked`.
6. `_clip_square_sum_feature_chunked` — **new path**: per-block `clip_val[col_subset]`
   via `block_info`; results **concatenated** along genes (not summed across blocks
   like the row-chunked branch).

**Observe / ask yourself:**

- Why can feature-chunked blocks be concatenated? (Each block holds all
  observations for a disjoint gene set; per-gene sums are already final.)
- What breaks if you feed a row-chunked array into `_clip_square_sum_feature_chunked`?

---

## Lesson 4 - `seurat_v3`, batched (aggregate + row masking)

**Launch:** `Lesson 4: seurat_v3 (batched, aggregate + row masking)`
(`test_seurat_v3_matches_in_memory[batched-seurat_v3]`).

**Goal:** deepest **scanpy** path — `scanpy.get.aggregate` on column-chunked dask,
plus per-batch row masking. Still starts with anndata lazy counts.

**Breakpoints:**

1. [`_seurat_v3.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_seurat_v3.py)
   at `aggregate(adata_agg, by="__hvg_v3_batch_info__", func=["mean", "var"])`.
2. **Step Into** [`_aggregated.py`](../../scanpy/src/scanpy/get/_aggregated.py)
   `aggregate` → `aggregate_dask` (and helpers like `aggregate_dask_mean_var`).
   Watch the column-chunked branch aggregate per gene-block.
3. Back in `_highly_variable_genes_seurat_v3`, `data_batch = data[batch_info == b]` —
   row masking on a column-chunked dask array.
4. Optional: `sort_cols` / `sort_ascending` — `seurat_v3` vs `seurat_v3_paper`
   ranking tie-breaks.

**Observe / ask yourself:**

- How does `aggregate` keep gene-axis chunking while grouping observations?
- How do `seurat_v3` and `seurat_v3_paper` differ in sort order?

---

## Lesson 5 - Full pipeline (`normalize_total` layout flip)

**Launch:** `Lesson 5: Full pipeline (normalize_total layout flip)`
(`test_full_pipeline_matches_in_memory`).

**Goal:** start from lazy **raw** counts, run `normalize_total` → `log1p` → HVG,
and see that **in-memory scipy** and **dask** handle CSC differently mid-pipeline.

**Breakpoints:**

1. [`_support.py`](_support.py) `load_lazy_csc(..., "layers/counts")` — anndata read.
2. [`_normalization.py`](../../scanpy/src/scanpy/preprocessing/_normalization.py)
   `normalize_total`, around the CSC branch:

   ```python
   if isinstance(x, CSCBase):
       x = x.tocsr()
   ```

   That fires for **in-memory** scipy CSC. For a **dask** array with CSC `_meta`,
   this branch does not convert the same way — layout can stay CSC + column-chunked.
3. Optional: [`_simple.py`](../../scanpy/src/scanpy/preprocessing/_simple.py) `log1p`
   / `log1p_anndata`.
4. Then HVG as in Lesson 1 (`_main` → `_dispersion`, flavor `seurat`).

**Observe / ask yourself:**

- In the test, after `normalize_total`: in-memory `X` is `csr_matrix`; dask `X`
  still has CSC `_meta` and `chunksize[1] != shape[1]`. Why the difference?
- Final HVG stats still match. What does that say about what HVG depends on?

---

## Lesson 6 - Real dask executor (`LocalCluster`)

**Launch:** `Lesson 6: Real dask executor (LocalCluster)`
(`test_seurat_v3_under_local_cluster`).

**Goal:** same `seurat_v3` column-chunked computation under a real
`distributed.LocalCluster` (2 worker **processes**). Compare to Lesson 3’s
in-process scheduler.

**Breakpoints that still hit in the client process:**

1. [`conftest.py`](conftest.py) `dask_client` — cluster comes up
   (`processes=True`, `threads_per_worker=1`).
2. [`_support.py`](_support.py) `load_lazy_csc` — anndata lazy read still runs in
   the main process (graph construction).
3. [`_seurat_v3.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_seurat_v3.py)
   setup / `clip_square_sum` **dispatch** — may hit when the graph is built.
4. In the test: `with distributed.get_task_stream(...)` and
   `assert len(ts.data) > 0`.

**What usually will *not* hit:**

- Breakpoints inside `_clip_square_sum_feature_chunked` / numba kernels — those run
  on **worker processes**, not the debugger’s process. That is the mental-model
  shift vs Lesson 3.

**Observe / ask yourself:**

- Why `processes=True` and one thread per worker? (numba + multi-threaded workers;
  see scanpy’s `maybe_dask_process_context`.)
- On-disk zarr still has monolithic CSC component chunks; dask parallelism here is
  over **logical** gene blocks after `rechunk`, not separate files per gene chunk.
- Non-empty `ts.data` proves work ran on the cluster.

---

## Lesson 7A / 7B - Gene-block CSC on disk + LocalCluster

**Launch:**

- `Lesson 7A: LocalCluster pbmc3k gene-block CSC`
  (`test_lesson7a_pbmc3k_localcluster`)
- `Lesson 7B: LocalCluster pbmc68k gene-block CSC`
  (`test_lesson7b_pbmc68k_localcluster`)

**Goal:** run `seurat_v3` under the same `LocalCluster` model as Lesson 6 / the
[official scanpy dask tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/experimental/dask.html),
but with an on-disk layout where **each gene chunk is its own CSC zarr subgroup**
(`block_000`, `block_001`, …). That supplements Lessons 0–6: a plain
`write_zarr` CSC still stores `data`/`indices`/`indptr` as (typically) one chunk
each — `rechunk` only splits the **logical** dask graph.

**Why gene-block subgroups:** so `numblocks[1]` after
`da.concatenate([read_elem_lazy(block_i), ...], axis=1)` matches real on-disk
groups, not just in-memory slices of one blob. Workers can pull smaller CSC
pieces independently.

**Data:**

| Lesson | Dataset | Store |
| ------ | ------- | ----- |
| 7A | full `sc.datasets.pbmc3k()` (~2700 cells × ~32k genes) | `data/pbmc3k_geneblocks.zarr` |
| 7B | 10x Fresh 68k PBMCs (Donor A) **filtered** matrix (~68k **cells**; not `pbmc68k_reduced`) | `data/pbmc68k_geneblocks.zarr` |

Build / rebuild:

```bash
python build_geneblock_fixtures.py --dataset pbmc3k
python build_geneblock_fixtures.py --dataset pbmc68k   # downloads under data/downloads/
```

7B download is CC BY 4.0 (10x). Skip with `HVG_LESSON7_SKIP_DOWNLOAD=1`.

**Story narration (learning):** set `HVG_LESSON7_NARRATE=1` and run with pytest
`-s` (already on in the Lesson 7 launch configs). Near line-by-line stdout logs
walk the tutorial helpers, anndata `read_elem_lazy` / `read_sparse_as_dask`, and
scanpy `seurat_v3` (including worker `per_block`). Prefixes:
`[L7 client]`, `[L7 worker …]`, `[L7 scanpy]`, `[L7 anndata]`, `[L7 compare]`.
Output is intentionally very verbose; leave the env unset for quiet CI-style runs.

**Breakpoints (client process):**

1. [`build_geneblock_fixtures.py`](build_geneblock_fixtures.py) `_write_gene_blocks` —
   one `write_elem` per gene slice.
2. [`_support.py`](_support.py) `load_geneblock_csc` / `assert_on_disk_gene_blocks`.
3. [`_seurat_v3.py`](../../scanpy/src/scanpy/preprocessing/_highly_variable_genes/_seurat_v3.py)
   `_raise_if_unsupported_dask_chunking` and `clip_square_sum` **dispatch** (graph
   build). Same caveat as Lesson 6: `_clip_square_sum_feature_chunked` may not hit
   under the debugger because it runs on **worker processes**.

**Observe / ask yourself:**

- How does this differ from Lesson 6’s single CSC + `.rechunk`?
- Why is this a PR-companion path rather than replacing the official dask tutorial?
- Confirm `n_blocks >= 2` on disk and a non-empty task stream.

---

## Suggested progression recap

| Lesson | Focus | Packages you should Step Into |
| ------ | ----- | ----------------------------- |
| 0 | Write fixture to zarr | **anndata** `write_zarr` / `write_csc` |
| 1 | Lazy read + rechunk + `seurat` HVG | **anndata** `read_elem_lazy` / `read_sparse_as_dask`; **scanpy** `_dispersion` |
| 2 | Batched `cell_ranger` | **scanpy** `_highly_variable_genes_batched` |
| 3 | New `clip_square_sum` feature-chunked path | **scanpy** `_seurat_v3` (+ anndata load) |
| 4 | `aggregate_dask` + batch masking | **scanpy** `_aggregated` / `_seurat_v3` |
| 5 | `normalize_total` CSC→CSR vs dask | **scanpy** `_normalization` (+ anndata load) |
| 6 | Distributed workers | client-side **anndata**/scanpy setup; workers run off-process |
| 7A/7B | Gene-block CSC on disk + LocalCluster | **anndata** `write_elem` / `read_elem_lazy`; client-side HVG setup |
