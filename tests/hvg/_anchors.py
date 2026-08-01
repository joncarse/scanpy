"""Sparse expected-value anchors for the deterministic HVG matrix suite.

Values were taken from a known-good run on the deterministic 1000×1000 count
matrix with ``n_top_genes=100``. They are intentionally incomplete: enough to
catch ranking/metric/selection regressions without storing full result tables.
"""

from __future__ import annotations

from typing import Any

# Genes sampled across the index for metric spot-checks.
SAMPLE_GENES = (
    "gene_0000",
    "gene_0007",
    "gene_0013",
    "gene_0042",
    "gene_0100",
    "gene_0250",
    "gene_0500",
    "gene_0750",
    "gene_0999",
)

N_TOP_GENES = 100

# Keyed by (flavor, batched).
ANCHORS: dict[tuple[str, bool], dict[str, Any]] = {
    ("seurat", False): {
        "n_highly_variable": 101,  # tie at the cutoff
        "top5_genes": [
            "gene_0311",
            "gene_0661",
            "gene_0270",
            "gene_0620",
            "gene_0970",
        ],
        "means": {
            "gene_0000": 2.4114558804854216,
            "gene_0007": 2.4117540716095727,
            "gene_0013": 2.410122081625041,
            "gene_0042": 2.4022736513812677,
            "gene_0250": 2.3948450694558003,
            "gene_0999": 2.4031114445935224,
        },
        "dispersions_norm": {
            "gene_0000": 0.06930687418914917,
            "gene_0007": -0.32581919494100287,
            "gene_0013": 0.07502225357753765,
            "gene_0042": -0.781867969954033,
            "gene_0250": 1.2196131606179355,
            "gene_0999": 1.114273610636215,
        },
        "sample_is_hvg": {
            "gene_0000": False,
            "gene_0250": False,
            "gene_0311": True,
            "gene_0999": False,
        },
    },
    ("seurat", True): {
        "n_highly_variable": 100,
        "top5_genes": [
            "gene_0307",
            "gene_0657",
            "gene_0314",
            "gene_0664",
            "gene_0300",
        ],
        "means": {
            "gene_0000": 2.411432177494018,
            "gene_0007": 2.4117430268265703,
            "gene_0250": 2.394778066006817,
            "gene_0999": 2.403063815288265,
        },
        "dispersions_norm": {
            "gene_0000": 0.60307195102455,
            "gene_0007": 0.23195937812300096,
            "gene_0250": 1.5821441384372612,
            "gene_0999": 1.5890720826165683,
        },
        "sample_is_hvg": {
            "gene_0000": False,
            "gene_0250": True,
            "gene_0307": True,
            "gene_0999": True,
        },
    },
    ("cell_ranger", False): {
        "n_highly_variable": 101,  # tie at the cutoff
        "top5_genes": [
            "gene_0027",
            "gene_0377",
            "gene_0727",
            "gene_0000",
            "gene_0350",
        ],
        "means": {
            "gene_0000": 0.5740418825149536,
            "gene_0007": 0.5745528782606125,
            "gene_0013": 0.5733780233860016,
            "gene_0042": 0.5741146738529206,
            "gene_0999": 0.5721924759149551,
        },
        "dispersions_norm": {
            "gene_0000": 5.554494450727354,
            "gene_0007": 2.2639864517353985,
            "gene_0013": 2.0773618841670856,
            "gene_0042": 0.5532455269338656,
            "gene_0999": 0.7467034943016849,
        },
        "sample_is_hvg": {
            "gene_0000": True,
            "gene_0007": True,
            "gene_0013": True,
            "gene_0042": False,
            "gene_0027": True,
        },
    },
    ("cell_ranger", True): {
        "n_highly_variable": 100,
        "top5_genes": [
            "gene_0104",
            "gene_0454",
            "gene_0804",
            "gene_0069",
            "gene_0419",
        ],
        "means": {
            "gene_0000": 0.5740418825149536,
            "gene_0007": 0.5745528782606124,
            "gene_0013": 0.5733780233860016,
            "gene_0999": 0.5721924759149551,
        },
        "dispersions_norm": {
            "gene_0000": 0.33520224479432204,
            "gene_0007": 1.0019852288014588,
            "gene_0013": -1.128348576905428,
            "gene_0999": 0.5197935038758431,
        },
        "sample_is_hvg": {
            "gene_0000": False,
            "gene_0007": False,
            "gene_0104": True,
            "gene_0999": False,
        },
    },
    ("seurat_v3", False): {
        "n_highly_variable": 100,
        "top5_genes": [
            "gene_0526",
            "gene_0176",
            "gene_0876",
            "gene_0883",
            "gene_0183",
        ],
        "top5_ranks": {
            "gene_0526": 0.0,
            "gene_0176": 1.0,
            "gene_0876": 2.0,
            "gene_0883": 3.0,
            "gene_0183": 4.0,
        },
        "means": {
            "gene_0000": 3.666,
            "gene_0007": 3.673,
            "gene_0013": 3.638,
            "gene_0042": 3.658,
            "gene_0100": 3.661,
            "gene_0526": 3.613,
            "gene_0999": 3.633,
        },
        "variances_norm": {
            "gene_0000": 0.9997527229204873,
            "gene_0007": 1.0002207858617467,
            "gene_0013": 1.0008048118524333,
            "gene_0042": 0.9971738442327458,
            "gene_0100": 1.0023815312970454,
            "gene_0526": 1.007726499479217,
            "gene_0999": 1.0018475375322167,
        },
        "sample_is_hvg": {
            "gene_0000": False,
            "gene_0526": True,
            "gene_0176": True,
            "gene_0999": False,
        },
    },
    ("seurat_v3", True): {
        "n_highly_variable": 100,
        "top5_genes": [
            "gene_0443",
            "gene_0993",
            "gene_0093",
            "gene_0952",
            "gene_0602",
        ],
        "means": {
            "gene_0000": 3.666,
            "gene_0007": 3.673,
            "gene_0100": 3.661,
            "gene_0500": 3.652,
            "gene_0999": 3.633,
        },
        "variances_norm": {
            "gene_0000": 1.0006981759895526,
            "gene_0007": 1.0009823327079341,
            "gene_0100": 1.002780047636487,
            "gene_0500": 1.001122887490042,
            "gene_0999": 1.0019210922737083,
        },
        "sample_is_hvg": {
            "gene_0000": False,
            "gene_0443": True,
            "gene_0993": True,
            "gene_0999": False,
        },
    },
    ("seurat_v3_paper", False): {
        "n_highly_variable": 100,
        "top5_genes": [
            "gene_0526",
            "gene_0176",
            "gene_0876",
            "gene_0883",
            "gene_0183",
        ],
        "top5_ranks": {
            "gene_0526": 0.0,
            "gene_0176": 1.0,
            "gene_0876": 2.0,
            "gene_0883": 3.0,
            "gene_0183": 4.0,
        },
        "means": {
            "gene_0000": 3.666,
            "gene_0526": 3.613,
            "gene_0999": 3.633,
        },
        "variances_norm": {
            "gene_0000": 0.9997527229204873,
            "gene_0526": 1.007726499479217,
            "gene_0999": 1.0018475375322167,
        },
        "sample_is_hvg": {
            "gene_0000": False,
            "gene_0526": True,
            "gene_0176": True,
        },
    },
    ("seurat_v3_paper", True): {
        "n_highly_variable": 100,
        "top5_genes": [
            "gene_0443",
            "gene_0993",
            "gene_0093",
            "gene_0952",
            "gene_0602",
        ],
        "means": {
            "gene_0000": 3.666,
            "gene_0100": 3.661,
            "gene_0999": 3.633,
        },
        "variances_norm": {
            "gene_0000": 1.0006981759895526,
            "gene_0100": 1.002780047636487,
            "gene_0999": 1.0019210922737083,
        },
        "sample_is_hvg": {
            "gene_0000": False,
            "gene_0443": True,
            "gene_0993": True,
        },
    },
}
