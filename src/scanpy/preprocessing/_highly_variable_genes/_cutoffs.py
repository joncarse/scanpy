from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, TypedDict

from ..._compat import warn

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from numpy.typing import NDArray

    from ..._compat import DaskArray


@dataclass
class _Cutoffs:
    min_disp: float
    max_disp: float
    min_mean: float
    max_mean: float

    @classmethod
    def validate(
        cls,
        *,
        n_top_genes: int | None,
        min_disp: float,
        max_disp: float,
        min_mean: float,
        max_mean: float,
    ) -> _Cutoffs | int:
        if n_top_genes is None:
            return cls(min_disp, max_disp, min_mean, max_mean)

        from ._main import highly_variable_genes

        cutoffs = {"min_disp", "max_disp", "min_mean", "max_mean"}
        defaults = {
            p.name: p.default
            for p in signature(highly_variable_genes).parameters.values()
            if p.name in cutoffs
        }
        if {k: v for k, v in locals().items() if k in cutoffs} != defaults:
            msg = "If you pass `n_top_genes`, all cutoffs are ignored."
            warn(msg, UserWarning)
        return n_top_genes

    def in_bounds(
        self,
        mean: NDArray[np.floating] | DaskArray,
        dispersion_norm: NDArray[np.floating] | DaskArray,
    ) -> NDArray[np.bool] | DaskArray:
        return (
            (mean > self.min_mean)
            & (mean < self.max_mean)
            & (dispersion_norm > self.min_disp)
            & (dispersion_norm < self.max_disp)
        )


class HvgArgs(TypedDict):
    cutoff: _Cutoffs | int
    n_bins: int
    flavor: Literal["seurat", "cell_ranger"]
