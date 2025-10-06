from __future__ import annotations

from typing import List, Optional, Literal
import numpy as np
from .core import DLVH, Histogram1D, Histogram2D
from .utils import suggest_common_edges, suggest_common_edges_2d


class DLVHCohort:
    """
    Manage a cohort of DLVH objects and compute cohort-level statistics.

    - DVH-only cohort: all patients provide `dose` (LET may be None).
    - LVH-only cohort: all patients provide `let` (Dose may be None).
    - Full DLVH cohort: all patients provide both `dose` and `let`.

    Notes
    -----
    - For 1D aggregations, binning is unified across patients with
      `suggest_common_edges()` unless explicit bin_edges are given.
    - For 2D aggregations, binning is unified across patients with
      `suggest_common_edges_2d()`.
    - Aggregation is voxelwise: mean ± std, or median with IQR.
    """

    def __init__(self, dlvhs: List[DLVH]):
        if not dlvhs:
            raise ValueError("DLVHCohort requires at least one DLVH.")
        self.dlvhs = dlvhs

    # -------------------- 1D aggregation --------------------
    def aggregate_1d(
        self,
        *,
        quantity: Literal["dose", "let"],
        stat: Literal["mean", "median"] = "median",
        bin_edges: Optional[np.ndarray] = None,
        bin_width: Optional[float] = None,
        normalize: bool = True,
        cumulative: bool = True,
        **kwargs,
    ) -> Histogram1D:
        """Aggregate DVH or LVH across the cohort."""
        if quantity == "dose" and any(d.dose is None for d in self.dlvhs):
            raise RuntimeError("All patients must provide dose for DVH aggregation.")
        if quantity == "let" and any(d.let is None for d in self.dlvhs):
            raise RuntimeError("All patients must provide LET for LVH aggregation.")

        if bin_edges is None:
            arrays = [d.dose for d in self.dlvhs] if quantity == "dose" else [d.let for d in self.dlvhs]
            bin_edges = suggest_common_edges(arrays=arrays, bin_width=bin_width, **kwargs)

        maps = []
        for d in self.dlvhs:
            if quantity == "dose":
                h = d.dose_volume_histogram(bin_edges=bin_edges,
                                            normalize=normalize, cumulative=cumulative)
            else:
                h = d.let_volume_histogram(bin_edges=bin_edges,
                                           normalize=normalize, cumulative=cumulative)
            maps.append(h.values)
        stack = np.stack(maps, axis=0)

        if stat == "mean":
            agg = np.mean(stack, axis=0)
            err = np.std(stack, axis=0, ddof=1)
            p_lo = p_hi = None
        elif stat == "median":
            agg = np.median(stack, axis=0)
            p_lo = np.percentile(stack, q=25, axis=0)
            p_hi = np.percentile(stack, q=75, axis=0)
            err = None
        else:
            raise ValueError("Unsupported stat. Choose 'mean' or 'median'.")

        xlab = f"Dose [{self.dlvhs[0].dose_units}]" if quantity == "dose" else f"LET [{self.dlvhs[0].let_units}]"
        return Histogram1D(values=agg, edges=bin_edges,
                           quantity=quantity, normalize=normalize,
                           cumulative=cumulative, x_label=xlab,
                           err=err, p_lo=p_lo, p_hi=p_hi, stat=stat)

    # -------------------- 2D aggregation --------------------
    def aggregate_2d(
        self,
        *,
        stat: Literal["mean", "median"] = "median",
        dose_edges: Optional[np.ndarray] = None,
        let_edges: Optional[np.ndarray] = None,
        normalize: bool = True,
        cumulative: bool = True,
        **kwargs,
    ) -> Histogram2D:
        """Aggregate 2D Dose–LET histograms across the cohort."""
        if any((d.dose is None or d.let is None) for d in self.dlvhs):
            raise RuntimeError("All patients must provide both dose and LET for 2D aggregation.")

        if dose_edges is None or let_edges is None:
            doses = [d.dose for d in self.dlvhs]
            lets = [d.let for d in self.dlvhs]
            dose_edges, let_edges = suggest_common_edges_2d(dose_arrays=doses, let_arrays=lets, **kwargs)

        maps = []
        for d in self.dlvhs:
            h2d = d.dose_let_volume_histogram(dose_edges=dose_edges,
                                              let_edges=let_edges,
                                              normalize=normalize,
                                              cumulative=cumulative)
            maps.append(h2d.values)
        stack = np.stack(maps, axis=0)

        if stat == "mean":
            agg = np.mean(stack, axis=0)
            err = np.std(stack, axis=0, ddof=1)
            p_lo = p_hi = None
        elif stat == "median":
            agg = np.median(stack, axis=0)
            p_lo = np.percentile(stack, q=25, axis=0)
            p_hi = np.percentile(stack, q=75, axis=0)
            err = None
        else:
            raise ValueError("Unsupported stat. Choose 'mean' or 'median'.")

        return Histogram2D(values=agg,
                           dose_edges=dose_edges,
                           let_edges=let_edges,
                           normalize=normalize,
                           cumulative=cumulative,
                           dose_label=f"Dose [{self.dlvhs[0].dose_units}]",
                           let_label=f"LET [{self.dlvhs[0].let_units}]",
                           err=err, p_lo=p_lo, p_hi=p_hi, stat=stat)

    # -------------------- Marginals from 2D aggregation --------------------
    def aggregate_marginals(
        self,
        *,
        kind: Literal["dose", "let"],
        stat: Literal["mean", "median"] = "median",
        normalize: bool = True,
        cumulative: bool = True,
        **kwargs,
    ) -> Histogram1D:
        """
        Aggregate DVH or LVH derived from cohort 2D maps.

        - mean → curve media con banda ±std
        - median → curva mediana con banda IQR
        """
        h2d = self.aggregate_2d(stat=stat, normalize=normalize,
                                cumulative=cumulative, **kwargs)
        edges, values = h2d.get_marginals(kind=kind)
        xlab = h2d.dose_label if kind == "dose" else h2d.let_label
        return Histogram1D(values=values, edges=edges,
                           quantity=kind, normalize=normalize,
                           cumulative=cumulative, x_label=xlab,
                           err=h2d.err[:, 0] if stat == "mean" and h2d.err is not None and kind == "dose" else None,
                           p_lo=h2d.p_lo[:, 0] if stat == "median" and h2d.p_lo is not None and kind == "dose" else None,
                           p_hi=h2d.p_hi[:, 0] if stat == "median" and h2d.p_hi is not None and kind == "dose" else None)
