from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Literal, Optional
from .utils import _auto_bins, _suffix_cumsum2d


class Histogram1D:
    """
    Internal 1D histogram container with plotting capability.
    Not part of the public API.
    """

    def __init__(self, *, values: np.ndarray, edges: np.ndarray,
                 quantity: str, normalize: bool, cumulative: bool,
                 x_label: Optional[str] = None, y_unit: Optional[str] = None):
        self.values = np.asarray(values, dtype=float)
        self.edges = np.asarray(edges, dtype=float)
        self.quantity = str(quantity)
        self.normalize = bool(normalize)
        self.cumulative = bool(cumulative)
        self.x_label = x_label
        self.y_unit = y_unit

    def get_data(self, *, x: Literal["edges", "centers"] = "edges") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return x-axis coordinates and histogram values.

        Parameters
        ----------
        x : {"edges", "centers"}, default="edges"
            - "edges": return (edges, values). If cumulative, auto-pad to 0 for
              visualization consistency (prepend 0 to edges and duplicate first value).
            - "centers": return (centers, values). No padding applied.
        """
        edges = self.edges.copy()
        values = self.values.copy()

        if x == "centers":
            return 0.5 * (edges[:-1] + edges[1:]), values

        if self.cumulative and edges[0] > 0:
            edges = np.insert(edges, 0, 0.0)
            values = np.insert(values, 0, values[0])

        return edges, values

    def plot(self, *, ax: Optional[plt.Axes] = None, **kwargs):
        """Plot the histogram (cumulative curve or differential bar chart)."""
        edges, values = self.get_data(x="edges")

        if ax is None:
            _, ax = plt.subplots()

        if self.cumulative:
            ax.step(edges[:-1], values, where="post", **kwargs)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        else:
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            ax.bar(centers, values, width=widths, align="center", **kwargs)
            ax.set_ylim(bottom=0)

        # Labels
        if self.x_label:
            ax.set_xlabel(self.x_label)
        elif self.quantity == "dose":
            ax.set_xlabel("Dose")
        elif self.quantity == "let":
            ax.set_xlabel("LET")

        if self.normalize:
            ax.set_ylabel(f"{'Cumulative' if self.cumulative else 'Differential'} volume [%]")
        else:
            ax.set_ylabel("Volume [cm³]")

        return ax


class Histogram2D:
    """
    Internal 2D histogram container with plotting capability.
    Not part of the public API.
    """

    def __init__(self, *, values: np.ndarray,
                 dose_edges: np.ndarray, let_edges: np.ndarray,
                 normalize: bool, cumulative: bool,
                 dose_label: str = "Dose [Gy]",
                 let_label: str = "LET [keV/µm]"):
        self.values = np.asarray(values, dtype=float)
        self.dose_edges = np.asarray(dose_edges, dtype=float)
        self.let_edges = np.asarray(let_edges, dtype=float)
        self.normalize = bool(normalize)
        self.cumulative = bool(cumulative)
        self.dose_label = dose_label
        self.let_label = let_label

    def plot(self, *, ax: Optional[plt.Axes] = None,
             cmap: str = "viridis", colorbar: bool = True, **kwargs):
        """Plot the 2D histogram as a heatmap."""
        if ax is None:
            _, ax = plt.subplots()

        mesh = ax.pcolormesh(self.dose_edges,
                             self.let_edges,
                             self.values.T,
                             cmap=cmap, **kwargs)
        ax.set_xlabel(self.dose_label)
        ax.set_ylabel(self.let_label)

        if self.cumulative:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        if colorbar:
            import matplotlib.pyplot as plt
            plt.colorbar(mesh, ax=ax,
                         label="Volume [%]" if self.normalize else "Volume [cm³]")
        return ax

    def get_marginals(self, *, kind: Literal["dose", "let"] = "dose") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the marginal histogram as (edges, values). Only for cumulative 2D.
        """
        if not self.cumulative:
            raise NotImplementedError(
                "Marginal extraction is only implemented for cumulative 2D histograms. "
            )

        if kind == "dose":
            edges = self.dose_edges.copy()
            values = self.values[:, 0].copy()  # LET ≥ 0
        elif kind == "let":
            edges = self.let_edges.copy()
            values = self.values[0, :].copy()  # Dose ≥ 0
        else:
            raise ValueError("Argument 'kind' must be either 'dose' or 'let'.")

        if edges[0] > 0:
            edges = np.insert(edges, 0, 0.0)
            values = np.insert(values, 0, values[0])

        return edges, values

    def plot_marginals(self, *, kind: Literal["dose", "let"] = "dose"):
        """Plot DVH or LVH derived from the cumulative 2D histogram."""
        edges, values = self.get_marginals(kind=kind)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))

        if kind == "dose":
            ax.step(edges[:-1], values, where="post", color="C0", label="DVH")
            ax.set_xlabel(self.dose_label)
            ax.set_title("DVH from 2D cumulative")
        else:
            ax.step(edges[:-1], values, where="post", color="C1", label="LVH")
            ax.set_xlabel(self.let_label)
            ax.set_title("LVH from 2D cumulative")

        ax.set_ylabel("Volume [%]" if self.normalize else "Volume [cm³]")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        return ax


class DLVH:
    """
    Dose–LET Volume Histogram (DLVH).

    Stores paired dose and LET values for a specific ROI, already masked.
    The i-th dose value is associated with the i-th LET value and (optionally)
    a relative volume weight w_i such that sum_i w_i = 1.
    """

    def __init__(self, *,
                 dose: np.ndarray,
                 let: np.ndarray,
                 volume_cc: float,
                 relative_volumes: Optional[np.ndarray] = None,
                 dose_units: str = "Gy",
                 let_units: str = "keV/µm"):
        self.dose = self._validate_array(dose, "dose")
        self.let = self._validate_array(let, "let")

        if self.dose.shape != self.let.shape:
            raise ValueError("Dose and LET arrays must have the same shape.")

        if volume_cc <= 0 or not np.isfinite(volume_cc):
            raise ValueError("Volume must be a positive finite value (cm³).")
        self.volume_cc = float(volume_cc)

        # --- relative volume weights (sum to 1) ---
        if relative_volumes is None:
            relw = np.full(self.dose.size, 1.0 / self.dose.size, dtype=float)
        else:
            if not isinstance(relative_volumes, np.ndarray):
                raise TypeError("relative_volumes must be a numpy.ndarray.")
            relw = np.asarray(relative_volumes, dtype=float).ravel()
            if relw.shape != self.dose.shape:
                raise ValueError("relative_volumes must have the same shape as dose/let.")
            if np.any(~np.isfinite(relw)) or np.any(relw < 0):
                raise ValueError("relative_volumes must be finite and non-negative.")
            sumw = float(relw.sum())
            if sumw <= 0:
                raise ValueError("Sum of relative_volumes must be > 0.")
            relw = relw / sumw
        self.relw = relw

        self.n_voxels = int(self.dose.size)
        self.dose_units = dose_units
        self.let_units = let_units

    @staticmethod
    def _validate_array(arr: np.ndarray, label: str) -> np.ndarray:
        """Ensure array is numpy.ndarray, flattened, non-empty, and non-negative."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{label} must be a numpy.ndarray (got {type(arr)}).")
        arr = arr.ravel()
        if arr.size == 0:
            raise ValueError(f"{label} array cannot be empty.")
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{label} array contains non-finite values.")
        if np.any(arr < 0):
            raise ValueError(f"{label} array must contain only non-negative values.")
        return arr

    # ---------- internal builder ----------
    def _volume_histogram(self, *, data: np.ndarray, quantity: str,
                          bin_width: Optional[float] = None,
                          bin_edges: Optional[np.ndarray] = None,
                          normalize: bool = True,
                          cumulative: bool = True) -> Histogram1D:
        # Binning
        if bin_edges is not None:
            edges = np.asarray(bin_edges, dtype=float)
        elif bin_width is None:
            edges = _auto_bins(arr=data)
        else:
            xmax = float(np.max(data))
            n_bins = int(np.ceil(xmax / bin_width)) if bin_width > 0 else 1
            n_bins = max(n_bins, 1)
            edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)

        # Weighted differential volumes per bin (absolute cm³)
        vol_weights = self.relw * self.volume_cc
        vols, _ = np.histogram(data, bins=edges, weights=vol_weights)

        # Cumulative (right) if requested: V(X ≥ x)
        if cumulative:
            vols = np.cumsum(vols[::-1])[::-1]

        values = vols.astype(float)
        if normalize:
            values = (values / self.volume_cc) * 100.0

        xlab = f"Dose [{self.dose_units}]" if quantity == "dose" else f"LET [{self.let_units}]"
        return Histogram1D(values=values, edges=edges,
                           quantity=quantity,
                           normalize=normalize, cumulative=cumulative,
                           x_label=xlab)

    # ---------- public API ----------
    def dose_volume_histogram(self, *, bin_width: Optional[float] = None,
                              bin_edges: Optional[np.ndarray] = None,
                              normalize: bool = True,
                              cumulative: bool = True,
                              let_threshold: float = 0.0) -> Histogram1D:
        """
        Compute and return a DVH as Histogram1D object.
        """
        mask = self.let >= let_threshold
        data = self.dose[mask]
        relw_sub = self.relw[mask]
        old_relw = self.relw
        try:
            self.relw = relw_sub
            return self._volume_histogram(data=data, quantity="dose",
                                          bin_width=bin_width, bin_edges=bin_edges,
                                          normalize=normalize, cumulative=cumulative)
        finally:
            self.relw = old_relw

    def let_volume_histogram(self, *, bin_width: Optional[float] = None,
                             bin_edges: Optional[np.ndarray] = None,
                             normalize: bool = True,
                             cumulative: bool = True,
                             dose_threshold: float = 0.0) -> Histogram1D:
        """
        Compute and return a LVH as Histogram1D object.
        """
        mask = self.dose >= dose_threshold
        data = self.let[mask]
        relw_sub = self.relw[mask]
        old_relw = self.relw
        try:
            self.relw = relw_sub
            return self._volume_histogram(data=data, quantity="let",
                                          bin_width=bin_width, bin_edges=bin_edges,
                                          normalize=normalize, cumulative=cumulative)
        finally:
            self.relw = old_relw

    def dose_let_volume_histogram(self, *,
                                  bin_width_dose: Optional[float] = None,
                                  bin_width_let: Optional[float] = None,
                                  dose_edges: Optional[np.ndarray] = None,
                                  let_edges: Optional[np.ndarray] = None,
                                  normalize: bool = True,
                                  cumulative: bool = True) -> 'Histogram2D':
        """
        Compute and return a Dose–LET Volume Histogram (2D).
        """
        # --- dose binning ---
        if dose_edges is not None:
            d_edges = np.asarray(dose_edges, dtype=float)
        elif bin_width_dose is None:
            d_edges = _auto_bins(arr=self.dose)
        else:
            dmax = float(np.max(self.dose))
            nd = int(np.ceil(dmax / bin_width_dose))
            nd = max(nd, 1)
            d_edges = np.linspace(0.0, nd * bin_width_dose, nd + 1)

        # --- let binning ---
        if let_edges is not None:
            l_edges = np.asarray(let_edges, dtype=float)
        elif bin_width_let is None:
            l_edges = _auto_bins(arr=self.let)
        else:
            lmax = float(np.max(self.let))
            nl = int(np.ceil(lmax / bin_width_let))
            nl = max(nl, 1)
            l_edges = np.linspace(0.0, nl * bin_width_let, nl + 1)

        # --- 2D histogram (differential absolute volumes) ---
        vol_weights = self.relw * self.volume_cc
        vols, d_edges, l_edges = np.histogram2d(self.dose, self.let,
                                                bins=(d_edges, l_edges),
                                                weights=vol_weights)

        if cumulative:
            values = _suffix_cumsum2d(vols)
        else:
            values = vols.astype(float)

        if normalize:
            values = (values / self.volume_cc) * 100.0

        return Histogram2D(values=values,
                           dose_edges=d_edges,
                           let_edges=l_edges,
                           normalize=normalize,
                           cumulative=cumulative,
                           dose_label=f"Dose [{self.dose_units}]",
                           let_label=f"LET [{self.let_units}]")
