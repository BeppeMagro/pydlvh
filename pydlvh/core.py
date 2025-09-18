import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Literal
from .utils import _auto_bins


class Histogram1D:
    """
    Internal 1D histogram container with plotting capability.
    Not part of the public API.
    """

    def __init__(self, *, values: np.ndarray, edges: np.ndarray,
                 quantity: str, normalize: bool, cumulative: bool):
        self.values = np.asarray(values, dtype=float)
        self.edges = np.asarray(edges, dtype=float)
        self.quantity = str(quantity)
        self.normalize = bool(normalize)
        self.cumulative = bool(cumulative)

    def get_data(self, *, x: Literal["edges", "centers"] = "edges") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return x-axis coordinates and histogram values.

        Parameters
        ----------
        x : {"edges", "centers"}, default="edges"
            - "edges": return (edges, values). If cumulative, auto-pad to 0 for
              visualization consistency (prepend 0 to edges and duplicate first value).
            - "centers": return (centers, values). No padding applied.

        Returns
        -------
        xcoords : np.ndarray
            Bin edges (len = N+1) or centers (len = N), depending on `x`.
        values : np.ndarray
            Histogram values; length N (centers) or N(+1 if padded for cumulative edges).
        """
        # x == "edges"
        edges = self.edges.copy()
        values = self.values.copy()
        
        if x == "centers":
            return 0.5 * (edges[:-1] + edges[1:]), values

        if self.cumulative and edges[0] > 0:
            # Auto pad to zero for cumulative visualization
            edges = np.insert(edges, 0, 0.0)
            values = np.insert(values, 0, values[0])

        return edges, values

    def plot(self, *, ax: plt.Axes = None, **kwargs):
        """
        Plot the histogram (cumulative curve or differential bar chart).
        """
        # For plotting we always use edges (and let get_data handle padding)
        edges, values = self.get_data(x="edges")

        if ax is None:
            _, ax = plt.subplots()

        if self.cumulative:
            # Step plot using edges directly
            ax.step(edges[:-1], values, where="post", **kwargs)
            # Force axes start at 0
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        else:
            # Differential → bar chart at bin centers
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            ax.bar(centers, values, width=widths, align="center", **kwargs)
            ax.set_ylim(bottom=0)

        # --- Labels ---
        if self.quantity == "dose":
            ax.set_xlabel("Dose [Gy]")
        elif self.quantity == "let":
            ax.set_xlabel("LET [keV/µm]")

        if self.normalize:
            ax.set_ylabel("Volume [%]" if self.cumulative else "Relative volume")
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
                 normalize: bool, cumulative: bool):
        self.values = np.asarray(values, dtype=float)
        self.dose_edges = np.asarray(dose_edges, dtype=float)
        self.let_edges = np.asarray(let_edges, dtype=float)
        self.normalize = bool(normalize)
        self.cumulative = bool(cumulative)

    def plot(self, *, ax: plt.Axes = None,
             cmap: str = "viridis", colorbar: bool = True, **kwargs):
        """
        Plot the 2D histogram as a heatmap.
        """
        if ax is None:
            _, ax = plt.subplots()

        mesh = ax.pcolormesh(self.dose_edges,
                             self.let_edges,
                             self.values.T,
                             cmap=cmap, **kwargs)
        ax.set_xlabel("Dose [Gy]")
        ax.set_ylabel("LET [keV/µm]")

        # Auto pad to zero for visualization if cumulative
        if self.cumulative:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        if colorbar:
            plt.colorbar(mesh, ax=ax,
                         label="Volume [%]" if self.normalize else "Volume [cm³]")
        return ax

    def get_marginals(self, *, kind: Literal["dose", "let"] = "dose") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the marginal histogram as (edges, values). Only for cumulative 2D.

        Parameters
        ----------
        kind : {"dose", "let"}, default="dose"
            - "dose": DVH marginal → take first column (LET ≥ 0).
            - "let" : LVH marginal → take first row (Dose ≥ 0).

        Returns
        -------
        edges : np.ndarray
            Bin edges of the selected axis.
        values : np.ndarray
            Marginal values aligned to edges (auto-padded to 0).

        Raises
        ------
        NotImplementedError
            If called on a differential 2D histogram (cumulative == False).
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

        # Auto pad to zero for cumulative visualization
        if edges[0] > 0:
            edges = np.insert(edges, 0, 0.0)
            values = np.insert(values, 0, values[0])

        return edges, values

    def plot_marginals(self, *, kind: Literal["dose", "let"] = "dose"):
        """
        Plot DVH or LVH derived from the cumulative 2D histogram.

        Parameters
        ----------
        kind : {"dose", "let"}, default="dose"
            Which marginal to plot:
            - "dose": DVH (LET ≥ 0).
            - "let" : LVH (Dose ≥ 0).

        Returns
        -------
        ax : matplotlib.axes.Axes
            Matplotlib axis with the marginal plot.

        Notes
        -----
        - Works only if the 2D histogram is cumulative, since the
          definition is V(D ≥ d, LET ≥ l).
        - Plotting follows the same rules as Histogram1D cumulative plots
          (auto pad to zero, step plot, axes forced to start at 0).
        """
        edges, values = self.get_marginals(kind=kind)

        fig, ax = plt.subplots(figsize=(6, 5))

        if kind == "dose":
            ax.step(edges[:-1], values, where="post", color="C0", label="DVH")
            ax.set_xlabel("Dose [Gy]")
            ax.set_title("DVH from 2D cumulative")
        else:  # "let"
            ax.step(edges[:-1], values, where="post", color="C1", label="LVH")
            ax.set_xlabel("LET [keV/µm]")
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
    Dose-LET Volume Histogram (DLVH).

    Stores paired dose and LET values for a specific ROI, already masked.
    The input arrays must have strict 1:1 correspondence, and the total
    physical volume of the ROI must be provided.

    Parameters
    ----------
    dose : numpy.ndarray
        Flattened array of dose values (Gy). Must be non-negative.
    let : numpy.ndarray
        Flattened array of LET values (keV/µm). Must be non-negative.
    volume_cc : float
        Total physical volume of the ROI in cm³. Must be > 0.

    Notes
    -----
    - Arrays are assumed to be already masked to the ROI of interest.
    - The i-th dose value is associated with the i-th LET value.
    """

    def __init__(self, *, dose: np.ndarray, let: np.ndarray, volume_cc: float):
        self.dose = self._validate_array(dose, "dose")
        self.let = self._validate_array(let, "let")

        if self.dose.shape != self.let.shape:
            raise ValueError("Dose and LET arrays must have the same shape.")

        if volume_cc <= 0:
            raise ValueError("Volume must be a positive value (cm³).")
        self.volume_cc = float(volume_cc)

        self.n_voxels = int(self.dose.size)

    @staticmethod
    def _validate_array(arr: np.ndarray, label: str) -> np.ndarray:
        """Ensure array is numpy.ndarray, flattened, non-empty, and non-negative."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{label} must be a numpy.ndarray (got {type(arr)}).")
        arr = arr.ravel()
        if arr.size == 0:
            raise ValueError(f"{label} array cannot be empty.")
        if np.any(arr < 0):
            raise ValueError(f"{label} array must contain only non-negative values.")
        return arr

    # ---------- internal builder ----------
    def _volume_histogram(self, *, data: np.ndarray, quantity: str,
                          bin_width: float = None,
                          normalize: bool = True,
                          cumulative: bool = True) -> Histogram1D:
        # Binning
        if bin_width is None:
            bin_edges, _ = _auto_bins(arr=data)
        else:
            xmax = float(np.max(data))
            n_bins = int(np.ceil(xmax / bin_width)) if bin_width > 0 else 1
            n_bins = max(n_bins, 1)
            bin_edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)

        counts, _ = np.histogram(data, bins=bin_edges)

        if cumulative:
            # right-cumulative counts: V(X ≥ x)
            counts = np.cumsum(counts[::-1])[::-1]

        # counts → volume [cm³]
        hist = counts.astype(float) * (self.volume_cc / self.n_voxels)

        if normalize:
            if cumulative:
                # Normalize to ROI volume → [%]
                hist = (hist / self.volume_cc) * 100.0
            else:
                # Differential: relative so that sum = 1
                total = hist.sum()
                hist = hist / total if total > 0 else hist

        return Histogram1D(values=hist, edges=bin_edges,
                           quantity=quantity,
                           normalize=normalize, cumulative=cumulative)

    # ---------- public API ----------
    def dose_volume_histogram(self, *, bin_width: float = None,
                              normalize: bool = True,
                              cumulative: bool = True,
                              let_threshold: float = 0.0) -> Histogram1D:
        """
        Compute and return a DVH as Histogram1D object.

        Parameters
        ----------
        bin_width : float, optional
            Bin resolution (Gy). If None, auto-selected.
        normalize : bool, default=True
            Normalize to % (cumulative) or relative (differential).
        cumulative : bool, default=True
            Return cumulative (V(D≥d)) or differential.
        let_threshold : float, default=0.0
            Consider only voxels with LET >= threshold.
        """
        mask = self.let >= let_threshold
        data = self.dose[mask]
        return self._volume_histogram(data=data, quantity="dose",
                                      bin_width=bin_width,
                                      normalize=normalize,
                                      cumulative=cumulative)

    def let_volume_histogram(self, *, bin_width: float = None,
                             normalize: bool = True,
                             cumulative: bool = True,
                             dose_threshold: float = 0.0) -> Histogram1D:
        """
        Compute and return a LVH as Histogram1D object.

        Parameters
        ----------
        bin_width : float, optional
            Bin resolution (keV/µm). If None, auto-selected.
        normalize : bool, default=True
            Normalize to % (cumulative) or relative (differential).
        cumulative : bool, default=True
            Return cumulative (V(LET≥l)) or differential.
        dose_threshold : float, default=0.0
            Consider only voxels with dose >= threshold.
        """
        mask = self.dose >= dose_threshold
        data = self.let[mask]
        return self._volume_histogram(data=data, quantity="let",
                                      bin_width=bin_width,
                                      normalize=normalize,
                                      cumulative=cumulative)

    def dose_let_volume_histogram(self, *,
                                  bin_width_dose: float = None,
                                  bin_width_let: float = None,
                                  normalize: bool = True,
                                  cumulative: bool = True) -> Histogram2D:
        """
        Compute and return a Dose–LET Volume Histogram (2D).

        Parameters
        ----------
        bin_width_dose : float, optional
            Bin resolution for dose (Gy). If None, auto-selected.
        bin_width_let : float, optional
            Bin resolution for LET (keV/µm). If None, auto-selected.
        normalize : bool, default=True
            - If cumulative=True: normalize to ROI volume (max = 100%).
            - If cumulative=False: normalize relative so that sum = 1.
        cumulative : bool, default=True
            If True, compute cumulative DLVH:
            V(d,l) = volume of voxels with Dose ≥ d and LET ≥ l.
        """
        # --- dose binning ---
        if bin_width_dose is None:
            dose_edges, _ = _auto_bins(arr=self.dose)
        else:
            dmax = float(np.max(self.dose))
            nd = int(np.ceil(dmax / bin_width_dose))
            nd = max(nd, 1)
            dose_edges = np.linspace(0.0, nd * bin_width_dose, nd + 1)

        # --- let binning ---
        if bin_width_let is None:
            let_edges, _ = _auto_bins(arr=self.let)
        else:
            lmax = float(np.max(self.let))
            nl = int(np.ceil(lmax / bin_width_let))
            nl = max(nl, 1)
            let_edges = np.linspace(0.0, nl * bin_width_let, nl + 1)

        # --- 2D histogram (differential counts) ---
        counts, dose_edges, let_edges = np.histogram2d(
            self.dose, self.let, bins=(dose_edges, let_edges)
        )
        voxel_vol = self.volume_cc / self.n_voxels

        if cumulative:
            # Build V(D ≥ d, LET ≥ l) on the same grid
            values = np.zeros_like(counts, dtype=float)
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    mask = (self.dose >= dose_edges[i]) & (self.let >= let_edges[j])
                    values[i, j] = np.sum(mask) * voxel_vol
        else:
            values = counts.astype(float) * voxel_vol

        # --- normalization ---
        if normalize:
            if cumulative:
                values = (values / self.volume_cc) * 100.0
            else:
                total = values.sum()
                if total > 0:
                    values = values / total  # sum = 1

        return Histogram2D(values=values,
                           dose_edges=dose_edges,
                           let_edges=let_edges,
                           normalize=normalize,
                           cumulative=cumulative)
