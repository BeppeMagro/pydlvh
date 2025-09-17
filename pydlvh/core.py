import numpy as np
import matplotlib.pyplot as plt
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

    @property
    def centers(self) -> np.ndarray:
        """Bin centers."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def plot(self, *, ax: plt.Axes = None, **kwargs):
        """
        Plot the histogram (cumulative curve or differential bar chart).
        """
        edges = self.edges
        values = self.values

        if ax is None:
            _, ax = plt.subplots()

        if self.cumulative:
            # --- auto pad to zero ---
            if edges[0] > 0:
                edges = np.insert(edges, 0, 0.0)
                values = np.insert(values, 0, values[0])

            # For cumulative: plot step using edges directly
            ax.step(edges[:-1], values, where="post", **kwargs)

            # Force axes start at 0
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        else:
            # Differential → bar chart
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            ax.bar(centers, values, width=widths, align="center", **kwargs)

            # Force y axis start at 0
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

        # --- auto pad to zero for visualization if cumulative ---
        if self.cumulative:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        if colorbar:
            plt.colorbar(mesh, ax=ax,
                         label="Volume [%]" if self.normalize else "Volume [cm³]")
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
        # binning
        if bin_width is None:
            bin_edges, _ = _auto_bins(arr1=data)
        else:
            xmax = float(np.max(data))
            n_bins = int(np.ceil(xmax / bin_width)) if bin_width > 0 else 1
            n_bins = max(n_bins, 1)
            bin_edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)

        counts, _ = np.histogram(data, bins=bin_edges)

        if cumulative:
            counts = np.cumsum(counts[::-1])[::-1]

        # counts → volume [cm³]
        hist = counts.astype(float) * (self.volume_cc / self.n_voxels)

        if normalize:
            if cumulative:
                hist = (hist / self.volume_cc) * 100.0
            else:
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
        Compute a 2D Dose-LET Volume Histogram (DLVH).

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

        Returns
        -------
        Histogram2D
        """
        # --- binning ---
        if bin_width_dose is None:
            dose_edges, _ = _auto_bins(arr1=self.dose)
        else:
            dmax = float(np.max(self.dose))
            nd = int(np.ceil(dmax / bin_width_dose))
            nd = max(nd, 1)
            dose_edges = np.linspace(0.0, nd * bin_width_dose, nd + 1)

        if bin_width_let is None:
            let_edges, _ = _auto_bins(arr1=self.let)
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



