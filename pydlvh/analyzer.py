from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Literal, Optional, Union
from scipy.stats import mannwhitneyu

from pydlvh.core import Histogram1D, Histogram2D


class Analyzer:
    """ 
        Class to analyze control and adverse event (ae) groups and investigate possible statistically significant differences.
        The class assumes that 
    """

    def __init__(self, *, 
                 control_group: Union[Histogram1D, Histogram2D], 
                 ae_group: Union[Histogram1D, Histogram2D],
                 quantity: Optional[str] = None):
        

        if type(control_group) != type(ae_group):
            raise ValueError("Both histograms must be of the same type (1D or 2D).")

        if isinstance(control_group, Histogram2D):
            if (not np.array_equal(control_group.dose_edges, ae_group.dose_edges) or
                not np.array_equal(control_group.let_edges, ae_group.let_edges)):
                raise ValueError("Control and ae histograms must have matching dose and LET edges")

        if isinstance(control_group, Histogram1D):
            if not np.array_equal(control_group.edges, ae_group.edges):
                raise ValueError("Control and ae histograms must have matching edges.")
            
        if isinstance(control_group, Histogram1D) and (quantity is None or quantity.lower() not in ["dose", "let"]):
            raise AttributeError("Histogram1Ds were provided, but not the specified quantity (dose or let)")
        
        if control_group.stat != "median" or ae_group.stat != "median":
            raise AttributeError("Control and ae histograms must have been computed as median quantities.")
        
        self.control_group = control_group
        self.ae_group = ae_group
        if isinstance(control_group, Histogram1D):
            if quantity == "dose":
                self.dose_edges = control_group.edges
                self.let_edges = None
            elif quantity == "let":
                self.let_edges = control_group.edges
                self.dose_edges = None
        else:
            self.dose_edges = control_group.dose_edges
            self.let_edges = control_group.let_edges


    def perform_voxel_wise_Mann_Whitney_test(self, *, 
                                             significance_level: float = 0.05,
                                             useBonferronicorrection: bool = False) -> np.ndarray:

        """ Perform voxel-wise Mann-Whitney U test between control and ae groups. """

        p_values = []

        if isinstance(self.control_group, Histogram1D):

            for control, ae in zip(self.control_group.values, self.ae_group.values):
                _, p = mannwhitneyu(control, ae, alternative="two-sided")
                p_values.append(p) # TODO: Sputare fuori un array con p-values (1D o 2D a seconda del tipo di histo)
                # TODO: sputa fuori anche un array di bool per valutare dove i p-values passano o meno il livello di significanza di input

            return p_values
        
        else:
            return 1


    def get_data(self, *, x: Literal["edges", "centers"] = "edges") -> Tuple[np.ndarray, np.ndarray]:
        """Return x-axis coordinates and histogram values."""
        edges = self.edges.copy()
        values = self.values.copy()

        if x == "centers":
            return 0.5 * (edges[:-1] + edges[1:]), values

        if self.cumulative and edges[0] > 0:
            edges = np.insert(edges, 0, 0.0)
            values = np.insert(values, 0, values[0])

        return edges, values

    def plot(self, *, ax: Optional[plt.Axes] = None,
             show_band: bool = True, **kwargs):
        """
        Plot the histogram (cumulative curve or differential bar chart).
    
        Parameters
        ----------
        show_band : bool, default=True
            If True, draw shaded uncertainty/percentile bands if available.
        """
        edges, values = self.get_data(x="edges")
        if ax is None:
            _, ax = plt.subplots()
    
        # --- linea principale ---
        if self.cumulative:
            ax.step(edges[:-1], values, where="post", **kwargs)
            x_band = edges[:-1]  # garantisce stessa len di 'values' anche se c'è padding
            step_kw = "post"
        else:
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            ax.bar(centers, values, width=widths, align="center", **kwargs)
            x_band = centers
            step_kw = None
    
        # --- bande di incertezza/percentili ---
        if show_band:
            n = len(values)
    
            def _fix_len(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if arr is None:
                    return None
                arr = np.asarray(arr, dtype=float)
                if arr.shape[0] == n:
                    return arr
                # caso tipico: cumulativo con padding ha values di len n, ma bande di len n-1
                if self.cumulative and arr.shape[0] == n - 1:
                    return np.insert(arr, 0, arr[0])
                # altrimenti non plottiamo la banda per evitare errori
                return None
    
            # std band
            if self.err is not None:
                err = _fix_len(self.err)
                if err is not None:
                    y_lo = values - err
                    y_hi = values + err
                    ax.fill_between(x_band, y_lo, y_hi,
                                    step=step_kw, alpha=0.3, color="gray", label="±std")
    
            # percentile band
            if self.p_lo is not None and self.p_hi is not None:
                plo = _fix_len(self.p_lo)
                phi = _fix_len(self.p_hi)
                if plo is not None and phi is not None:
                    ax.fill_between(x_band, plo, phi,
                                    step=step_kw, alpha=0.2, color="orange", label="IQR")
    
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
    
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        return ax


class Histogram2D:
    """Internal 2D histogram container with plotting capability."""

    def __init__(self, *, values: np.ndarray,
                 dose_edges: np.ndarray, let_edges: np.ndarray,
                 normalize: bool, cumulative: bool,
                 dose_label: str = "Dose [Gy]",
                 let_label: str = "LET [keV/µm]",
                 err: Optional[np.ndarray] = None,
                 p_lo: Optional[np.ndarray] = None,
                 p_hi: Optional[np.ndarray] = None):
        self.values = np.asarray(values, dtype=float)
        self.dose_edges = np.asarray(dose_edges, dtype=float)
        self.let_edges = np.asarray(let_edges, dtype=float)
        self.normalize = bool(normalize)
        self.cumulative = bool(cumulative)
        self.dose_label = dose_label
        self.let_label = let_label

        # Optional cohort statistics
        self.err = None if err is None else np.asarray(err, dtype=float)
        self.p_lo = None if p_lo is None else np.asarray(p_lo, dtype=float)
        self.p_hi = None if p_hi is None else np.asarray(p_hi, dtype=float)
   
    def plot(self, *, ax: Optional[plt.Axes] = None,
             cmap: str = "viridis", colorbar: bool = True,
             mode: Literal["values", "err", "p_lo", "p_hi"] = "values",
             isovolumes: Optional[List[float]] = None,
             interactive: bool = False,
             **kwargs):
        """
        Plot the 2D histogram or auxiliary maps, with optional (interactive) isovolumes.
        Matching the viewer.plot2D layout and slider behavior.

        Parameters
        ----------
        mode : {"values", "err", "p_lo", "p_hi"}, default="values"
            Which map to display: central values, std, or percentile bands.
        isovolumes : list of float, optional
            Contour levels in percent [%] of total volume.
            - If self.normalize=True: values are already %.
            - If self.normalize=False: values are interpreted as % and converted to cm³.
        interactive : bool, default=False
            If True, show an interactive slider to add a single isovolume contour.
        """

        # --- Ensure visual padding to 0 for cumulative maps ---
        if self.cumulative:
            if self.dose_edges[0] > 0:
                self.dose_edges = np.insert(self.dose_edges, 0, 0.0)
                self.values = np.insert(self.values, 0, self.values[0, :], axis=0)
                if self.err is not None:
                    self.err = np.insert(self.err, 0, self.err[0, :], axis=0)
                if self.p_lo is not None and self.p_hi is not None:
                    self.p_lo = np.insert(self.p_lo, 0, self.p_lo[0, :], axis=0)
                    self.p_hi = np.insert(self.p_hi, 0, self.p_hi[0, :], axis=0)
            if self.let_edges[0] > 0:
                self.let_edges = np.insert(self.let_edges, 0, 0.0)
                self.values = np.insert(self.values, 0, self.values[:, 0], axis=1)
                if self.err is not None:
                    self.err = np.insert(self.err, 0, self.err[:, 0], axis=1)
                if self.p_lo is not None and self.p_hi is not None:
                    self.p_lo = np.insert(self.p_lo, 0, self.p_lo[:, 0], axis=1)
                    self.p_hi = np.insert(self.p_hi, 0, self.p_hi[:, 0], axis=1)

        # --- Select data to plot ---
        if mode == "values":
            data = self.values.T
        elif mode == "err" and self.err is not None:
            data = self.err.T
        elif mode == "p_lo" and self.p_lo is not None:
            data = self.p_lo.T
        elif mode == "p_hi" and self.p_hi is not None:
            data = self.p_hi.T
        else:
            raise ValueError(f"No data available for mode='{mode}'.")

        # --- Setup figure/axes with viewer-style spacing ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=False)
        else:
            fig = ax.figure
        fig.subplots_adjust(bottom=0.6 if interactive else 0.12)

        # --- Heatmap ---
        mesh = ax.pcolormesh(self.dose_edges,
                             self.let_edges,
                             data,
                             cmap=cmap, **kwargs)
        ax.set_xlabel(self.dose_label)
        ax.set_ylabel(self.let_label)
        ax.set_title("Cumulative Dose–LET Volume Histogram (DLVH)")

        if self.cumulative:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        if colorbar:
            plt.colorbar(mesh, ax=ax,
                         label="Volume [%]" if self.normalize else "Volume [cm³]")

        # --- Helper: estimate total volume (needed if normalize=False) ---
        def _total_volume(arr: np.ndarray) -> float:
            if self.cumulative:
                return float(arr[0, 0]) if arr.size > 0 else 0.0
            return float(np.nansum(arr))

        total_abs = _total_volume(self.values.T if mode == "values" else data)

        # --- Draw static isovolumes ---
        if isovolumes:
            if self.normalize:
                levels_abs = list(isovolumes)  # already in %
            else:
                levels_abs = [p / 100.0 * total_abs for p in isovolumes]
            CS = ax.contour(self.dose_edges[:-1],
                            self.let_edges[:-1],
                            data,
                            levels=levels_abs,
                            colors="white",
                            linewidths=1.2)
            fmt = (lambda v: f"{v:g}%") if self.normalize else (lambda v: f"{v:.2f} cm³")
            ax.clabel(CS, inline=True, fontsize=8, fmt=fmt)

        # --- Interactive slider ---
        if interactive:
            # keep references on self to avoid GC
            self._fig2d = fig
            self._ax2d = ax
            self._data2d = data
            self._total_abs = total_abs
            self._interactive_contour = None
            self._interactive_labels = []

            ax_slider = plt.axes([0.15, 0.00001, 0.7, 0.04])
            self._slider = Slider(ax_slider,
                                  "Isovol [%]",
                                  valmin=0,
                                  valmax=100,
                                  valinit=0,
                                  valstep=1)

            def _update(val):
                # clear previous contour
                if self._interactive_contour is not None:
                    for coll in self._interactive_contour.collections:
                        coll.remove()
                    self._interactive_contour = None
                for lbl in self._interactive_labels:
                    try:
                        lbl.remove()
                    except Exception:
                        pass
                self._interactive_labels = []

                level_pct = float(self._slider.val)
                if level_pct <= 0 or level_pct >= 100:
                    self._fig2d.canvas.draw_idle()
                    return

                if self.normalize:
                    level_abs = level_pct
                else:
                    level_abs = level_pct / 100.0 * self._total_abs

                self._interactive_contour = self._ax2d.contour(
                    self.dose_edges[:-1],
                    self.let_edges[:-1],
                    self._data2d,
                    levels=[level_abs],
                    colors="red",
                    linewidths=1.5
                )
                self._interactive_labels = self._ax2d.clabel(
                    self._interactive_contour,
                    inline=True,
                    fontsize=8,
                    fmt=lambda _: f"{level_pct:.0f}%"
                )
                self._fig2d.canvas.draw_idle()

            self._slider.on_changed(_update)

        plt.show()
        return ax


    def get_marginals(self, *, kind: Literal["dose", "let"] = "dose") -> Tuple[np.ndarray, np.ndarray]:
        """Return the marginal histogram as (edges, values). Only for cumulative 2D."""
        if not self.cumulative:
            raise NotImplementedError("Marginal extraction is only implemented for cumulative 2D histograms.")

        if kind == "dose":
            edges = self.dose_edges.copy()
            values = self.values[:, 0].copy()
        elif kind == "let":
            edges = self.let_edges.copy()
            values = self.values[0, :].copy()
        else:
            raise ValueError("Argument 'kind' must be either 'dose' or 'let'.")

        if edges[0] > 0:
            edges = np.insert(edges, 0, 0.0)
            values = np.insert(values, 0, values[0])

        return edges, values

    def plot_marginals(self, *, kind: Literal["dose", "let"] = "dose"):
        """Plot DVH or LVH derived from the cumulative 2D histogram."""
        edges, values = self.get_marginals(kind=kind)
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

    You may provide only `dose` (for DVH) or only `let` (for LVH),
    or both (to compute DVH, LVH and 2D DLVH).
    """

    def __init__(self, *,
                 dose: Optional[np.ndarray] = None,
                 let: Optional[np.ndarray] = None,
                 volume_cc: float,
                 relative_volumes: Optional[np.ndarray] = None,
                 dose_units: str = "Gy",
                 let_units: str = "keV/µm"):
        if dose is None and let is None:
            raise ValueError("At least one of dose or let must be provided.")

        self.dose = self._validate_array(dose, "dose") if dose is not None else None
        self.let = self._validate_array(let, "let") if let is not None else None

        if self.dose is not None and self.let is not None:
            if self.dose.shape != self.let.shape:
                raise ValueError("Dose and LET arrays must have the same shape.")

        if volume_cc <= 0 or not np.isfinite(volume_cc):
            raise ValueError("Volume must be a positive finite value (cm³).")
        self.volume_cc = float(volume_cc)

        size = self.dose.size if self.dose is not None else self.let.size
        if relative_volumes is None:
            relw = np.full(size, 1.0 / size, dtype=float)
        else:
            relw = np.asarray(relative_volumes, dtype=float).ravel()
            if relw.shape[0] != size:
                raise ValueError("relative_volumes must match dose/let length.")
            if np.any(~np.isfinite(relw)) or np.any(relw < 0):
                raise ValueError("relative_volumes must be finite and non-negative.")
            sumw = float(relw.sum())
            if sumw <= 0:
                raise ValueError("Sum of relative_volumes must be > 0.")
            relw = relw / sumw
        self.relw = relw

        self.n_voxels = size
        self.dose_units = dose_units
        self.let_units = let_units

    @staticmethod
    def _validate_array(arr: Optional[np.ndarray], label: str) -> np.ndarray:
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            raise ValueError(f"{label} array cannot be empty.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{label} array must contain non-negative finite values.")
        return arr

    def _volume_histogram(self, *, data: np.ndarray, weights: np.ndarray,
                          quantity: str,
                          bin_width: Optional[float] = None,
                          bin_edges: Optional[np.ndarray] = None,
                          normalize: bool = True,
                          cumulative: bool = True) -> Histogram1D:
        if bin_edges is not None:
            edges = np.asarray(bin_edges, dtype=float)
        elif bin_width is None:
            edges = _auto_bins(arr=data)
        else:
            xmax = float(np.max(data))
            n_bins = int(np.ceil(xmax / bin_width)) if bin_width > 0 else 1
            edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)

        vols, _ = np.histogram(data, bins=edges, weights=weights)
        if cumulative:
            vols = np.cumsum(vols[::-1])[::-1]

        values = vols.astype(float)
        if normalize:
            values = (values / self.volume_cc) * 100.0

        xlab = f"Dose [{self.dose_units}]" if quantity == "dose" else f"LET [{self.let_units}]"
        return Histogram1D(values=values, edges=edges,
                           quantity=quantity, normalize=normalize,
                           cumulative=cumulative, x_label=xlab)

    def dose_volume_histogram(self, *, bin_width: Optional[float] = None,
                              bin_edges: Optional[np.ndarray] = None,
                              normalize: bool = True,
                              cumulative: bool = True,
                              let_threshold: float = 0.0) -> Histogram1D:
        if self.dose is None:
            raise RuntimeError("Dose array not available for DVH.")
        if let_threshold > 0 and self.let is None:
            raise RuntimeError("LET array required to apply let_threshold.")

        mask = np.ones_like(self.dose, dtype=bool) if self.let is None else self.let >= let_threshold
        data = self.dose[mask]
        weights = (self.relw * self.volume_cc)[mask]

        return self._volume_histogram(data=data, weights=weights, quantity="dose",
                                      bin_width=bin_width, bin_edges=bin_edges,
                                      normalize=normalize, cumulative=cumulative)

    def let_volume_histogram(self, *, bin_width: Optional[float] = None,
                             bin_edges: Optional[np.ndarray] = None,
                             normalize: bool = True,
                             cumulative: bool = True,
                             dose_threshold: float = 0.0) -> Histogram1D:
        if self.let is None:
            raise RuntimeError("LET array not available for LVH.")
        if dose_threshold > 0 and self.dose is None:
            raise RuntimeError("Dose array required to apply dose_threshold.")

        mask = np.ones_like(self.let, dtype=bool) if self.dose is None else self.dose >= dose_threshold
        data = self.let[mask]
        weights = (self.relw * self.volume_cc)[mask]

        return self._volume_histogram(data=data, weights=weights, quantity="let",
                                      bin_width=bin_width, bin_edges=bin_edges,
                                      normalize=normalize, cumulative=cumulative)

    def dose_let_volume_histogram(self, *,
                                  bin_width_dose: Optional[float] = None,
                                  bin_width_let: Optional[float] = None,
                                  dose_edges: Optional[np.ndarray] = None,
                                  let_edges: Optional[np.ndarray] = None,
                                  normalize: bool = True,
                                  cumulative: bool = True) -> Histogram2D:
        if self.dose is None or self.let is None:
            raise RuntimeError("Both dose and LET arrays are required for 2D DLVH.")

        # dose edges
        if dose_edges is not None:
            d_edges = np.asarray(dose_edges, dtype=float)
        elif bin_width_dose is None:
            d_edges = _auto_bins(arr=self.dose)
        else:
            dmax = float(np.max(self.dose))
            nd = int(np.ceil(dmax / bin_width_dose))
            d_edges = np.linspace(0.0, nd * bin_width_dose, nd + 1)

        # let edges
        if let_edges is not None:
            l_edges = np.asarray(let_edges, dtype=float)
        elif bin_width_let is None:
            l_edges = _auto_bins(arr=self.let)
        else:
            lmax = float(np.max(self.let))
            nl = int(np.ceil(lmax / bin_width_let))
            l_edges = np.linspace(0.0, nl * bin_width_let, nl + 1)

        weights = self.relw * self.volume_cc
        vols, d_edges, l_edges = np.histogram2d(self.dose, self.let,
                                                bins=(d_edges, l_edges),
                                                weights=weights)

        values = _suffix_cumsum2d(vols) if cumulative else vols.astype(float)
        if normalize:
            values = (values / self.volume_cc) * 100.0

        return Histogram2D(values=values,
                           dose_edges=d_edges,
                           let_edges=l_edges,
                           normalize=normalize,
                           cumulative=cumulative,
                           dose_label=f"Dose [{self.dose_units}]",
                           let_label=f"LET [{self.let_units}]")
