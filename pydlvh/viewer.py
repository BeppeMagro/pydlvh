from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Optional, List
from .core import DLVH


class DLVHViewer:
    """
    Viewer for DVH, LVH, and 2D DLVH.

    - Use plot1D() for interactive DVH/LVH viewer with slider.
    - Use plot2D() for Dose–LET Volume Histogram with optional isovolume lines.
    """

    def __init__(self, dlvh: DLVH):
        self.dlvh = dlvh
        self.fig: Optional[plt.Figure] = None
        self.ax_dvh: Optional[plt.Axes] = None
        self.ax_lvh: Optional[plt.Axes] = None
        self.slider: Optional[Slider] = None
        self.vline: Optional[plt.Line2D] = None
        self.cross: Optional[plt.Line2D] = None
        self.annotation: Optional[plt.Text] = None
        self.lvh_edges: Optional[np.ndarray] = None
        self.lvh_values: Optional[np.ndarray] = None

        # for 2D interactive state
        self._interactive_contour = None
        self._interactive_label_texts: List[plt.Text] = []
        self._h2d_cache = None
        self._ax2d: Optional[plt.Axes] = None
        self._fig2d: Optional[plt.Figure] = None
        self._normalize2d: Optional[bool] = None
        self._slider2d: Optional[Slider] = None

    # =========================================================
    # 1D interactive DVH/LVH
    # =========================================================
    def plot1D(self) -> None:
        """Interactive viewer with DVH (cumulative) and LVH (cumulative, absolute)."""
        self.fig, (self.ax_dvh, self.ax_lvh) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.25, wspace=0.3)

        #  DVH cumulative [%] 
        dvh = self.dlvh.dose_volume_histogram(cumulative=True, normalize=True)
        edges, values = dvh.edges, dvh.values
        if edges[0] > 0:
            edges = np.insert(edges, 0, 0.0)
            values = np.insert(values, 0, values[0])

        (self.dvh_line,) = self.ax_dvh.step(
            edges[:-1], values, where="post", color="C0", label="DVH\n(cumulative)")
        self.ax_dvh.set_title("DVH")
        self.ax_dvh.set_xlabel(f"Dose [{self.dlvh.dose_units}]")
        self.ax_dvh.set_ylabel("Volume [%]")
        self.ax_dvh.set_xlim(left=0)
        self.ax_dvh.set_ylim(bottom=0)
        self.ax_dvh.grid(True, alpha=0.2)

        # Vertical threshold line on DVH
        self.vline = self.ax_dvh.axvline(0, color="red", linestyle="--", lw=1.5)

        # DVH differential overlay [%] 
        dvh_diff = self.dlvh.dose_volume_histogram(cumulative=False, normalize=True)
        edges_d, values_d = dvh_diff.edges, dvh_diff.values
        centers = 0.5 * (edges_d[:-1] + edges_d[1:])
        widths = np.diff(edges_d)

        ax_dvh_diff = self.ax_dvh.twinx()
        ax_dvh_diff.bar(
            centers, values_d,
            width=widths,
            color="C0", alpha=0.3,
            align="center", label="DVH\n(differential)")
        ax_dvh_diff.set_ylabel("Differential volume [%]")
        ax_dvh_diff.set_ylim(bottom=0)
        self.fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=self.ax_dvh.transAxes, frameon=False)

        # LVH cumulative (absolute cm³) 
        lvh = self.dlvh.let_volume_histogram(cumulative=True, normalize=False)
        self.lvh_edges, self.lvh_values = lvh.edges, lvh.values
        if self.lvh_edges[0] > 0:
            self.lvh_edges = np.insert(self.lvh_edges, 0, 0.0)
            self.lvh_values = np.insert(self.lvh_values, 0, self.lvh_values[0])

        (self.lvh_line,) = self.ax_lvh.step(
            self.lvh_edges[:-1], self.lvh_values, where="post",
            color="C1", label="LVH")
        self.ax_lvh.set_title("LVH")
        self.ax_lvh.set_xlabel(fr"LET$_d$ [{self.dlvh.let_units}]")
        self.ax_lvh.set_ylabel("Volume [cm³]")
        self.ax_lvh.set_xlim(left=0)
        self.ax_lvh.set_ylim(bottom=0)
        self.ax_lvh.grid(True, alpha=0.2)
        self.ax_lvh.legend(frameon=False)

        # Cross marker + text annotation for LVH
        (self.cross,) = self.ax_lvh.plot([], [], "ro", markersize=6)
        self.annotation = self.ax_lvh.text(
            0.5, -0.2, "", transform=self.ax_lvh.transAxes,
            ha="center", va="top", fontsize=9)

        # Slider 
        bin_width = float(np.diff(dvh.edges).mean()) if len(dvh.edges) > 1 else 0.1
        ax_slider = plt.axes([0.12, 0.1, 0.35, 0.05])
        self.slider = Slider(
            ax_slider, f"Dose\nthreshold\n[{self.dlvh.dose_units}]",
            valmin=0.0,
            valmax=float(np.max(self.dlvh.dose)),
            valinit=0.0,
            valstep=bin_width if np.isfinite(bin_width) and bin_width > 0 else 0.1)
        self.slider.on_changed(self._update1D)

        # Connect clicks
        assert self.fig is not None
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        plt.show()

    def _update1D(self, val: float) -> None:
        """Update LVH when slider moves."""
        thr = float(self.slider.val)
        self.vline.set_xdata([thr, thr])

        lvh_thr = self.dlvh.let_volume_histogram(
            cumulative=True, normalize=False, dose_threshold=thr
        )
        self.lvh_edges, self.lvh_values = lvh_thr.edges, lvh_thr.values
        if self.lvh_edges[0] > 0:
            self.lvh_edges = np.insert(self.lvh_edges, 0, 0.0)
            self.lvh_values = np.insert(self.lvh_values, 0, self.lvh_values[0])

        self.lvh_line.set_data(self.lvh_edges[:-1], self.lvh_values)

        self.cross.set_data([], [])
        self.annotation.set_text("")

        self.ax_lvh.relim()
        self.ax_lvh.autoscale_view(scalex=True, scaley=True)
        self.ax_lvh.set_xlim(left=0)
        self.ax_lvh.set_ylim(bottom=0)
        self.fig.canvas.draw_idle() 

    def _on_click(self, event) -> None:
        """Handle clicks on LVH panel."""
        if (event.inaxes != self.ax_lvh or
                self.lvh_edges is None or
                len(self.lvh_edges) < 2):
            return

        centers = 0.5 * (self.lvh_edges[:-1] + self.lvh_edges[1:])
        idx = int(np.argmin(np.abs(centers - event.xdata)))

        let_val = float(centers[idx])
        vol_abs = float(self.lvh_values[idx])
        vol_rel = 100.0 * vol_abs / self.dlvh.volume_cc

        self.cross.set_data([let_val], [vol_abs])
        self.annotation.set_text(                
            f"LET ≈ {let_val:.2f} {self.dlvh.let_units}, "
            f"V = {vol_abs:.2f} cm³ ({vol_rel:.1f} %)"
        )

        self.fig.canvas.draw_idle()

    # =========================================================
    # 2D DLVH
    # =========================================================
    def plot2D(self, *,
               bin_width_dose: Optional[float] = 1.0,
               bin_width_let: Optional[float] = 0.1,
               dose_edges: Optional[np.ndarray] = None,
               let_edges: Optional[np.ndarray] = None,
               normalize: bool = True,
               cumulative: bool = True,
               isovolumes: Optional[List[float]] = None,
               cmap: str = "plasma",
               interactive: bool = False) -> None:
        """Plot the 2D Dose–LET Volume Histogram (DLVH)."""
        h2d = self.dlvh.dose_let_volume_histogram(
            bin_width_dose=bin_width_dose,
            bin_width_let=bin_width_let,
            dose_edges=dose_edges,
            let_edges=let_edges,
            normalize=normalize,
            cumulative=cumulative
        )

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=False)
        fig.subplots_adjust(bottom=0.15 if interactive else 0.12)

        mesh = ax.pcolormesh(h2d.dose_edges,
                             h2d.let_edges,
                             h2d.values.T,
                             cmap=cmap)
        ax.set_xlabel(f"Dose [{self.dlvh.dose_units}]")
        ax.set_ylabel(f"LET [{self.dlvh.let_units}]")
        ax.set_title("2D Dose–LET Volume Histogram")

        if cumulative:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Volume [%]" if normalize else "Volume [cm³]")

        if isovolumes:
            levels = (isovolumes if normalize
                      else [p / 100.0 * self.dlvh.volume_cc for p in isovolumes])
            CS = ax.contour(h2d.dose_edges[:-1],
                            h2d.let_edges[:-1],
                            h2d.values.T,
                            levels=levels,
                            colors="white",
                            linewidths=1.2)
            ax.clabel(CS, inline=True, fontsize=8, fmt="%g")

        if interactive:
            self._interactive_contour = None
            self._interactive_label_texts = []
            self._h2d_cache = h2d
            self._ax2d = ax
            self._fig2d = fig
            self._normalize2d = normalize

            ax_slider = plt.axes([0.15, 0.02, 0.7, 0.04])
            self._slider2d = Slider(ax_slider,
                                    "Isovol [%]",
                                    valmin=0,
                                    valmax=100,
                                    valinit=0,
                                    valstep=1)
            self._slider2d.on_changed(self._update2D)

        plt.show()

    def _update2D(self, val: float) -> None:
        """Update interactive isovolume line in 2D DLVH plot."""
        h2d = self._h2d_cache
        ax = self._ax2d
        fig = self._fig2d
        normalize = self._normalize2d

        if self._interactive_contour is not None:
            for coll in self._interactive_contour.collections:
                coll.remove()
            self._interactive_contour = None
        if self._interactive_label_texts:
            for lbl in self._interactive_label_texts:
                try:
                    lbl.remove()
                except Exception:
                    pass
            self._interactive_label_texts = []

        level = float(self._slider2d.val)
        if level <= 0 or level >= 100:
            fig.canvas.draw_idle()
            return

        level_val = level if normalize else level / 100.0 * self.dlvh.volume_cc
        self._interactive_contour = ax.contour(
            h2d.dose_edges[:-1],
            h2d.let_edges[:-1],
            h2d.values.T,
            levels=[level_val],
            colors="red",
            linewidths=1.5
        )
        self._interactive_label_texts = ax.clabel(
            self._interactive_contour,
            inline=True,
            fontsize=8,
            fmt=lambda _: f"{level:.0f}%"
        )
        fig.canvas.draw_idle()
