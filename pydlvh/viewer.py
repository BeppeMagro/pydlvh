import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from .core import DLVH


class DLVHViewer:
    """
    Viewer for DVH, LVH, and 2D DLVH.

    - Use plot1D() for interactive DVH/LVH viewer with slider.
    - Use plot2D() for Dose–LET Volume Histogram with optional isovolume lines.
    """

    def __init__(self, dlvh: DLVH):
        """
        Parameters
        ----------
        dlvh : DLVH
            Instance of DLVH with dose, let, and volume_cc.
        """
        self.dlvh = dlvh
        self.fig = None
        self.ax_dvh = None
        self.ax_lvh = None
        self.slider = None
        self.vline = None
        self.cross = None
        self.annotation = None
        self.lvh_edges = None
        self.lvh_values = None

    # =========================================================
    # 1D interactive DVH/LVH
    # =========================================================
    def plot1D(self):
        """Interactive viewer with DVH (cumulative) and LVH (cumulative, absolute)."""
        self.fig, (self.ax_dvh, self.ax_lvh) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.25, wspace=0.3)

        # --- DVH cumulative ---
        dvh = self.dlvh.dose_volume_histogram(cumulative=True, normalize=True)
        edges, values = dvh.edges, dvh.values
        if edges[0] > 0:
            edges = np.insert(edges, 0, 0.0)
            values = np.insert(values, 0, values[0])

        (self.dvh_line,) = self.ax_dvh.step(
            edges[:-1], values, where="post", color="C0", label="DVH (cumulative)"
        )
        self.ax_dvh.set_title("DVH (cumulative)")
        self.ax_dvh.set_xlabel("Dose [Gy]")
        self.ax_dvh.set_ylabel("Volume [%]")
        self.ax_dvh.set_xlim(left=0)
        self.ax_dvh.set_ylim(bottom=0)
        self.ax_dvh.grid(True)
        self.ax_dvh.legend(loc="upper right")

        # Vertical threshold line on DVH
        self.vline = self.ax_dvh.axvline(0, color="red", linestyle="--", lw=1.5)

        # --- DVH differential overlay ---
        dvh_diff = self.dlvh.dose_volume_histogram(cumulative=False, normalize=True)
        edges_d, values_d = dvh_diff.edges, dvh_diff.values
        centers = 0.5 * (edges_d[:-1] + edges_d[1:])
        widths = np.diff(edges_d)

        ax_dvh_diff = self.ax_dvh.twinx()
        ax_dvh_diff.bar(
            centers, values_d,
            width=widths,
            color="C0", alpha=0.3,
            align="center", label="DVH (differential)"
        )
        ax_dvh_diff.set_ylabel("Differential volume [%]")
        ax_dvh_diff.set_ylim(bottom=0)

        # --- LVH cumulative (absolute) ---
        lvh = self.dlvh.let_volume_histogram(cumulative=True, normalize=False)
        self.lvh_edges, self.lvh_values = lvh.edges, lvh.values
        if self.lvh_edges[0] > 0:
            self.lvh_edges = np.insert(self.lvh_edges, 0, 0.0)
            self.lvh_values = np.insert(self.lvh_values, 0, self.lvh_values[0])

        (self.lvh_line,) = self.ax_lvh.step(
            self.lvh_edges[:-1], self.lvh_values, where="post",
            color="C1", label="LVH"
        )
        self.ax_lvh.set_title("LVH (cumulative)")
        self.ax_lvh.set_xlabel("LET [keV/µm]")
        self.ax_lvh.set_ylabel("Volume [cm³]")
        self.ax_lvh.set_xlim(left=0)
        self.ax_lvh.set_ylim(bottom=0)
        self.ax_lvh.grid(True)
        self.ax_lvh.legend()

        # Cross marker + text annotation for LVH
        self.cross, = self.ax_lvh.plot([], [], "ro", markersize=6)
        self.annotation = self.ax_lvh.text(
            0.5, -0.2, "", transform=self.ax_lvh.transAxes,
            ha="center", va="top", fontsize=9
        )

        # --- slider ---
        bin_width = np.diff(dvh.edges).mean()
        ax_slider = plt.axes([0.12, 0.1, 0.35, 0.05])
        self.slider = Slider(
            ax_slider, "Dose thr [Gy]",
            valmin=0,
            valmax=np.max(self.dlvh.dose),
            valinit=0,
            valstep=bin_width
        )
        self.slider.on_changed(self._update1D)

        # connect clicks
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        plt.show()

    def _update1D(self, val: float):
        """Update LVH when slider moves."""
        thr = self.slider.val
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

    def _on_click(self, event):
        """Handle clicks on LVH panel."""
        if event.inaxes != self.ax_lvh or not self.lvh_edges.any():
            return

        centers = 0.5 * (self.lvh_edges[:-1] + self.lvh_edges[1:])
        idx = np.argmin(np.abs(centers - event.xdata))

        let_val = centers[idx]
        vol_abs = self.lvh_values[idx]
        vol_rel = 100 * vol_abs / self.dlvh.volume_cc

        self.cross.set_data([let_val], [vol_abs])
        self.annotation.set_text(
            f"LET ≈ {let_val:.2f} keV/µm, "
            f"V = {vol_abs:.2f} cm³ ({vol_rel:.1f} %)"
        )

        self.fig.canvas.draw_idle()

    # =========================================================
    # 2D DLVH
    # =========================================================
    def plot2D(self, *,
               bin_width_dose: float = 1.0,
               bin_width_let: float = 0.1,
               normalize: bool = True,
               cumulative: bool = True,
               isovolumes: list = None,
               cmap: str = "plasma",
               interactive: bool = False):
        """
        Plot the 2D Dose–LET Volume Histogram (DLVH).
        """
        h2d = self.dlvh.dose_let_volume_histogram(
            bin_width_dose=bin_width_dose,
            bin_width_let=bin_width_let,
            normalize=normalize,
            cumulative=cumulative
        )

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=False)

        if interactive:
            fig.subplots_adjust(bottom=0.15)
        else:
            fig.subplots_adjust(bottom=0.12)

        mesh = ax.pcolormesh(h2d.dose_edges,
                             h2d.let_edges,
                             h2d.values.T,
                             cmap=cmap)
        ax.set_xlabel("Dose [Gy]")
        ax.set_ylabel("LET [keV/µm]")
        ax.set_title("2D Dose–LET Volume Histogram")

        if cumulative:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Volume [%]" if normalize else "Volume [cm³]")

        if isovolumes:
            levels = (isovolumes if normalize
                      else [p/100.0 * self.dlvh.volume_cc for p in isovolumes])
            CS = ax.contour(h2d.dose_edges[:-1],
                            h2d.let_edges[:-1],
                            h2d.values.T,
                            levels=levels,
                            colors="white",
                            linewidths=1.2)
            ax.clabel(CS, inline=True, fontsize=8, fmt="%g")

        if interactive:
            from matplotlib.widgets import Slider

            # Stato per linea/etichette interattive
            self._interactive_contour = None
            self._interactive_label_texts = []
            self._h2d_cache = h2d
            self._ax2d = ax
            self._fig2d = fig
            self._normalize2d = normalize

            # Slider ben sotto al grafico
            ax_slider = plt.axes([0.15, 0.02, 0.7, 0.04])
            self._slider2d = Slider(ax_slider,
                                    "Isovol [%]",
                                    valmin=0,
                                    valmax=100,
                                    valinit=0,  
                                    valstep=1)
            self._slider2d.on_changed(self._update2D)

        plt.show()

    def _update2D(self, val: float):
        """Update interactive isovolume line in 2D DLVH plot."""
        h2d = self._h2d_cache
        ax = self._ax2d
        fig = self._fig2d
        normalize = self._normalize2d

        # clear old contour
        if self._interactive_contour is not None:
            for coll in self._interactive_contour.collections:
                coll.remove()
            self._interactive_contour = None
        # clear old labels
        if self._interactive_label_texts:
            for lbl in self._interactive_label_texts:
                try:
                    lbl.remove()
                except Exception:
                    pass
            self._interactive_label_texts = []

        level = self._slider2d.val
        if level <= 0 or level >= 100:
            fig.canvas.draw_idle()
            return

        level_val = level if normalize else level/100.0 * self.dlvh.volume_cc
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




