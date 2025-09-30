import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import Histogram1D, Histogram2D


def demo_histogram1d():
    edges = np.linspace(0, 10, 21)         # 20 bin
    centers = 0.5 * (edges[:-1] + edges[1:])

    # --- valori sintetici tipo gaussiana ---
    values_diff = np.exp(-0.5 * ((centers - 5)/1.5)**2) * 100
    err_diff = np.random.rand(len(values_diff)) * 5
    p_lo_diff = values_diff - np.random.rand(len(values_diff)) * 10
    p_hi_diff = values_diff + np.random.rand(len(values_diff)) * 10

    hist_diff = Histogram1D(
        values=values_diff,
        edges=edges,
        quantity="dose",
        normalize=True,
        cumulative=False,
        err=err_diff,
        p_lo=p_lo_diff,
        p_hi=p_hi_diff,
    )

    ax = hist_diff.plot(color="C0", label="Differential values")
    ax.legend()
    ax.set_title("Histogram1D differenziale con bande ±std e percentili")
    plt.show()

    # --- valori sintetici cumulativi decrescenti ---
    values_cum = np.linspace(100, 0, len(centers))
    err_cum = np.random.rand(len(values_cum)) * 3
    p_lo_cum = values_cum - np.random.rand(len(values_cum)) * 5
    p_hi_cum = values_cum + np.random.rand(len(values_cum)) * 5

    hist_cum = Histogram1D(
        values=values_cum,
        edges=edges,
        quantity="dose",
        normalize=True,
        cumulative=True,
        err=err_cum,
        p_lo=p_lo_cum,
        p_hi=p_hi_cum,
    )

    ax = hist_cum.plot(color="C1", label="Cumulative values")
    ax.legend()
    ax.set_title("Histogram1D cumulativo con bande ±std e percentili")
    plt.show()


def demo_histogram2d():
    dose_edges = np.linspace(0, 10, 21)
    let_edges = np.linspace(0, 5, 11)
    nd, nl = len(dose_edges)-1, len(let_edges)-1

    values = np.random.rand(nd, nl) * 100
    err = np.random.rand(nd, nl) * 10
    p_lo = values - np.random.rand(nd, nl) * 15
    p_hi = values + np.random.rand(nd, nl) * 15

    hist2d = Histogram2D(
        values=values,
        dose_edges=dose_edges,
        let_edges=let_edges,
        normalize=True,
        cumulative=True,
        err=err,
        p_lo=p_lo,
        p_hi=p_hi,
    )

    # Plot central values
    hist2d.plot(cmap="viridis", mode="values")
    plt.title("Histogram2D - central values")
    plt.show()

    # Plot std map
    hist2d.plot(cmap="magma", mode="err")
    plt.title("Histogram2D - std map")
    plt.show()

    # Plot percentile high map
    hist2d.plot(cmap="plasma", mode="p_hi")
    plt.title("Histogram2D - upper percentile map")
    plt.show()


if __name__ == "__main__":
    demo_histogram1d()
    demo_histogram2d()
