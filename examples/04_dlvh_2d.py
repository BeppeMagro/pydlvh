"""
04_dlvh_2d.py
=============
Example usage of the 2D Dose–LET Volume Histogram (DLVH).

- Generate synthetic Gaussian dose and LET values.
- Build a cumulative DLVH (V(D ≥ d, LET ≥ l)).
- Plot the 2D histogram as a heatmap.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH


def main():
    # --- synthetic data ---
    n_voxels = 20000
    volume_cc = 10.0

    dose = np.random.normal(loc=50.0, scale=8.0, size=n_voxels)  # Gy
    dose = np.clip(dose, 0, None)

    let = np.random.normal(loc=2.5, scale=0.5, size=n_voxels)    # keV/µm
    let = np.clip(let, 0, None)

    # --- create DLVH object ---
    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # --- build 2D histogram ---
    h2d = dlvh.dose_let_volume_histogram(
        bin_width_dose=1.0,
        bin_width_let=0.1,
        normalize=True,       # show as %
        cumulative=True       # default = cumulative DLVH
    )

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 6))
    h2d.plot(ax=ax, cmap="plasma", isovolumes=[5, 10, 20], interactive=True)
    

    # ax.set_title("Cumulative Dose–LET Volume Histogram (DLVH)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
