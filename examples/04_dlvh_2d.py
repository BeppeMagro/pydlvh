"""
04_dlvh_2d.py
=============
Basic usage of the 2D Dose–LET Volume Histogram (DLVH).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH


def main():
    # Synthetic dose and let distributions (Gaussian)
    n_voxels = 20000
    volume_cc = 10.0
    dose = np.random.normal(loc=50.0, scale=8.0, size=n_voxels)
    dose = np.clip(dose, 0, None)
    let = np.random.normal(loc=2.5, scale=0.5, size=n_voxels)
    let = np.clip(let, 0, None)

    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # 1) Build DLVH
    h2d = dlvh.dose_let_volume_histogram(
        bin_width_dose=1.0,
        bin_width_let=0.1,
        normalize=True,
        cumulative=True
    )

    # 2) Plot DLVH (with interactive isovolume slider)
    fig, ax = plt.subplots()
    isovolumes_colors = [ "#21918c", "#DF0E6F", "#fde725"]
    h2d.plot(ax=ax, isovolumes=[20, 50, 80], isovolumes_colors=isovolumes_colors, interactive=True)
    # ax.set_title("Cumulative Dose–LET Volume Histogram (DLVH)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
