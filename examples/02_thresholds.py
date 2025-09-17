"""
02_thresholds.py
================
Example of DVH/LVH with threshold filters:
- DVH conditioned on LET ≥ threshold
- LVH conditioned on Dose ≥ threshold
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH


def main():
    # --- Synthetic Gaussian data ---
    n_voxels = 30000
    volume_cc = 10.0

    # Dose ~ Gaussian (mean 45 Gy, std 8 Gy, clipped at 0)
    dose = np.random.normal(loc=45.0, scale=8.0, size=n_voxels)
    dose = np.clip(dose, 0, None)

    # LET ~ Gaussian (mean 2.5 keV/µm, std 0.7, clipped at 0)
    let = np.random.normal(loc=2.5, scale=0.7, size=n_voxels)
    let = np.clip(let, 0, None)

    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # --- Build histograms with thresholds ---
    dvh_all = dlvh.dose_volume_histogram(cumulative=True, normalize=True)
    dvh_thr = dlvh.dose_volume_histogram(cumulative=True, normalize=True,
                                         let_threshold=2.5)

    lvh_all = dlvh.let_volume_histogram(cumulative=True, normalize=True)
    lvh_thr = dlvh.let_volume_histogram(cumulative=True, normalize=True,
                                        dose_threshold=20.0)

    # --- Plot DVH comparison ---
    fig, ax = plt.subplots(figsize=(6, 5))
    dvh_all.plot(ax=ax, color="C0", label="DVH (all voxels)")
    dvh_thr.plot(ax=ax, color="C1", linestyle="--", label="DVH (LET ≥ 2.5)")
    ax.legend()
    ax.grid(True)
    ax.set_title("DVH with LET threshold")
    plt.tight_layout()
    plt.show()

    # --- Plot LVH comparison ---
    fig, ax = plt.subplots(figsize=(6, 5))
    lvh_all.plot(ax=ax, color="C0", label="LVH (all voxels)")
    lvh_thr.plot(ax=ax, color="C3", linestyle="--", label="LVH (Dose ≥ 20 Gy)")
    ax.legend()
    ax.grid(True)
    ax.set_title("LVH with Dose threshold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
