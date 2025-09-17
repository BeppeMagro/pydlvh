"""
01_basic_usage.py
=================
Minimal usage of DLVH: explore DVH and LVH variations
(cumulative vs differential, normalized vs absolute).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH


def main():
    # --- Synthetic Gaussian data ---
    n_voxels = 20000
    volume_cc = 5.0

    # Dose ~ Gaussian (mean 60 Gy, std 2.5 Gy, clipped at 0)
    dose = np.random.normal(loc=60.0, scale=2.5, size=n_voxels)
    dose = np.clip(dose, 0, None)

    # LET ~ Gaussian (mean 5 keV/µm, std 0.6, clipped at 0)
    let = np.random.normal(loc=5.0, scale=0.6, size=n_voxels)
    let = np.clip(let, 0, None)

    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # --- Build DVHs ---
    dvh_cum_norm  = dlvh.dose_volume_histogram(cumulative=True,  normalize=True)
    dvh_cum_abs   = dlvh.dose_volume_histogram(cumulative=True,  normalize=False)
    dvh_diff_norm = dlvh.dose_volume_histogram(cumulative=False, normalize=True)
    dvh_diff_abs  = dlvh.dose_volume_histogram(cumulative=False, normalize=False)

    # --- Build LVHs ---
    lvh_cum_norm  = dlvh.let_volume_histogram(cumulative=True,  normalize=True)
    lvh_cum_abs   = dlvh.let_volume_histogram(cumulative=True,  normalize=False)
    lvh_diff_norm = dlvh.let_volume_histogram(cumulative=False, normalize=True)
    lvh_diff_abs  = dlvh.let_volume_histogram(cumulative=False, normalize=False)

    # --- Plot DVH combos ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    dvh_cum_norm.plot(ax=axes[0, 0], color="C0", label="Cumulative, Norm [%]")
    dvh_cum_abs.plot(ax=axes[0, 1], color="C1", label="Cumulative, Abs [cm³]")
    dvh_diff_norm.plot(ax=axes[1, 0], color="C2", label="Differential, Norm [∑=1]")
    dvh_diff_abs.plot(ax=axes[1, 1], color="C3", label="Differential, Abs [cm³]")

    for ax in axes.flat:
        ax.legend()
        ax.grid(True)
    fig.suptitle("DVH")
    plt.tight_layout()
    plt.show()

    # --- Plot LVH combos ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    lvh_cum_norm.plot(ax=axes[0, 0], color="C0", label="Cumulative, Norm [%]")
    lvh_cum_abs.plot(ax=axes[0, 1], color="C1", label="Cumulative, Abs [cm³]")
    lvh_diff_norm.plot(ax=axes[1, 0], color="C2", label="Differential, Norm [∑=1]")
    lvh_diff_abs.plot(ax=axes[1, 1], color="C3", label="Differential, Abs [cm³]")

    for ax in axes.flat:
        ax.legend()
        ax.grid(True)
    fig.suptitle("LVH")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
