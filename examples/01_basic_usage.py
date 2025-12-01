"""
01_basic_usage.py
=================
Basic usage of DLVH: explore DVH and LVH contruction
(cumulative vs differential, normalized vs absolute).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH


def main():
    # Synthetic dose and let distributions (Gaussian)
    n_voxels = 20000
    volume_cc = 5.0
    dose = np.random.normal(loc=60.0, scale=2.5, size=n_voxels)
    dose = np.clip(dose, 0, None)
    let = np.random.normal(loc=5.0, scale=0.6, size=n_voxels)
    let = np.clip(let, 0, None)

    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # 1) Build DVHs (defualt binning)
    # For cumulative DVH, volume parsing is used by default (this allows easier Dx% extraction): aggregatedby="volume"
    # For differential DVH, dose parsing is used by default: aggregatedby="dose"/"let"
    dvh_cum_norm  = dlvh.dose_volume_histogram(cumulative=True,  normalize=True)
    dvh_cum_abs   = dlvh.dose_volume_histogram(cumulative=True,  normalize=False)
    dvh_diff_norm = dlvh.dose_volume_histogram(cumulative=False, normalize=True)
    dvh_diff_abs  = dlvh.dose_volume_histogram(cumulative=False, normalize=False)

    # 2) Build LVHs (defualt binning)
    lvh_cum_norm  = dlvh.let_volume_histogram(cumulative=True,  normalize=True)
    lvh_cum_abs   = dlvh.let_volume_histogram(cumulative=True,  normalize=False)
    lvh_diff_norm = dlvh.let_volume_histogram(cumulative=False, normalize=True)
    lvh_diff_abs  = dlvh.let_volume_histogram(cumulative=False, normalize=False)

    # 3) Plot DVHs
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    dvh_cum_norm.plot(ax=axes[0, 0], color="C0", label="Cumulative, Norm [%]")
    dvh_cum_abs.plot(ax=axes[0, 1], color="C1", label="Cumulative, Abs [cm³]")
    dvh_diff_norm.plot(ax=axes[1, 0], color="C0", label="Differential, Norm [%]")
    dvh_diff_abs.plot(ax=axes[1, 1], color="C1", label="Differential, Abs [cm³]")

    for ax in axes.flat:
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.2)
    fig.suptitle("DVH")
    plt.tight_layout()
    plt.show()

    # 4) Plot LVHs
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    lvh_cum_norm.plot(ax=axes[0, 0], color="C2", label="Cumulative, Norm [%]")
    lvh_cum_abs.plot(ax=axes[0, 1], color="C3", label="Cumulative, Abs [cm³]")
    lvh_diff_norm.plot(ax=axes[1, 0], color="C2", label="Differential, Norm [%]")
    lvh_diff_abs.plot(ax=axes[1, 1], color="C3", label="Differential, Abs [cm³]")

    for ax in axes.flat:
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.2)
    fig.suptitle("LVH")
    plt.tight_layout()
    plt.show()

    # 5) Build DVHs (dose binning)
    dvh_dose_centers        = dlvh.dose_volume_histogram(cumulative=True, normalize=True, bin_centers=np.arange(0.5, 70.5, 0.5), aggregatedby="dose")
    dvh_dose_bin_width      = dlvh.dose_volume_histogram(cumulative=True, normalize=False, bin_width=0.5, aggregatedby="dose") # bin width: 0.5 Gy
    dvh_volume_centers      = dlvh.dose_volume_histogram(cumulative=True, normalize=True, bin_centers=np.arange(0., 100.5, 0.5), aggregatedby="volume")
    dvh_volume_bin_width    = dlvh.dose_volume_histogram(cumulative=True, normalize=False, bin_width=0.01, aggregatedby="volume") # bin width: 0.01 cc

    # 6) Plot DVHs with manual binning 
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    dvh_dose_centers.plot(ax=axes[0, 0], color="C0", label="Dose binning, Centers")
    dvh_dose_bin_width.plot(ax=axes[0, 1], color="C1", label="Dose binning, Bin width")
    dvh_volume_centers.plot(ax=axes[1, 0], color="C2", label="Volume binning, Centers")
    dvh_volume_bin_width.plot(ax=axes[1, 1], color="C3", label="Volume binning, Bin width")

    for ax in axes.flat:
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.2)
    fig.suptitle("LVH")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
