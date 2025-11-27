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

    # 1) Build DVHs
    dvh_cum_norm  = dlvh.dose_volume_histogram(cumulative=True,  normalize=True)
    dvh_cum_abs   = dlvh.dose_volume_histogram(cumulative=True,  normalize=False)
    dvh_diff_norm = dlvh.dose_volume_histogram(cumulative=False, normalize=True)
    dvh_diff_abs  = dlvh.dose_volume_histogram(cumulative=False, normalize=False)

    # 2) Build LVHs
    # lvh_cum_norm  = dlvh.let_volume_histogram(cumulative=True,  normalize=True)
    # lvh_cum_abs   = dlvh.let_volume_histogram(cumulative=True,  normalize=False)
    # lvh_diff_norm = dlvh.let_volume_histogram(cumulative=False, normalize=True)
    # lvh_diff_abs  = dlvh.let_volume_histogram(cumulative=False, normalize=False)

    # 3) Plot DVHs
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    dvh_cum_norm.plot(ax=axes[0, 0], color="C0", label="Cumulative, Norm [%]")
    dvh_cum_abs.plot(ax=axes[0, 1], color="C1", label="Cumulative, Abs [cm³]")
    dvh_diff_norm.plot(ax=axes[1, 0], color="C0", alpha=0.7, label="Differential, Norm [%]")
    dvh_diff_abs.plot(ax=axes[1, 1], color="C1", alpha=0.7, label="Differential, Abs [cm³]")

    for ax in axes.flat:
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.2)
    fig.suptitle("DVH")
    plt.tight_layout()
    plt.show()

    # 4) Plot LVHs
    # fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    # lvh_cum_norm.plot(ax=axes[0, 0], color="C0", label="Cumulative, Norm [%]")
    # lvh_cum_abs.plot(ax=axes[0, 1], color="C1", label="Cumulative, Abs [cm³]")
    # lvh_diff_norm.plot(ax=axes[1, 0], color="C0", alpha=0.7, label="Differential, Norm [%]")
    # lvh_diff_abs.plot(ax=axes[1, 1], color="C1", alpha=0.7, label="Differential, Abs [cm³]")

    # for ax in axes.flat:
    #     ax.legend(loc="best", frameon=False)
    #     ax.grid(True, alpha=0.2)
    # fig.suptitle("LVH")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
