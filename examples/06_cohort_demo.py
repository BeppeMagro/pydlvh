"""
06_cohort_demo.py
============================
Handle a cohort of DLVHs and compute aggregate statistics.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH
import pydlvh.analyzer as analyzer

def create_synthetic_patient(n_voxels=4000,
                             mu_dose=30.0, sigma_dose=7.0,
                             mu_let=30.0, sigma_let=10.0,
                             volume_rng=(80.0, 120.0)):
    """
        Create synthetic dose, let and relative volumes distributions.
    """

    dose = np.random.normal(loc=mu_dose, scale=sigma_dose, size=n_voxels)
    dose = np.clip(dose, 0.0, None)
    let = np.random.normal(loc=mu_let, scale=sigma_let, size=n_voxels)
    let = np.clip(let, 0.0, None)

    relative_volumes = np.exp(-0.5 * ((dose - mu_dose) / max(sigma_dose, 1e-6))**2)
    if not np.any(relative_volumes > 0):
        relative_volumes[:] = 1.0
    relative_volumes = relative_volumes / relative_volumes.sum()

    volume_cc = np.random.uniform(*volume_rng)

    return DLVH(dose=dose, let=let, volume_cc=volume_cc, relative_volumes=relative_volumes)

def main():
    np.random.seed(7)

    # 1) Create synthetic patients
    dose_shapes = [(29, 6.5), (30, 7.0), (33, 8.0), (29, 6.5), (32, 7.5), (30, 5.0), (28, 6.5)]
    dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes]

    # 2) Mean DVH (± iqr)
    _, mean_dvh = analyzer.aggregate(dlvhs=dlvhs,
                                  quantity="dvh",
                                  stat="mean",
                                  normalize=True,
                                  cumulative=True)
    ax = mean_dvh.plot(color="C0", label="Mean DVH", show_band=True)
    ax.legend(loc="best", frameon=False)
    ax.set_title("Mean DVH")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # 3) Mean LVH (± iqr)
    _, median_lvh = analyzer.aggregate(dlvhs=dlvhs,
                                    quantity="lvh",
                                    stat="median",
                                    normalize=True,
                                    cumulative=True)
    ax = median_lvh.plot(color="C1", label="Mean LVH", show_band=True)
    ax.legend(loc="best", frameon=False)
    ax.set_title("Mean LVH")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # 4) Median DLVH  (+ P25/P75)
    _, median_dlvh = analyzer.aggregate(dlvhs=dlvhs,
                                        quantity="dlvh",
                                        stat="median",
                                        normalize=True,
                                        cumulative=True)
    # Plot
    _, ax = plt.subplots(1, 3, figsize=(12, 4))
    median_dlvh.plot(mode="values", isovolumes=[20, 50, 80], ax=ax[0])
    ax[0].set_title("Median DLVH 2D")
    # 25th percentile
    median_dlvh.plot(mode="p_lo", ax=ax[1], isovolumes=[20, 50, 80])
    ax[1].set_title("DLVH 25th percentile (IQR lower)")
    # 75th percentile
    median_dlvh.plot(mode="p_hi", ax=ax[2], isovolumes=[20, 50, 80])
    ax[2].set_title("DLVH 75th percentile (IQR upper)")
    plt.tight_layout()
    plt.show()

    # 5) Check median DVH/LVH from DLVH margins 
    ax = median_dlvh.plot_marginals(quantity="dose", lw=2, label="extracted DVH")
    ax.legend(loc="best")
    ax.set_title("DVH from median DLVH (median + IQR)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
