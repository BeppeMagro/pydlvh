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
        Dose is uniform in [0, dose_max], LET is Gaussian troncated for 
        values >=0, the relative weights are Gaussian.
    """

    dose = np.random.normal(loc=mu_dose, scale=sigma_dose, size=n_voxels)
    dose = np.clip(dose, 0, None)
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
    dose_shapes = [(27, 6.0), (30, 7.0), (33, 8.0), (29, 6.5), (32, 7.5), (31, 5.0), (26, 9)]
    dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes]

    # 2) Median DVH (± iqr)
    median_dvh = analyzer.aggregate(dlvhs=dlvhs,
                                    quantity="dvh",
                                    stat="median",
                                    normalize=True,
                                    cumulative=True)

    ax = median_dvh.plot(color="C0", label="Median DVH", show_band=True)
    ax.legend(loc="best", frameon=False)
    ax.set_title("Median DVH")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # 3) Mean LVH (± iqr)
    # median_lvh = analyzer.aggregate(dlvhs=dlvhs,
    #                                 quantity="lvh",
    #                                 stat="mean",
    #                                 normalize=True,
    #                                 cumulative=True)
    # ax = median_lvh.plot(color="C1", label="Mean LVH", show_band=True)
    # ax.legend(loc="best", frameon=False)
    # ax.set_title("Median LVH")
    # ax.grid(True, alpha=0.2)
    # plt.tight_layout()
    # plt.show()

    # # # 4) Median DLVH  (+ P25/P75)
    # median_dlvh = analyzer.aggregate(dlvhs=dlvhs,
    #                                  quantity="dlvh",
    #                                  stat="median",
    #                                  normalize=True,
    #                                  cumulative=True)
    # # Median
    # median_dlvh.plot(mode="values", isovolumes=[5, 10, 20])
    # plt.title("Median DLVH 2D")
    # plt.tight_layout()
    # plt.show()
    # # 25th percentile
    # median_dlvh.plot(mode="p_lo")
    # plt.title("DLVH 25th percentile (IQR lower)")
    # plt.tight_layout()
    # plt.show()
    # # 75th percentile
    # median_dlvh.plot(mode="p_hi")
    # plt.title("DLVH 75th percentile (IQR upper)")
    # plt.tight_layout()
    # plt.show()

    # # # 5) Check median DVH/LVH from DLVH margins 
    # dvh_from2d = analyzer.aggregate_marginals(dlvhs=dlvhs,
    #                                           quantity="dose", stat="median",
    #                                           normalize=True, cumulative=True)
    # ax = dvh_from2d.plot(color="C2", label="extracted DVH", show_band=True)
    # ax.legend(loc="best")
    # ax.set_title("DVH from median DLVH (median + IQR)")
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
