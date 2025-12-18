"""
02_thresholds.py
================
Example of DVH/LVH with threshold filters:
- DVH conditioned on LETd ≥ threshold
- LVH conditioned on Dose ≥ threshold
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH


def main():
    # Synthetic dose and let distributions (Gaussian)
    n_voxels = 30000
    volume_cc = 10.0
    dose = np.random.normal(loc=45.0, scale=8.0, size=n_voxels)
    dose = np.clip(dose, 0, None)
    let = np.random.normal(loc=2.5, scale=0.7, size=n_voxels)
    let = np.clip(let, 0, None)

    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # 1) Build DVHs with thresholds
    dvh_all = dlvh.dose_volume_histogram(cumulative=True, normalize=True)
    dvh_thr = dlvh.dose_volume_histogram(cumulative=True, normalize=True,
                                         let_threshold=2.5)

    # 2) Build LVHs with thresholds
    lvh_all = dlvh.let_volume_histogram(cumulative=True, normalize=True)
    lvh_thr = dlvh.let_volume_histogram(cumulative=True, normalize=True,
                                        dose_threshold=40)

    # 3) Plot DVHs
    _, ax = plt.subplots()
    dvh_all.plot(ax=ax, color="C0", label="DVH (all voxels)")
    dvh_thr.plot(ax=ax, color="C3", linestyle="--", label=r"DVH (LET$_{d}$ ≥ 2.5)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    ax.set_title("DVH with LET threshold")
    plt.tight_layout()
    plt.show()

    # 4) Plot LVH
    _, ax = plt.subplots()
    lvh_all.plot(ax=ax, color="C0", label="LVH (all voxels)")
    lvh_thr.plot(ax=ax, color="C3", linestyle="--", label="LVH (D ≥ 20 GyE)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    ax.set_title("LVH with dose threshold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
