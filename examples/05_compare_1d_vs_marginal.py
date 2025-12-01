"""
05_compare_1d_vs_marginal.py
============================
Compare 1D DVH/LVH with the corresponding marginals extracted
from a cumulative DLVH.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH


def main():
    # Synthetic dose and let distributions (Gaussian)
    n_voxels = 20000
    volume_cc = 8.0
    cumulative = True
    normalize = False
    dose = np.random.normal(loc=50.0, scale=8.0, size=n_voxels)  # Gy
    dose = np.clip(dose, 0, None)
    let = np.random.normal(loc=2.5, scale=0.5, size=n_voxels)    # keV/µm
    let = np.clip(let, 0, None)

    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # 1) Build DVHL/LVH with explicit bin width settings
    dvh_1d = dlvh.dose_volume_histogram(
        bin_width=1.0, cumulative=cumulative, normalize=normalize, aggregatedby="dose"
    )
    lvh_1d = dlvh.let_volume_histogram(
        bin_width=0.1, cumulative=cumulative, normalize=normalize, aggregatedby="let"
    )

    # 2) Build ciumulative DLVH with matching bin width settings
    h2d = dlvh.dose_let_volume_histogram(
        bin_width_dose=1.0,   # match DVH 1D
        bin_width_let=0.1,    # match LVH 1D
        normalize=normalize,
        cumulative=cumulative
    )

    # 3) DVH comparison: 1D vs marginal from 2D DLVH
    edges_1d, values_1d = dvh_1d.get_data(x="edges")
    edges_marg_dose, values_marg_dose = h2d.get_marginals(quantity="dose")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.step(edges_1d[:-1], values_1d, where="post",
             color="C0", lw=2, label="DVH (1D)")
    ax1.step(edges_marg_dose[:-1], values_marg_dose, where="post",
             color="C1", ls="--", lw=2, label="DVH (2D marginal)")
    ax1.set_xlabel("Dose [GyE]")
    ax1.set_ylabel("Volume [%]" if normalize else "Volume [cm³]")
    ax1.set_title("DVH: 1D vs 2D marginal")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.2)
    ax1.legend(frameon=False)

    # 4) LVH comparison: 1D vs marginal from 2D DLVH
    edges_1d, values_1d = lvh_1d.get_data(x="edges")
    edges_marg_let, values_marg_let = h2d.get_marginals(quantity="let")

    ax2.step(edges_1d[:-1], values_1d, where="post",
             color="C0", lw=2, label="LVH (1D)")
    ax2.step(edges_marg_let[:-1], values_marg_let, where="post",
             color="C1", ls="--", lw=2, label="LVH (2D marginal)")
    ax2.set_xlabel(r"LET$_d$ [keV/µm]")
    ax2.set_ylabel("Volume [%]" if normalize else "Volume [cm³]")
    ax2.set_title("LVH: 1D vs 2D marginal")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.2)
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
