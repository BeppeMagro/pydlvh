"""
01a_dvh_binning.py
=================
Basic usage of DLVH: explore DVH (and LVH)
binning options (dose vs volume binning, bin centers
vs bin edges vs bin width).
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

    # 1) Build DVHs (compare dose/volume binning and binning approach(bin centers/edges/width))
    dose_range = (0., 70.) # binning from 0 to 70 Gy
    dose_step = 0.1 # bin width: 0.1 Gy
    volume_range = (0., 100.) # binning from 0 to 100 %
    volume_step = 0.01 # bin width: 0.01 %

    dvh_dose_centers        = dlvh.dose_volume_histogram(cumulative=True, normalize=True,
                                                         bin_centers=np.arange(dose_range[0]+dose_step/2, dose_range[-1]+dose_step/2, dose_step),
                                                         aggregatedby="dose")
    dvh_volume_centers      = dlvh.dose_volume_histogram(cumulative=True, normalize=True,
                                                         bin_centers=np.arange(volume_range[0]+volume_step/2, volume_range[-1]+volume_step/2, volume_step), 
                                                         aggregatedby="volume")
    dvh_dose_edges          = dlvh.dose_volume_histogram(cumulative=True, normalize=True,
                                                         bin_edges=np.arange(dose_range[0], dose_range[-1]+dose_step, dose_step),
                                                         aggregatedby="dose")
    dvh_volume_edges        = dlvh.dose_volume_histogram(cumulative=True, normalize=True, 
                                                         bin_edges=np.arange(volume_range[0], volume_range[-1]+volume_step, volume_step), 
                                                         aggregatedby="volume")
    dvh_dose_bin_width      = dlvh.dose_volume_histogram(cumulative=True, normalize=True,
                                                         bin_width=dose_step, aggregatedby="dose")
    dvh_volume_bin_width    = dlvh.dose_volume_histogram(cumulative=True, normalize=True,
                                                         bin_width=volume_step, aggregatedby="volume")
    
    # 2) Plot DVHs with manual binning 
    fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True)
    dvh_dose_centers.plot(ax=axes[0, 0], color="C0", label="Dose binning, Centers")
    dvh_volume_centers.plot(ax=axes[0, 1], color="C1", label="Volume binning, Centers")
    dvh_dose_edges.plot(ax=axes[1, 0], color="C2", label="Dose binning, Bin edges")
    dvh_volume_edges.plot(ax=axes[1, 1], color="C3", label="Volume binning, Bin edges")
    dvh_dose_bin_width.plot(ax=axes[2, 0], color="C4", label="Dose binning, Bin width")
    dvh_volume_bin_width.plot(ax=axes[2, 1], color="C5", label="Volume binning, Bin width")

    for ax in axes.flat:
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.2)
    fig.suptitle("DVH")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
