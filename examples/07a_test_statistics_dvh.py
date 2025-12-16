"""
07_test_statistics.py
============================
Compare two DLVH cohorts applying different statistical
tests.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from pydlvh.core import DLVH
from pydlvh.utils import _get_bin_centers
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

    # 1) Create synthetic control and ae cohorts
    mu_dose_control, sigma_dose_control = 50.0, 5.0
    dose_shapes = [(x, np.abs(y)) for x, y in zip(np.random.normal(loc=mu_dose_control, scale=1.0, size=100), np.random.normal(loc=sigma_dose_control, scale=1.0, size=100))]
    control_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Control group
    mu_dose_ae, sigma_dose_ae = 52.0, 4.0
    dose_shapes = [(x, np.abs(y)) for x, y in zip(np.random.normal(loc=mu_dose_ae, scale=1.0, size=100), np.random.normal(loc=sigma_dose_ae, scale=1.0, size=100))]
    ae_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Adverse event (AE) group

    # 2) Set uniform binning
    # To ensure that both control and ae aggregates are computed with the same binning,
    # it can be manually set as below. For DVH comparison, it is chosen to consider a common volume binning
    # in order to compare Dx% values.
    volume_range = (0., 100.) # binning from 0 to 100 %
    volume_step = 1 # bin width: 1 %
    volume_edges = np.arange(volume_range[0]+volume_step/2, volume_range[-1]+volume_step, volume_step)  # Dx% with x in [0, 100]

    # 3) Median DVHs
    all_control_dlvhs, median_control_dvh = analyzer.aggregate(dlvhs=control_dlvhs,
                                                               stat="median",
                                                               quantity="dvh",
                                                               aggregateby="volume",
                                                               volume_edges=volume_edges)
    all_ae_dlvhs, median_ae_dvh = analyzer.aggregate(dlvhs=ae_dlvhs,
                                                     stat="median",
                                                     quantity="dvh",
                                                     aggregateby="volume",
                                                     volume_edges=volume_edges)

    # 4) Compute statistical significance between control and AE DVHs (Mann-Whitney u-test).
    # The default settings for DVH comparison is based on binning selected during aggregation.
    alpha = 0.05
    pvalues, significance = analyzer.voxel_wise_Mann_Whitney_test(control_histograms=all_control_dlvhs, 
                                                                  ae_histograms=all_ae_dlvhs,
                                                                  alpha=alpha,
                                                                  correction="fdr_bh")
    # Print significant Dx%
    if np.any(significance):
        print(rf"\nMann-Whitney U-test significant p-values (α={alpha})")
        volume_centers = _get_bin_centers(edges=volume_edges)
        maskedvolumes = volume_centers[(significance) & (volume_centers == volume_centers.astype(int))] # Filter on significance + int volumes
        maskedpvalues = pvalues[(significance) & (volume_centers == volume_centers.astype(int))]
        for volume, pvalue in zip(maskedvolumes, maskedpvalues):
            print(f"D{volume:.0f}: p-value={pvalue:.2f}") # Statistical difference observed (alpha<0.05)

    # 5) Plot median DVHs
    _, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    median_control_dvh.plot(ax=ax, color="C0", label="Control", show_band=True, band_color="C0")
    median_ae_dvh.plot(ax=ax, color="C1", label="AE", show_band=True, band_color="C1")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)
    plt.show()

if __name__ == "__main__":
    main()
