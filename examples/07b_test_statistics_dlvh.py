"""
07b_test_statistics_dlvh.py
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
    mu_dose_control, sigma_dose_control = 52.0, 5.0
    dose_shapes = [(x, np.abs(y)) for x, y in zip(np.random.normal(loc=mu_dose_control, scale=3.0, size=100), np.random.normal(loc=sigma_dose_control, scale=1.0, size=100))]
    control_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Control group
    mu_dose_ae, sigma_dose_ae = 50.0, 5.0
    dose_shapes = [(x, np.abs(y)) for x, y in zip(np.random.normal(loc=mu_dose_ae, scale=3.0, size=100), np.random.normal(loc=sigma_dose_ae, scale=1.0, size=100))]
    ae_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Adverse event (AE) group

    # 2) Median DLVHs
    # Set manual uniform dose+let binning for aggregation
    dose_edges = np.arange(0., 70.1, 0.1)  # D in [0, 70] with step 0.1 Gy
    let_edges = np.arange(0., 50.1, 0.1)  # D in [0, 50] with step 0.1 keV/um
    all_control_dlvhs, median_control_dlvh = analyzer.aggregate(dlvhs=control_dlvhs,
                                                                stat="median",
                                                                quantity="dlvh",
                                                                dose_edges=dose_edges,
                                                                let_edges=let_edges)
    all_ae_dlvhs, median_ae_dlvh = analyzer.aggregate(dlvhs=ae_dlvhs,
                                                      stat="median",
                                                      quantity="dlvh",
                                                      dose_edges=dose_edges,
                                                      let_edges=let_edges)

    # 3) Compute statistical significance between control and AE DLVHs (Mann-Whitney u-test).
    alpha = 0.05
    pvalues, significance = analyzer.voxel_wise_Mann_Whitney_test(control_histograms=all_control_dlvhs, 
                                                                  ae_histograms=all_ae_dlvhs,
                                                                  alpha=alpha,
                                                                  correction="fdr_bh")
    # Print 5 most significant DLx%
    if np.any(significance):
        print(f"\nMann-Whitney U-test significant p-values (α={alpha}, Bonferroni-Holm corrected):")
        maskedvolumes = volumes[significance]
        maskedpvalues = pvalues[significance]
        for volume, pvalue in zip(maskedvolumes, maskedpvalues):
            print(f"D{volume:.0f}%: p-value={pvalue:.4f}") # Statistical difference observed (alpha<0.05)
    
    # TODO: auc score

    # 4) Plot median DLVHs
    _, ax = plt.subplots(1, 2, figsize=(9, 6.5))
    median_control_dlvh.plot(ax=ax[0])
    median_ae_dlvh.plot(ax=ax[1])
    plt.show()

if __name__ == "__main__":
    main()
