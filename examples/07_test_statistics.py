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
import pydlvh.analyzer as analyzer

def create_synthetic_patient(n_voxels=4000,
                             dose_max=60.0,
                             mu_dose=30.0, sigma_dose=7.0,
                             mu_let=30.0, sigma_let=10.0,
                             volume_rng=(80.0, 120.0)):
    """
        Create synthetic dose, let and relative volumes distributions.
        Dose is uniform in [0, dose_max], LET is Gaussian troncated for 
        values >=0, the relative weights are Gaussian.
    """
    dose = np.random.uniform(0.0, dose_max, size=n_voxels)
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
    dose_shapes = [(50, 0.8), (48, 1.7), (48, 2.5), (52, 2.2), (54, 1.5)]
    control_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Control group
    dose_shapes = [(27, 0.5), (26, 2.0), (30, 1.3), (29, 2.3), (28, 1.4)]
    ae_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Adverse event (AE) group
    all_dlvhs = control_dlvhs + ae_dlvhs

    # 2) Set uniform binning
    # To ensure that both control and ae aggregates are computed with the same binning,
    # it can be manually set as below. For DVH comparison, it is chosen to consider a common volume binning
    # in order to compare Dx% values.
    volume_range = (0., 100.) # binning from 0 to 100 %
    volume_step = 0.01 # bin width: 0.01 %
    volume_edges = np.arange(volume_range[0], volume_range[-1]+volume_step, volume_step)  # Dx% with x in [0, 100]

    # 3) Median DVHs
    print("\nComputing median DVHs for control and AE cohorts...")
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

    # 4) Compute statistical significance between control and AE DVHs (Mann-Whitney u-test with Bonferroni correction).
    # The default settings for DVH comparison is based on binning selected during aggregation.
    print("\nPerforming statistical test between control and AE cohorts...")
    all_control_dlvhs, _ = analyzer.get_all_cohort_histograms(dlvhs=control_dlvhs, quantity="dvh", x_edges=volume_edges)
    all_ae_dlvhs, _ = analyzer.get_all_cohort_histograms(dlvhs=ae_dlvhs, quantity="dvh", x_edges=volume_edges)

    pvalues, significance = analyzer.voxel_wise_Mann_Whitney_test(control_histograms=all_control_dlvhs, 
                                                                  ae_histograms=all_ae_dlvhs,
                                                                  volume_grid=volume_edges,
                                                                  alpha=0.05,
                                                                  correction="fdr_bh")
    # Print significant Dx%
    if np.any(significance):
        maskedvolumes = volume_edges[significance]
        print("\nSignificant difference for volumes:", maskedvolumes)
        print("p-values:", pvalues[significance])

    
    # 6) Plot median DVHs
    _, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    median_control_dvh.plot(ax=ax, color="C0", label="Control", show_band=True)
    median_ae_dvh.plot(ax=ax, color="C1", label="AE", show_band=True)
    ax.set_legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)
    plt.show()

    # 5) Median DLVHs 
    # median_control_dlvh = analyzer.aggregate(dlvhs=control_dlvhs,
    #                                          stat="median",
    #                                          quantity="dlvh",
    #                                          dose_edges=dose_edges,
    #                                          let_edges=let_edges)
    # median_ae_dlvh = analyzer.aggregate(dlvhs=ae_dlvhs,
    #                                     stat="median",
    #                                     quantity="dlvh",
    #                                     dose_edges=dose_edges,
    #                                     let_edges=let_edges)

    """   

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    median_control_dvh.plot(ax=axes[0], color="C0", label="Control", show_band=True)
    median_ae_dvh.plot(ax=axes[1], color="C1", label="AE", show_band=True)
    fig.suptitle("Median DVH")

    # 3) Investigate for possible statistically significant difference between median control and AE DVHs (Mann-Whitney u-test)
    # Including Bonferroni correction
    alpha = 0.05
    p_values, significance = analyzer.voxel_wise_Mann_Whitney_test(median_control_dvh, median_ae_dvh, alpha=alpha)#, correction="holm")
    print(significance)
    # Plot significant DVH points
    # Padding
    edges = np.insert(dose_edges, 0, 0.0)
    values = [np.insert(histo.values, 0, histo.values[0]) for histo in [median_control_dvh, median_ae_dvh]]
    significance = np.insert(significance, 0, False)
    # Plot
    for i, ax in enumerate(axes):
        ax.scatter(edges[:-1][significance], values[i][significance], label=f"p<{alpha:.2f}", color="red")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()"""


if __name__ == "__main__":
    main()
