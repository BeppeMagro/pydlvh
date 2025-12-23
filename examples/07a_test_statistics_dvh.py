"""
07a_test_statistics_dvh.py
============================
Compare two DVH/LVH cohorts applying different statistical
tests.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
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

    # 2) Median DVHs
    # Set manual uniform volume binning for aggregation
    volumes = np.arange(0., 100.01, 0.01)  # V in [0, 100] with step 0.1%
    all_control_dlvhs, median_control_dvh = analyzer.aggregate(dlvhs=control_dlvhs,
                                                               stat="median",
                                                               quantity="dvh",
                                                               aggregateby="volume",
                                                               centers=volumes)
    all_ae_dlvhs, median_ae_dvh = analyzer.aggregate(dlvhs=ae_dlvhs,
                                                     stat="median",
                                                     quantity="dvh",
                                                     aggregateby="volume",
                                                     centers=volumes)
    
    # 3) Compute statistical significance between control and AE DVHs (Mann-Whitney u-test)
    # The default settings for DVH comparison is based on binning selected during aggregation
    alpha = 0.05
    volumes = np.arange(0., 101., 1.0)  # Dx% with integer x in [0, 100]
    pvalues, significance = analyzer.voxel_wise_Mann_Whitney_test(control_histograms=all_control_dlvhs, 
                                                                  ae_histograms=all_ae_dlvhs,
                                                                  dose_at_volumes=volumes,
                                                                  alpha=alpha,
                                                                  correction="fdr_bh")
    # Print (top 5 most) significant Dx%
    if np.any(significance):
        masked_volumes = volumes[significance]
        masked_pvalues = pvalues[significance]
        df = pd.DataFrame({
            "Dx%": masked_volumes,
            "p-value": masked_pvalues
        })

        df = df.sort_values("p-value").head(5)
        print(f"\nTop 5 Mann–Whitney U-test most significant Dx% (α={alpha}, BH corrected):\n")
        print(df.to_markdown(index=False, floatfmt=".4g"))
        print("\n")

    # 4) Plot median DVHs
    _, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    median_control_dvh.plot(ax=ax, color="C0", label="Control", show_band=True, band_color="C0")
    median_ae_dvh.plot(ax=ax, color="C1", label="AE", show_band=True, band_color="C1")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)

    # 5) Print median Dx% values
    for percentage in [20, 50, 80]:
        dose_at_volume = analyzer.get_quantity_at_volume(histo=median_control_dvh, volumes=percentage)
        print(f"Median control D{percentage}%:\n{dose_at_volume: .2f} Gy(RBE)")
        dose_at_volume = analyzer.get_quantity_at_volume(histo=median_ae_dvh, volumes=percentage)
        print(f"AE control D{percentage}%:\n{dose_at_volume: .2f} Gy(RBE)")
    plt.show()

    # # 6) Mean DVHs
    # # Set uniform dose binning for aggregation
    # dose_edges = np.arange(0., 70.1, 0.1)  # D in [0, 70] with step 0.1 Gy
    # _, mean_control_dvh = analyzer.aggregate(dlvhs=control_dlvhs,
    #                                          stat="mean",
    #                                          quantity="dvh",
    #                                          aggregateby="dose",
    #                                          dose_edges=dose_edges)
    # _, mean_ae_dvh = analyzer.aggregate(dlvhs=ae_dlvhs,
    #                                     stat="mean",
    #                                     quantity="dvh",
    #                                     aggregateby="dose",
    #                                     dose_edges=dose_edges)
    
    # # 7) Plot mean DVHs
    # _, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    # mean_control_dvh.plot(ax=ax, color="C2", label="Control", show_band=True, band_color="C2")
    # mean_ae_dvh.plot(ax=ax, color="C3", label="AE", show_band=True, band_color="C3")
    # ax.legend(loc="best", frameon=False)
    # ax.grid(alpha=0.2)
    # plt.show()

if __name__ == "__main__":
    main()
