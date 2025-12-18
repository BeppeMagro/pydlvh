"""
07b_test_statistics_dlvh.py
============================
Compare two DLVH cohorts applying different statistical
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
    mu_dose_control, sigma_dose_control = 60.0, 5.0
    dose_shapes = [(x, np.abs(y)) for x, y in zip(np.random.normal(loc=mu_dose_control, scale=3.0, size=10), np.random.normal(loc=sigma_dose_control, scale=1.0, size=100))]
    control_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Control group
    mu_dose_ae, sigma_dose_ae = 50.0, 5.0
    dose_shapes = [(x, np.abs(y)) for x, y in zip(np.random.normal(loc=mu_dose_ae, scale=3.0, size=10), np.random.normal(loc=sigma_dose_ae, scale=1.0, size=100))]
    ae_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Adverse event (AE) group

    # 2) Median DLVHs
    # Set manual uniform dose+let binning for aggregation
    dose_edges = np.arange(0., 71, 1)  # D in [0, 70] with step 1 Gy
    let_edges = np.arange(0., 51, 1)  # LET in [0, 50] with step 1 keV/um
    print("\nAggregating control DLVHs...")
    all_control_dlvhs, median_control_dlvh = analyzer.aggregate(dlvhs=control_dlvhs,
                                                                stat="median",
                                                                quantity="dlvh",
                                                                dose_edges=dose_edges,
                                                                let_edges=let_edges)
    print("\nAggregating AE DLVHs...")
    all_ae_dlvhs, median_ae_dlvh = analyzer.aggregate(dlvhs=ae_dlvhs,
                                                      stat="median",
                                                      quantity="dlvh",
                                                      dose_edges=dose_edges,
                                                      let_edges=let_edges)

    # 3) Compute statistical significance between control and AE DLVHs (Mann-Whitney u-test)
    """alpha = 0.05
    print("\nComputing voxel-based Mann-Whitney test...")
    pvalues, significance = analyzer.voxel_wise_Mann_Whitney_test(control_histograms=all_control_dlvhs, 
                                                                  ae_histograms=all_ae_dlvhs,
                                                                  alpha=alpha,
                                                                  correction="fdr_bh")

    # Print (top 5 most) significant Dx%
    if np.any(significance):
        rows = []
        significant_indices = np.argwhere(significance)
        for i, j in significant_indices:
            rows.append({
                "Dose (Gy)": median_control_dlvh.dose_edges[i],
                "LET (keV/µm)": median_control_dlvh.let_edges[j],
                "Volume control": median_control_dlvh.values[i, j],
                "Volume AE": median_ae_dlvh.values[i, j],
                "p-value": pvalues[i, j],
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("p-value").head(5)
        print(f"\nTop 5 Mann–Whitney U-test most significant Dx% (α={alpha}, BH corrected):\n")
        print(df.to_markdown(index=False, floatfmt=".4g"))"""

    # 4) Plot median DLVHs
    _, ax = plt.subplots(1, 2, figsize=(9, 4))
    isovolumes_colors = [ "#21918c", "#DF0E6F", "#fde725"]
    median_control_dlvh.plot(ax=ax[0], isovolumes=[20, 50, 80], isovolumes_colors=isovolumes_colors)
    ax[0].set_title("Median Control DLVH")
    median_ae_dlvh.plot(ax=ax[1], isovolumes=[20, 50, 80], isovolumes_colors=isovolumes_colors)
    ax[1].set_title("Median AE DLVH")
    plt.tight_layout()
    plt.show()
    
    # 5) Compute AUC map between control and AE DLVHs    
    print("\nComputing ROC-AUC score...")
    auc_map = analyzer.get_auc_score(control_histograms=all_control_dlvhs,
                                     ae_histograms=all_ae_dlvhs)
    
    # 6) Plot AUC map and visualize signficant voxels
    _, ax = plt.subplots(figsize=(4, 4))
    auc_map.plot(ax=ax)
    plt.show()
    """fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for idx, auc_map in enumerate([auc_map_LEM, auc_map_MKM]):

        ax = axes[idx]
        mask70 = (auc_map > 0.7)
        mask72 = (auc_map > 0.719)

        # Show AUC map with real dose/LET axes
        cax = ax.imshow(
            auc_map.T,
            origin="lower",
            cmap=cmap,
            vmin=0.5, vmax=0.7,
            extent=[dose_edges[0], dose_edges[-1], let_edges[0], let_edges[-1]],
            aspect="auto"
        )

        # Overlay contour of significant voxels
        dose_centers = 0.5 * (dose_edges[:-1] + dose_edges[1:])
        let_centers = 0.5 * (let_edges[:-1] + let_edges[1:])
        ax.contour(
            dose_centers, let_centers, mask70.T,
            levels=[0.5], colors="darkblue", linewidths=2
        )
        ax.contour(
            dose_centers, let_centers, mask72.T,
            levels=[0.5], colors="darkred", linewidths=2
        )
        true_indices = np.argwhere(mask72)
        print(f"\n--- Points where AUC>0.7 (dose, LET) - {models[idx]} ---")
        print("(Dose, LETd, Volume(median control), Volume(median AE), AUC)")
        for (i, j) in true_indices:
            x = dose_centers[i]
            y = let_centers[j]
            z = medianControlDLVHs[idx].values[i, j]
            z_ae = medianAEDLVHs[idx].values[i, j]
            auc = auc_map[i, j]
            print(f"({x:.2f}, {y:.2f}, {z:.2f}, {z_ae:.2f}, {auc:.5f})")
            # print(f"({x}, {y}, {z:.2f}, {auc:.2f})")

        # Add colorbar
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("AUC")

        # Labels and title
        ax.set_xlabel("Dose [GyRBE]")
        ax.set_ylabel(r"LET$_d$ [keV/µm]")
        ax.set_title("Voxel-wise AUC (Control vs AE)")
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 80)

    plt.tight_layout()
    plt.show()"""

if __name__ == "__main__":
    main()
