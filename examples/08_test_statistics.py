import numpy as np
import matplotlib.pyplot as plt
from pydlvh import DLVH, DLVHCohort, analyzer

def create_synthetic_patient(n_voxels=4000,
                             dose_max=60.0,
                             mu_dose=30.0, sigma_dose=7.0,
                             mu_let=3.0, sigma_let=1.0,
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

    # 1) Create synthetic patients
    dose_shapes = [(50, 0.8), (48, 1.7), (48, 2.5), (52, 2.2), (54, 1.5)]
    control_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Control group
    control_cohort = DLVHCohort(control_dlvhs)
    dose_shapes = [(27, 0.5), (26, 2.0), (30, 1.3), (29, 2.3), (28, 1.4)]
    ae_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] #Adverse event (AE) group
    ae_cohort = DLVHCohort(ae_dlvhs)

    # 2) Median DVHs (± iqr)
    # Ensure that both control and ae aggregates are computed
    # with the same  binning.
    dose_edges = np.arange(0, 65, 0.5)
    let_edges = np.arange(0, 8, 0.05)
    median_control_dvh = control_cohort.aggregate_1d(
        quantity="dose",
        stat="median",
        normalize=True,
        cumulative=True,
        bin_edges=dose_edges
    )
    median_ae_dvh = ae_cohort.aggregate_1d(
        quantity="dose",
        stat="median",
        normalize=True,
        cumulative=True,
        bin_edges=dose_edges
    )
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
    plt.show()


"""
    # 5) Median DLVH  (+ P25/P75)
    h2d_med = cohort.aggregate_2d(
        stat="median",
        normalize=True,
        cumulative=True
    )
    # Median
    h2d_med.plot(cmap="plasma", mode="values", isovolumes=[5, 10, 20])
    plt.title("Median DLVH 2D")
    plt.tight_layout()
    plt.show()

    # 25th percentile
    h2d_med.plot(cmap="viridis", mode="p_lo")
    plt.title("DLVH 25th percentile (IQR lower)")
    plt.tight_layout()
    plt.show()

    # 75th percentile
    h2d_med.plot(cmap="viridis", mode="p_hi")
    plt.title("DLVH 75th percentile (IQR upper)")
    plt.tight_layout()
    plt.show()

    # 6) Check median DVH/LVH from DLVH margins 
    dvh_from2d = cohort.aggregate_marginals(kind="dose", stat="median",
                                            normalize=True, cumulative=True)
    ax = dvh_from2d.plot(color="C3", label="extracted DVH", show_band=True)
    ax.legend(loc="best")
    ax.set_title("DVH from median DLVH (median + IQR)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""

if __name__ == "__main__":
    main()
