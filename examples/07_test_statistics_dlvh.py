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
    dose_shapes = [(31, 1.8), (30, 1.7), (33, 2.5), (32, 2.2), (33, 0.5), (33, 1.8), (31, 1.7), (34, 2.5),
                   (33, 2.2), (34, 0.5), (33, 0.5), (32, 1.8), (31, 1.7), (31, 2.5), (33, 2.2), (35, 0.5),
                   (29, 1.4), (36, 1.8), (34, 1.7), (35, 2.5), (34, 2.2), (32, 0.5), (28, 1.3), (31, 2.3)]
    control_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Control group
    dose_shapes = [(25, 0.5), (27, 2.0), (30, 1.3), (29, 2.3), (28, 1.4), (29, 1.8), (30, 1.7), (29, 2.5),
                   (31, 2.2), (32, 0.5), (27, 1.4), (32, 1.8), (31, 1.7), (30, 2.5), (30, 2.2), (32, 0.5),
                   (28, 1.4), (26, 1.8), (27, 1.7), (29, 2.5), (28, 2.2), (30, 0.5), (31, 1.3), (28, 2.3)]
    ae_dlvhs = [create_synthetic_patient(mu_dose=mu, sigma_dose=sd) for (mu, sd) in dose_shapes] # Adverse event (AE) group

    # 2) Median DLVHs
    all_control_dlvhs, median_control_dlvh = analyzer.aggregate(dlvhs=control_dlvhs,
                                                                stat="median",
                                                                quantity="dlvh")
    all_ae_dlvhs, median_ae_dlvh = analyzer.aggregate(dlvhs=ae_dlvhs,
                                                      stat="median",
                                                      quantity="dlvh")

    # 3) Compute statistical significance between control and AE DLVHs (Mann-Whitney u-test).
    alpha = 0.05
    pvalues, significance = analyzer.voxel_wise_Mann_Whitney_test(control_histograms=all_control_dlvhs, 
                                                                  ae_histograms=all_ae_dlvhs,
                                                                  alpha=alpha,
                                                                  correction="fdr_bh")
    
    # TODO: auc score

    # 4) Plot median DLVHs
    _, ax = plt.subplots(1, 2, figsize=(9, 6.5))
    median_control_dlvh.plot(ax=ax[0])
    median_ae_dlvh.plot(ax=ax[1])
    plt.show()

if __name__ == "__main__":
    main()
