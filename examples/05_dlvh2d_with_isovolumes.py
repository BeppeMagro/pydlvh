"""
05_dlvh2d_with_isovolumes.py
============================
Example usage of the 2D Dose–LET Volume Histogram (DLVH)
with isovolume contour lines.

- Generate synthetic Gaussian dose and LET values.
- Build a cumulative DLVH (V(D ≥ d, LET ≥ l)).
- Plot the 2D histogram with iso-volume lines at 5%, 10%, 20%.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from pydlvh.core import DLVH
from pydlvh.viewer import DLVHViewer


def main():
    # --- synthetic data ---
    n_voxels = 30000
    volume_cc = 12.0

    dose = np.random.normal(loc=50.0, scale=10.0, size=n_voxels)  # Gy
    dose = np.clip(dose, 0, None)

    let = np.random.normal(loc=2.5, scale=0.6, size=n_voxels)     # keV/µm
    let = np.clip(let, 0, None)

    # --- create DLVH object ---
    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # --- launch viewer for 2D cumulative DLVH with isovolumes ---
    viewer = DLVHViewer(dlvh)
    viewer.plot2D(
        bin_width_dose=1.0,
        bin_width_let=0.1,
        normalize=True,
        cumulative=True,
        isovolumes=[5, 10, 20],  # iso-lines as % of ROI volume
        cmap="plasma",
        interactive=True
    )


if __name__ == "__main__":
    main()
