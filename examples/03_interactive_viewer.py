"""
03_interactive_viewer.py
========================
Interactive viewer example:
- Left panel: DVH (cumulative, fixed)
- Right panel: LVH (cumulative, updates when dose threshold changes)
- Bottom slider: adjust dose threshold [GyE]
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from pydlvh.core import DLVH
from pydlvh.viewer import DLVHViewer


def main():
    # Synthetic dose and let distributions (Gaussian)
    n_voxels = 30000
    volume_cc = 12.0
    dose = np.random.normal(loc=45.0, scale=8.0, size=n_voxels)
    dose = np.clip(dose, 0, None)
    let = np.random.normal(loc=2.5, scale=0.7, size=n_voxels)
    let = np.clip(let, 0, None)

    dlvh = DLVH(dose=dose, let=let, volume_cc=volume_cc)

    # 1) Create interactive viewer for cumulative dvh
    viewer = DLVHViewer(dlvh)
    viewer.plot1D()


if __name__ == "__main__":
    main()
