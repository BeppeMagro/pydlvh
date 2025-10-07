from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Literal, Optional, Union
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

from pydlvh.core import Histogram1D, Histogram2D


class Analyzer:
    """ 
        Methods to analyze control and adverse event (ae) groups and investigate possible statistical difference.
    """
            
    def validate(histograms: Union[Histogram1D, Histogram2D, List[Union[Histogram1D, Histogram2D]]]):

        if not isinstance(histograms, list):
            histograms = [histograms]
        
        reference_histogram = histograms[0]
        reference_type = type(reference_histogram)

        for h in histograms[1:]:
            if type(h) != reference_type:
                raise ValueError("All histograms must be of the same type (1D or 2D).")
            
            if isinstance(reference_histogram, Histogram2D):
                if not (np.array_equal(reference_histogram.dose_edges, h.dose_edges) and
                        np.array_equal(reference_histogram.let_edges, h.let_edges)):
                    raise ValueError("All 2D histograms must have matching dose and LET edges.")
            
            elif isinstance(reference_histogram, Histogram1D):
                if not np.array_equal(reference_histogram.edges, h.edges):
                    raise ValueError("All 1D histograms must have matching edges.")
        

    def perform_voxel_wise_Mann_Whitney_test(control_group: Union[Histogram1D, Histogram2D],
                                             ae_group: Union[Histogram1D, Histogram2D], 
                                             alpha: float = 0.05,
                                             correction: Optional[Literal["holm", "fdr_bh"]] = None) -> Tuple[np.ndarray, np.ndarray]:

        """ Perform voxel-wise Mann-Whitney U test between control and ae groups. """

        Analyzer.validate([control_group, ae_group])
        if control_group.stat != "median" or ae_group.stat != "median":
            raise AttributeError("Control and ae histograms must have been computed as median quantities for the Mann-Whitney u-test.")

        shape = np.shape(control_group)
        original_p_values = np.zeros(shape)

        for idx in np.ndindex(shape):
            control = control_group.values[idx]
            ae = ae_group.values[idx]
            _, p = mannwhitneyu(control, ae, alternative="two-sided")
            original_p_values[idx] = p

        original_p_values = original_p_values.flatten()
        
        if correction:
            reject, p_values, _, _ = multipletests(original_p_values, alpha=alpha, method='holm')
            reject, p_values, _, _ = multipletests(original_p_values, alpha=alpha, method='fdr_bh')
        else:
            reject = original_p_values < alpha
            p_values = original_p_values

        p_values = p_values.reshape(shape)
        significance = reject.reshape(shape)

        return p_values, significance


    def get_auc_score(control_histograms: Union[List[Histogram1D], List[Histogram2D]],
                      ae_histograms: Union[List[Histogram1D], List[Histogram2D]])  -> np.ndarray:
        
        Analyzer.validate([*control_histograms, *ae_histograms])

        # Stack histogram values 
        control_group = np.stack([histo.values for histo in control_histograms])
        ae_group = np.stack([histo.values for histo in ae_histograms])
        
        shape = (control_group.shape[1], control_group.shape[2])
        auc_map = np.full(shape, 0.5)

        for idx in np.ndindex(shape):

            control_value = control_group[:, idx]
            ae_value = ae_group[:, idx]

            y_true = np.array([0]*len(control_value) + [1]*len(ae_value))
            y_scores = np.concatenate([control_value, ae_value])

            # Only compute AUC if there is variation in data
            if np.unique(y_scores).size > 1:
                auc_map[idx] = roc_auc_score(y_true, y_scores)
            else:
                auc_map[idx] = 0.5  # no discrimination
        
        return auc_map