from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Literal, Optional, Union
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

from pydlvh.core import Histogram1D, Histogram2D, DLVH
from .utils import suggest_common_edges, suggest_common_edges_2d

""" 
    Methods to analyze DVH/LVH/DLVH cohorts. The methods either extract the statistics
    of histogram cohorts or compare control and adverse event (ae) groups to investigate 
    possible statistical difference.
"""

def dose_at_volume(histogram: Histogram1D, 
                   volume_grid: np.ndarray) -> np.ndarray:

    """ Get dose at specified volume levels (Dx%s) from a DVH histogram. """

    doses = 0.5 * (histogram.edges[:-1] + histogram.edges[1:])  # bin centers
    volumes = histogram.values

    if volumes[0] < volumes[-1]:
        volumes = volumes[::-1]
        doses = doses[::-1]

    # Interpolate: volume -> dose
    return np.interp(volume_grid, volumes[::-1], doses[::-1]) 
        
def validate(histograms: Union[Histogram1D, Histogram2D, List[Union[Histogram1D, Histogram2D]]],
             validate_edges: bool = True):

    """ Check that all the histograms belong to the same cohort. """

    if isinstance(histograms, list):
    
        reference_histogram = histograms[0]
        reference_type = type(reference_histogram)

        for h in histograms[1:]:
            if type(h) != reference_type:
                raise ValueError("All histograms must be of the same type (1D or 2D).")
            
            if isinstance(reference_histogram, Histogram2D) and validate_edges:
                if not (np.array_equal(reference_histogram.dose_edges, h.dose_edges) and
                        np.array_equal(reference_histogram.let_edges, h.let_edges)):
                    raise ValueError("All 2D histograms must have matching dose and LET edges.")
            
            elif isinstance(reference_histogram, Histogram1D):
                # Validate either dose/let edges (x sampling) or volume edges (y sampling)
                if validate_edges:
                    if not np.array_equal(reference_histogram.edges, h.edges) or not np.array_equal(reference_histogram.values, h.values):
                        raise ValueError("All 1D histograms must have matching edges either on the dose/let or the volumes axis.")
                    
                # Validate quantity type
                if reference_histogram.quantity != h.quantity:
                    raise ValueError("All 1D histograms must contain the same quantity type ('dvh' or 'lvh').")
                
                # If aggregated, validate aggregation modality
                if hasattr(reference_histogram, 'aggregatedby') and hasattr(h, 'aggregatedby'):
                    if reference_histogram.aggregatedby != h.aggregatedby:
                        raise ValueError("If aggregated, all 1D histograms must be aggregated by the same modality ('dose', 'let' or 'volume').")

def build_statistics_matrix(control_histograms, ae_histograms,
                            fill_value: float = 1.0,
                            test: str = "Mann-Whitney",
                            volume_grid: np.ndarray = np.linspace(0, 100, 101)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Check what type of histogram was provided
    reference_histogram = control_histograms[0]
    histo_type = type(reference_histogram) 
    is_aggregated = hasattr(reference_histogram, 'aggregatedby') and reference_histogram.aggregatedby is not None

    if histo_type == Histogram1D: # Histogram1D
        if is_aggregated:
            if reference_histogram.aggregatedby == "volume":
                # Invert histograms and stack
                control_group = np.stack([dose_at_volume(histo, volume_grid=volume_grid) for histo in control_histograms])
                ae_group = np.stack([dose_at_volume(histo, volume_grid=volume_grid) for histo in ae_histograms])
            elif reference_histogram.aggregatedby in ["dose", "let"]:
                # Just stack histograms
                control_group = np.stack([histo.values for histo in control_histograms])
                ae_group = np.stack([histo.values for histo in ae_histograms])
            else:
                raise ValueError(f"For Histogram1D {test} test, histograms must be aggregated by 'volume', 'dose' or 'let'.")
        
        else:
            # Default option: assuming aggregation by volume
            # Invert histograms
            control_group = np.stack([dose_at_volume(histo, volume_grid=volume_grid) for histo in control_histograms])
            ae_group = np.stack([dose_at_volume(histo, volume_grid=volume_grid) for histo in ae_histograms])

        shape = (control_group.shape[1])
        original_stats_matrix = np.full(shape, fill_value, dtype=float)
    
    else: # Histogram2D
        # Stack histogram values 
        control_group = np.stack([histo.values for histo in control_histograms])
        ae_group = np.stack([histo.values for histo in ae_histograms])
        shape = (control_group.shape[1], control_group.shape[2])
        original_stats_matrix = np.full(shape, fill_value, dtype=float)

    return control_group, ae_group, original_stats_matrix, shape

def aggregate(
        dlvhs: Union[DLVH, List[DLVH]],
        quantity: Optional[Literal["dvh", "lvh", "dlvh"]] = None,
        aggregateby: Optional[Literal["volume", "dose", "let"]] = None,
        stat: Literal["mean", "median"] = "median",
        dose_edges: Optional[np.ndarray] = None,
        let_edges: Optional[np.ndarray] = None,
        volume_edges: Optional[np.ndarray] = None,
        normalize: bool = True,
        cumulative: bool = True,
        dose_units: str = "Gy(RBE)",
        let_units: str = "keV/µm"
    ) -> Union[Histogram1D, Histogram2D]:

    """ Aggregate DVH/LVH/DLVH across a cohort. """

    # Set default quantity
    if not quantity: 
        quantity = "dlvh"

    aggregate_histo_1D = True if (quantity == "dvh" or quantity == "lvh") else False
    aggregate_histo_2D = True if (quantity == "dlvh") else False

    # Check quantity and aggregation modality coherence 
    if aggregateby not in ["dose", "let", "volume", None]: 
        raise ValueError(f"Unsupported aggregateby. Choose 'dose', 'let' or 'volume'.")
    if aggregate_histo_1D:
        if not aggregateby: aggregateby = "volume" # Automatically choose aggregation by volume if not specified
        # Automatically correct possible assignment errors? 
        if quantity == "dvh" and aggregateby == "let": aggregateby = "dose"
        if quantity == "lvh" and aggregateby == "dose": aggregateby = "let"
    # if aggregate_histo_2D: only aggregation by (d,l) is meaningful, any aggregateby assigment can be ignored

    rebinned_histos, bin_edges = get_all_cohort_histograms(dlvhs=dlvhs, 
                                                           dose_edges=dose_edges, 
                                                           let_edges=let_edges, 
                                                           volume_edges=volume_edges,
                                                           quantity=quantity, 
                                                           aggregateby=aggregateby,
                                                           cumulative=cumulative, 
                                                           normalize=normalize)

    rebinned_values = [h.values for h in rebinned_histos]
    stack = np.stack(rebinned_values, axis=0) 

    if stat == "mean":
        aggregate = np.mean(stack, axis=0)
        error = np.std(stack, axis=0, ddof=1)
        lower_percentile = higher_percentile = None
    elif stat == "median":
        aggregate = np.median(stack, axis=0)
        lower_percentile = np.percentile(stack, q=25, axis=0)
        higher_percentile = np.percentile(stack, q=75, axis=0)
        error = None
    else:
        raise ValueError("Unsupported stat. Choose 'mean' or 'median'.")
    
    dose_label = f"Dose [{dose_units}]"
    let_label = r"LET$_{d}$ " + f"[{let_units}]"

    if not isinstance(dlvhs, list): print("\nWatch out! You have only passed one dlvh to be processed.n") 
    if not quantity: print("Default quantity was not provided but inferred to be 'dlvh'.")
    if aggregate_histo_1D:
        label = dose_label if quantity == "dvh" else let_label
        return Histogram1D(values=aggregate, edges=bin_edges,
                           quantity=quantity, normalize=normalize,
                           cumulative=cumulative, x_label=label,
                           err=error, p_lo=lower_percentile, p_hi=higher_percentile, stat=stat, aggregatedby=aggregateby)
    elif aggregate_histo_2D: 
        return Histogram2D(values=aggregate,
                           dose_edges=bin_edges[0], let_edges=bin_edges[1],
                           normalize=normalize, cumulative=cumulative,
                           dose_label=dose_label, let_label=let_label,
                           err=error, p_lo=lower_percentile, p_hi=higher_percentile, stat=stat)
    else:
        return None

def aggregate_marginals(
        dlvhs: Union[DLVH, List[DLVH]],
        quantity: Literal["dose", "let"] = "median",
        stat: Literal["mean", "median"] = "median",
        normalize: bool = True,
        cumulative: bool = True,
        units: str = None
    ) -> Histogram1D:

    """
        Aggregate DVH or LVH derived from the 2D DLVH aggregate.
    """

    # If not already, aggregate input
    if not isinstance(dlvhs, list) and dlvhs.aggregated:
        aggregated_histo = dlvhs
    else: 
        aggregated_histo = aggregate(dlvhs=dlvhs, stat=stat, normalize=normalize, cumulative=cumulative)

    edges, values = aggregated_histo.get_marginals(quantity=quantity)
    if not units: units = "Gy[RBE]" if quantity == "dose" else "keV/µm"
    label = f"{quantity} [{units}]"
    return Histogram1D(values=values, edges=edges,
                       quantity=quantity, normalize=normalize,
                       cumulative=cumulative, x_label=label,
                       err=aggregated_histo.err[:, 0] if stat == "mean" and aggregated_histo.err is not None and quantity == "dose" else None,
                       p_lo=aggregated_histo.p_lo[:, 0] if stat == "median" and aggregated_histo.p_lo is not None and quantity == "dose" else None,
                       p_hi=aggregated_histo.p_hi[:, 0] if stat == "median" and aggregated_histo.p_hi is not None and quantity == "dose" else None)

def get_all_cohort_histograms(
        dlvhs: Union[DLVH, List[DLVH]],
        dose_edges: Optional[np.ndarray] = None,
        let_edges: Optional[np.ndarray] = None,
        volume_edges: Optional[np.ndarray] = None,
        quantity: Literal["dhv", "lvh", "dlvh"] = None,
        aggregateby: Optional[Literal["volume", "dose", "let"]] = None, # TODO: set this as mandatory?
        cumulative: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:

    """ Return an array containing all the DVH/LVH/DLVHs from cohort after standardizing binning. """

    # Set default quantity
    if not quantity: 
        quantity = "dlvh"

    # Check what type of histogram was provided
    aggregate_histo_1D = True if (quantity == "dvh" or quantity == "lvh") else False
    aggregate_histo_2D = True if (quantity == "dlvh") else False

    # Create common bin edges if not declared
    if aggregate_histo_1D: 
        # if bin edges are provided
        all_edges = [dose_edges, let_edges]
        if any(x is not None for x in all_edges):
            provided_edges = [e for e in all_edges if e is not None]
            # If only one type of edges are provided, use them
            if len(provided_edges) == 1: bin_edges = provided_edges[0]
            else: bin_edges = provided_edges[0] # When too many are assigned, random assignment between dose/let

        # else automatic binning
        else:
            arrays = [dlvh.dose if aggregateby == "dose" else dlvh.let for dlvh in dlvhs]
            bin_edges = suggest_common_edges(arrays=arrays)

    elif aggregate_histo_2D:
        if dose_edges is None or let_edges is None:
            if dose_edges is not None: # Only dose provided
                print("Only dose binning has been specified for aggregated dlvh. Let binning will be atuomatically set.")
                lets = [dlvh.let for dlvh in dlvhs] 
                let_edges = suggest_common_edges(arrays=lets)

            elif let_edges is not None: # Only let provided
                print("Only let binning has been specified for aggregated dlvh. Dose binning will be atuomatically set.")
                doses = [dlvh.dose for dlvh in dlvhs] 
                dose_edges = suggest_common_edges(arrays=doses)

            else: # None specified
                doses = [dlvh.dose for dlvh in dlvhs]
                lets = [dlvh.let for dlvh in dlvhs] 
                dose_edges, let_edges = suggest_common_edges_2d(dose_arrays=doses, let_arrays=lets)
        bin_edges = [dose_edges, let_edges]

    # Rebin histos
    rebinned_histos = []
    for dlvh in dlvhs:
        if aggregate_histo_1D:
            if quantity == "dvh":
                h = dlvh.dose_volume_histogram(bin_edges=bin_edges, normalize=normalize, 
                                               cumulative=cumulative, aggregatedby=aggregateby)
                # If aggregating by volume, recompute bin edges to match output histogram
                if aggregateby == "volume":
                    if volume_edges is None: volume_edges = np.linspace(1, 99, 100) # Dx% with x in [1, 99]
                    recomputed_edges = dose_at_volume(h, volume_grid=volume_edges)
                    h = dlvh.dose_volume_histogram(bin_edges=recomputed_edges[::-1], normalize=normalize, 
                                                   cumulative=cumulative, aggregatedby=aggregateby)
                    print("histo values: ", h.values)

            else:
                h = dlvh.let_volume_histogram(bin_edges=bin_edges, normalize=normalize, 
                                              cumulative=cumulative, aggregatedby=aggregateby)
            rebinned_histos.append(h)
        else: 
            h2d = dlvh.dose_let_volume_histogram(dose_edges=dose_edges,
                                                 let_edges=let_edges,
                                                 normalize=normalize,
                                                 cumulative=cumulative)
            rebinned_histos.append(h2d)

    return rebinned_histos, bin_edges
    
def voxel_wise_Mann_Whitney_test(control_histograms: List[Union[Histogram1D, Histogram2D]],
                                 ae_histograms: List[Union[Histogram1D, Histogram2D]], 
                                 volume_grid: np.ndarray = np.linspace(0, 100, 101), # Volume grid for Mann-Whitney testing on Histogram1D (testing Dx%)
                                 alpha: float = 0.05,
                                 correction: Optional[Literal["holm", "fdr_bh"]] = None) -> Tuple[np.ndarray, np.ndarray]:
    
    """ Perform voxel-wise Mann-Whitney U test between control and ae groups. """

    validate([*control_histograms, *ae_histograms])

    # Check what type of histogram was provided
    reference_histogram = control_histograms[0]
    histo_type = type(reference_histogram) 
    control_group, ae_group, original_p_values, shape = build_statistics_matrix(control_histograms,
                                                                                ae_histograms, 
                                                                                fill_value=1.0,
                                                                                test="Mann-Whitney",
                                                                                volume_grid=volume_grid)

    # Perform Mann-Whitney u test
    for idx in np.ndindex(shape):
        if histo_type == Histogram1D:
            control = control_group[:, idx[0]]
            ae = ae_group[:, idx[0]]
        else: # Histogram2D
            control = control_group[:, idx[0], idx[1]]
            ae = ae_group[:, idx[0], idx[1]]
        _, p = mannwhitneyu(control, ae, alternative="two-sided")
        original_p_values[idx] = p

    original_p_values = original_p_values.flatten()

    # Apply test correction
    if correction:
        reject, p_values, _, _ = multipletests(original_p_values, alpha=alpha, method=correction)
    else:
        reject = original_p_values < alpha
        p_values = original_p_values

    p_values = p_values.reshape(shape)
    significance = reject.reshape(shape)

    return p_values, significance

def get_auc_score(control_histograms: Union[List[Histogram1D], List[Histogram2D]],
                  ae_histograms: Union[List[Histogram1D], List[Histogram2D]],
                  volume_grid: np.ndarray = np.linspace(0, 100, 101))  -> np.ndarray:
    
    validate([*control_histograms, *ae_histograms])

    # Check what type of histogram was provided
    reference_histogram = control_histograms[0]
    histo_type = type(reference_histogram) 
    control_group, ae_group, auc_map, shape = build_statistics_matrix(control_histograms,
                                                                      ae_histograms, 
                                                                      fill_value=0.5,
                                                                      test="AUC score",
                                                                      volume_grid=volume_grid)

    for idx in np.ndindex(shape):

        control_value = control_group[:, idx[0], idx[1]]
        ae_value = ae_group[:, idx[0], idx[1]]

        y_true = np.array([0]*len(ae_value) + [1]*len(control_value))
        y_scores = np.concatenate([ae_value, control_value])

        # Only compute AUC if there is variation in data
        if np.unique(y_scores).size > 1:
            auc_map[idx] = roc_auc_score(y_true, y_scores)
        else:
            auc_map[idx] = 0.5  # no discrimination
    
    return auc_map