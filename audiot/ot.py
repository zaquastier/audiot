import numpy as np
import ot
from scipy.sparse import csr_matrix

from .dist import *

def interpolate_frequency(f_s, f_t, alpha, method='int', support=None):
    """
        Return interpolated frequency index adapted to a desired support.

        Args:
            f_s (int): Source frequency index.
            f_t (int): Target frequency index.
            alpha (double): Interpolation parameter (between 0 and 1).
            method (string): Method to generate interpolated frequency index.
            (for some methods) support (np.ndarray): Support to generate interpolated frequency index.
        
            Returns:
                (int) interpolated frequency index
    """
    if method == 'int':
        return int((1 - alpha) * f_s + alpha * f_t)
    if method == 'round':
        return round((1 - alpha) * f_s + alpha * f_t)
    if method == 'closest':
        interpolated_frequency = (1 - alpha) * support[f_s] + alpha * support[f_t]
        return (np.abs(support - interpolated_frequency)).argmin()

def emd(support, source, target, cost_matrix, alpha, method='int'):
    """
    Returns interpolant and OT matrix between two normalized spectra.
    By default, uses euclidean cost matrix on the frequency support.

    Args:
        support (np.ndarray): Frequency support (in Hz).
        source (np.ndarray): Source spectrum.
        target (np.ndarray): Tource spectrum.
        cost_matrix (np.ndarray): Cost matrix.
        alpha (double): Interpolation parameter (between 0 and 1).
        method (string): Method to generate interpolated support. 

    Returns:
        emd_interpolation (np.ndarray): Interpolated spectrum.
        emd_plan (np.ndarray): Optimal transport matrix.
    """

    # solve +inf values in cost_matrix
    # max_finite_value = np.nanmax(cost_matrix[np.isfinite(cost_matrix)])
    # TODO: do this when creating matrix
    max_finite_value = np.nanmax(cost_matrix)
    
    cost_matrix[np.isinf(cost_matrix)] = max_finite_value
    cost_matrix[np.isnan(cost_matrix)] = max_finite_value

    emd_plan = ot.lp.emd(source, target, cost_matrix)
    # emd_plan[np.isnan(emd_plan)] = 0 # ?
    emd_plan = csr_matrix(emd_plan)
    emd_interpolation = np.zeros(len(support))
    row, col = emd_plan.nonzero()

    for f_s, f_t in zip(row, col):
        index = interpolate_frequency(f_s, f_t, alpha=alpha, method=method, support=support if method =='closest' else None)
        emd_interpolation[index] += emd_plan[f_s, f_t]

    emd_plan = np.array(emd_plan.todense())
    
    return emd_interpolation, emd_plan

def cost_matrix(source_support, target_support=None, dist=l2):
    """
    Generate cost matrix.

    Args:
        source_support (np.ndarray): Frequency source support (in Hz).
        (optional) target_support (np.ndarray): Frequency target support. If None, we suppose source and target have same supports
        dist (function): Distance function.

    Returns:
        M (np.ndarray): Cost matrix.
    """

    if target_support is None:
        target_support = source_support

    source_samples = len(source_support)
    target_samples = len(target_support)

    M = np.zeros((source_samples, target_samples))

    for i in range(source_samples):
        for j in range(target_samples):
            M[i, j] = dist(source_support[i], target_support[j])

    return M

import matplotlib.gridspec as gridspec