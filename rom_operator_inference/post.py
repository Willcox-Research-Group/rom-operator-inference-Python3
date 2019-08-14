# post.py
"""Tools for accuracy and error evaluation."""

import numpy as np


# Some error evaluators -------------------------------------------------------
def mape(truth, pred):
    """Mean absolute prediction error."""
    mask = truth > 1e-12
    return (((np.abs(truth-pred)/truth)[mask]).sum()+pred[~mask].sum())/len(truth)


def rpe(truth, pred):
    """Relative prediction error."""
    mask = np.abs(truth) > 1e-10
    rpe_mat = np.zeros_like(mask)
    rpe_mat[mask] = (np.abs(truth-pred)/np.abs(truth))[mask]
    rpe_mat[~mask] = np.abs(pred[~mask])
    return rpe_mat
