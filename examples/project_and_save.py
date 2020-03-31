# project_and_save.py
"""An old data management script for computing a POD basis, projecting data,
computing velocities, and saving the projected data for later.
"""

import os
import sys
import h5py
import numpy as np
from sklearn.utils.extmath import randomized_svd

import rom_operator_inference as roi


def offdiag_penalizers(reg, r, m):
    """Construct a list of regularization matrices that penalize the
    off-diagonal elements of A and all elements of the remaining operators,
    where the model has the form dx / dt = c + Ax + H(x⊗x) + Bu.
    """
    regI = np.sqrt(reg) * np.eye(1 + r + r*(r+1)//2 + m)
    Gs = [regI] * r
    for i in range(1, r+1):
        Gs[i][i,i] = 0
    return Gs


def compute_randomized_svd(data, savepath, n_components=500):
    """Compute and save randomized SVD following
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html

    Parameters
    ----------
    data : (n_variables, n_samples) ndarray
        Snapshot matrix for which the SVD will be computed.

    savepath : str
        Name of file to save the SVD to.

    n_components : float
        Number of singular values and vectors to compute and save.
    """
    Vr,Sigma,_ = randomized_svd(data, n_components, n_iter=15, random_state=42)

    with h5py.File(savepath, 'w') as svd_h5file:
        svd_h5file.create_dataset("U", data=Vr)
        svd_h5file.create_dataset("S", data=Sigma)

    print(f"SVD computed and saved in {savepath}")
    return Vr,Sigma


def project_data(data,
                 projected_data_folder,
                 projected_xdot_folder,
                 Vr,
                 dt,
                 r_vals=-1):
    """
    Compute xdot and project both data and xdot onto the POD basis.

    Projected data is saved as "<projected_data_folder>/data_reduced_%d.h5"
    Projected xdot data is saved as "<projected_xdot_folder>/xdot_reduced_%d.h5"

    Parameters
    ----------
    data : (n_variables, n_samples) ndarray
        Data to be projected and used to compute, project xdot.

    projected_data_folder : str
        Folder to save projected data to.

    projected_xdot_folder : str
        Folder to save projected xdot data to.

    Vr: (n_variables, r) ndarray
        The POD basis.

    dt: float
        Timestep size between data (for computing xdot)

    r_vals: list of ints <= Vr.shape[1] (the number of columns in Vr).
        Basis sizes to compute projection for.
    """

    if data.shape[1] < 2:
        raise ValueError("At least two snapshots required for x' computation")

    os.makedirs(os.path.dirname(projected_data_folder), exist_ok=True)
    os.makedirs(os.path.dirname(projected_xdot_folder), exist_ok=True)

    if r_vals == -1:
        r_vals = [Vr.shape[1]]

    for r in r_vals:
        print(f"r = {r}", flush=True)
        # select first r basis vectors
        VrT = Vr[:,:r].T

        # project snapshots
        print("\tProjecting snapshots...", end='', flush=True)
        data_reduced = VrT @ data
        print("done.")

        # project velocities
        print("\tComputing projected velocities...", end='', flush=True)
        if data.shape[1] <= 4:  # Too few time steps for 4th order derivatives
            xdot_reduced = np.gradient(data_reduced, dt, edge_order=2, axis=-1)
        else:
            xdot_reduced = roi.pre.xdot_uniform(data_reduced, dt, order=4)
        print("done.")

        # save results
        print("\tSaving files...", end='', flush=True)
        fname1 = os.path.join(projected_data_folder, f"data_reduced_{r:d}.h5")
        with open h5py.File(fname1,'w') as savexf:
            savexf.create_dataset('data', data=data_reduced)

        fname2 = os.path.join(projected_xdot_folder, f"xdot_reduced_{r:d}.h5")
        with open h5py.File(fname2,'w') as savexdotf:
            savexdotf.create_dataset('xdot', data=xdot_reduced)
        print("done.")


# Some custom error evaluators ------------------------------------------------
def mape(truth, pred):
    """Mean absolute prediction error."""
    mask = truth > 1e-12
    return (((np.abs(truth-pred)/truth)[mask]).sum()+pred[~mask].sum())/len(truth)


def rpe(truth,pred):
    """Relative prediction error."""
    mask = abs(truth) > 1e-10
    rpe_mat = np.zeros(mask.shape)
    rpe_mat[mask] = (np.abs(truth-pred)/abs(truth))[mask]
    rpe_mat[~mask] = abs(pred[~mask])
    return rpe_mat


if __name__ == '__main__':

    # make command line arguments for trainsize and whether to compute svd
    trainsize = int(input("How many snapshots?"))
    svdfile = os.path.join("data", f"svd_nt{trainsize}.h5")
    compute_svd = input(f"Compute SVD? (True or False -- Type False if it exists as '{svdfile}')")

    fulldatapath = "data/data_renee.h5"
    with h5py.File(fulldatapath,'r') as hf:
        data = hf['data'][:,:]

    if compute_svd != "False":
        print("Computing svd...", end='')
        V,_ = compute_randomized_svd(data[:,:trainsize], svdfile)
        print("done.")

    else:
        print("Loading svd...", end='')
        with h5py.File(svdfile, 'r') as hfs:
            V = hfs['Und'][:,:]
        print("done.")

    print("Computing xdot and reducing data...")
    target_folder = os.path.join("data", f"data_reduced_minmax_nt{trainsize}")
    project_data(data[:,:trainsize],
                 target_folder,
                 target_folder,
                 V,
                 1e-7,
                 [5,10,15])
