# projection_helpers.py

import os
import sys
import h5py
import numba
import numpy as np
from sklearn.utils.extmath import randomized_svd


def offdiag_penalizer(reg, r, m):
    """Construct a list of regularization matrices that penalize the
    off-diagonal elements of A and all elements of the remaining operators.
    """
    regI = _np.sqrt(reg) * _np.eye(r + r**2 + m + 1)
    Gs = [regI] * r
    for i in range(r):
        Gs[r][r,r] = 0
    return Gs


def compute_randomized_svd(data, savepath, n_components=500):
    """Compute and save randomized SVD following
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html

    Parameters
    ----------
    data : (n_variables, n_samples) ndarray
        Snapshot matrix for which the SVD will be computed.

    savepath : string
        Name of file to save the SVD to.

    n_components : float
        Number of singular values and vectors to compute and save.
    """


    Vr,Sigma,_ = randomized_svd(data, n_components, n_iter=15, random_state=42)

    svd_h5file = h5py.File(savepath,'w')
    svd_h5file.create_dataset("U", data=Vr)
    svd_h5file.create_dataset("S", data=Sigma)
    svd_h5file.close()

    return Vr,Sigma
    print(f"SVD computed and saved in {savepath}")


@numba.jit(nopython=True)
def compute_xdot(data, dt):
    """Compute xdot for a chunk of snapshots.

    Parameters
    ----------
    data : (Nfull, n_samples) ndarray
        The dataset. Here Nfull = numvariable*38523.
    dt : float
        The timestep size

    Returns
    -------
    xdot : (Nfull, n_samples-4) ndarray
        Time derivative of data using 4th order 5 point stecil approximation.
        See https://en.wikipedia.org/wiki/Five-point_stencil. No derivative is
        computed for first 2 and last 2 timesteps due to the stencil.
    """

    xdot = (data[:,0:-4] - 8*data[:,1:-3] + 8*data[:,3:-1] - data[:,4:])/(12*dt)
    return xdot


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

    projected_data_folder : string
        Folder to save projected data to.

    projected_xdot_folder : string
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

    # full dimension
    N = data.shape[0]

    # number of timesteps
    numtimesteps = data.shape[1]

    remainder = numtimesteps%1000

    chunck = numtimesteps // 1000 # for efficient memory chunking

    # if more than 1000 snapshots are passed, the projection is done 1000 snapshots at a time.
    # the remainder tells me if it

    for r in r_vals:
        print(" r = ", r)
        VrT = Vr[:,:r].T

        data_reduced = np.zeros((r,numtimesteps))
        xdot_reduced = np.zeros((r,numtimesteps))

        # forward difference for first 2 timesteps and project
        xdot_reduced[:,0] = VrT @ ((data[:,1] - data[:,0])/dt)
        xdot_reduced[:,1] = VrT @ ((data[:,2] - data[:,1])/dt)

        # project first two snapshots
        data_reduced[:,:2] = VrT @ data[:,:2]


        if (numtimesteps > 2):
            if numtimesteps == 3:
                # low order derivative
                xdot_reduced[:,2] = VrT @ ((data[:,2] - data[:,1])/dt)
                data_reduced[:,2] = VrT @ data[:,2]

            elif numtimesteps == 4:
                # low order derivate for last two
                xdot_reduced[:,2] = VrT @ ((data[:,2] - data[:,1])/dt)
                data_reduced[:,2] = VrT @ data[:,2]

                xdot_reduced[:,3] = VrT @ ((data[:,3] - data[:,2])/dt)
                data_reduced[:,3] = VrT @ data[:,3]

            else:
                if chunck == 0:
                    # less than 1000
                    print("Less than 1000")
                    xdot_reduced[:,2:-2] = VrT @ compute_xdot(data,dt)
                    data_reduced[:,:] = VrT @ data

                    # backward difference for last two
                    xdot_reduced[:,-2] = VrT @ ((data[:,-2] - data[:,-3])/dt)
                    xdot_reduced[:,-1] = VrT @ ((data[:,-1] - data[:,-2])/dt)

                else:
                    # more than 1000
                    if remainder != 0:
                        print("Not a multiple of 1000")
                        # not a multiple of 1000

                        # first 1000 - excluding the first 2
                        current_snap = data[:,:1002]
                        data_reduced[:,2:1000] = VrT @ current_snap[:,2:-2]
                        xdot_reduced[:,2:1000] = VrT @ compute_xdot(current_snap,dt)

                        for ii in range(1,chunck):
                            current_snap = data[:,(ii*1000)-2 : (ii+1)*1000 + 2] # grab 1004 snapshots because xdot chops off first 2 and last 2
                            data_reduced[:,ii*1000:(ii+1)*1000] = VrT @ current_snap[:,2:-2]
                            xdot_reduced[:,ii*1000:(ii+1)*1000] = VrT @ compute_xdot(current_snap,dt)

                        if remainder == 1:
                            # just one off (1001,2001,...)
                            xdot_reduced[:,-1] = VrT @ ((data[:,-1] - data[:,-2])/dt)
                            data_reduced[:,-1] = VrT @ data[:,-1]

                        elif remainder == 2:
                            # two off (1002,2002,...)
                            xdot_reduced[:,-2] = VrT @ ((data[:,-2] - data[:,-3])/dt)
                            xdot_reduced[:,-1] = VrT @ ((data[:,-1] - data[:,-2])/dt)
                            data_reduced[:,-2] = VrT @ data[:,-2]
                            data_reduced[:,-1] = VrT @ data[:,-1]

                        else:
                            # more than 2 we can use higher order xdot
                            xdot_reduced[:,chunck*1000:-2] = VrT @ compute_xdot(data[:,1000*chunck-2:],dt)
                            # and project data
                            data_reduced[:,chunck*1000:] = VrT @ data[:,1000*chunck:]

                            # low order for last two
                            xdot_reduced[:,-2] = VrT @ ((data[:,-2] - data[:,-3])/dt)
                            xdot_reduced[:,-1] = VrT @ ((data[:,-1] - data[:,-2])/dt)



                    else:
                        print(" A multiple of 1000")
                        # a multiple of 1000 snapshots

                        # first 1000 - excluding the first 2
                        current_snap = data[:,:1002]
                        data_reduced[:,2:1000] = VrT @ current_snap[:,2:-2]
                        xdot_reduced[:,2:1000] = VrT @ compute_xdot(current_snap,dt)

                        # compute high order derivatives 1000 at a time
                        for ii in range(1,chunck-1):
                            current_snap = data[:,(ii*1000)-2 : (ii+1)*1000 + 2] # grab 1004 snapshots because xdot chops off first 2 and last 2
                            data_reduced[:,ii*1000:(ii+1)*1000] = VrT @ current_snap[:,2:-2]
                            xdot_reduced[:,ii*1000:(ii+1)*1000] = VrT @ compute_xdot(current_snap,dt)

                        # deal with last 1000
                        current_snap = data[:,numtimesteps-1000-2:numtimesteps]
                        data_reduced[:,-1000:numtimesteps] = VrT @ current_snap[:,-1000:]

                        # high order xdot for last 998
                        xdot_reduced[:,numtimesteps-1000:-2] = VrT @ compute_xdot(current_snap,dt)

                        # first order xdot for last two
                        xdot_reduced[:,-2] = VrT @ ((data[:,-2] - data[:,-3])/dt)
                        xdot_reduced[:,-1] = VrT @ ((data[:,-1] - data[:,-2])/dt)



        print("Saving..")
        fname1 = "%sdata_reduced_%d.h5" %(projected_data_folder,r)
        savexf = h5py.File(fname1,'w')
        savexf.create_dataset('data',data = data_reduced)
        savexf.close()

        fname2 = "%sxdot_reduced_%d.h5" %(projected_xdot_folder,r)
        savexdotf = h5py.File(fname2,'w')
        savexdotf.create_dataset('xdot',data = xdot_reduced)
        savexdotf.close()


# Some error evaluators -------------------------------------------------------
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
    trainsize = input("How many snapshots?")
    trainsize = int(trainsize)
    comp_svd = input("Compute SVD? (True or False -- Type False if it exists in a file 'data/svd_nt%d')")
    # trainsize = 10000

    fulldatapath = "data/data_minmax.h5"
    hf = h5py.File(fulldatapath,'r')
    data = hf['data'][:,:]
    hf.close()
    if comp_svd:
        print("Computing svd...")
        V,_ = compute_randomized_svd(data[:,:trainsize],"data/svd_nt%d.h5" %trainsize)

    else:
        print("loading svd...")
        hfs = h5py.File("data/svd_nt%d.h5" %trainsize, 'r')
        V = hfs['Und'][:,:]

    print("Computing xdot and reducing data...")
    project_data(data[:,:trainsize],
                 f"data/data_reduced_minmax_nt{trainsize}/",
                 f"data/data_reduced_minmax_nt{trainsize}/",
                 V,
                 1e-7,
                 [5,10,15])
