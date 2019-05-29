import numpy as np
from numba import jit
from sklearn.utils.extmath import randomized_svd
import os
import h5py
import sys
import matplotlib.pyplot as plt

def compute_randomized_svd(data,savepath,n_components = 500):
	'''
	Compute and save randomized svd following 
	https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html

	--------------------
	INPUT
	--------------------

	data: array [n_variables, n_samples]
		data to take svd of 

	savepath: string
	 	filepath where you want to save svd to 

	n_components: float
		number of components of svd to save 

	--------------------
	OUTPUT
	--------------------
	None - computes and saves left singular vectors and singular values

	'''


	U,Sigma,_ = randomized_svd(data, n_components, n_iter = 15,random_state = 42)

	hfsvd = h5py.File(savepath,'w')
	hfsvd.create_dataset("U", data = U)
	hfsvd.create_dataset("S", data = Sigma)
	hfsvd.close()

	return U,Sigma
	print("SVD computed - saved in %s" %savepath)

@jit(nopython = True)
def compute_xdot(data,dt):
	'''
	This function computes xdot for a chunk of snapshots
	--------------------
	INPUT
	--------------------

	data - The dataset 
		   ndarray (Nfull x n_samples), Nfull = numvariable*38523
    dt  - the timestep size
    	  float

	--------------------
	OUTPUT
	--------------------

	xdot - time derivative of data using 4th order 5 point stecil approximation
		   https://en.wikipedia.org/wiki/Five-point_stencil
		   ndarray (Nfull x n_samples - 4), no derivative for first 2 and last 2 timesteps

	'''

	xdot = (data[:,0:-4] - 8*data[:,1:-3] + 8*data[:,3:-1] - data[:,4:])/(12*dt)
	return xdot 

def project_data(data,projected_data_folder, projected_xdot_folder,U,dt,r_vals = 0):

	'''
 	This function computes xdot and projects both data and xdot onto the POD basis.

 	Projected data is saved as "projected_data_folder/data_reduced_%d.h5"
 	Projected xdot data is saved as "projected_xdot_folder/xdot_reduced_%d.h5"

	--------------------
	INPUT
	--------------------

	data: array [n_variables, n_samples] 
			data to be projected and used to compute, project xdot

	projected_data_folder: string
			folder to save projected data

	projected_xdot_folder: string
			folder to save projected xdot data

	U: array [n_variables, r]
			the POD basis

	dt: float
			timestep size between data (for computing xdot)

	r_vals: list (defaults to number of columns in U)
			basis sizes to compute projection for <= U.shape[1]

	--------------------
	OUTPUT
	--------------------

	None - this function saves the data in the above filepaths 
	'''

	assert data.shape[1] > 1, "Only passed one snapshot - must pass two for xdot computation"

	os.makedirs(os.path.dirname(projected_data_folder), exist_ok=True)
	os.makedirs(os.path.dirname(projected_xdot_folder), exist_ok=True)

	if r_vals == 0:
		r_vals = [U.shape[1]]

	#full dimension
	N = data.shape[0]

	#number of timesteps
	numtimesteps = data.shape[1]

	remainder = numtimesteps%1000 

	chunck = np.math.floor(numtimesteps/1000) #for efficient memory chunking

	# if more than 1000 snapshots are passed, the projection is done 1000 snapshots at a time. 
	# the remainder tells me if it 

	for r in r_vals:
		print(" r = ", r)

		data_reduced = np.zeros((r,numtimesteps))
		xdot_reduced = np.zeros((r,numtimesteps))

		#forward difference for first 2 timesteps and project 
		xdot_reduced[:,0] = ((U[:,:r].T) @ (((data[:,1]) - (data[:,0]))/dt))
		xdot_reduced[:,1] = ((U[:,:r].T) @ (((data[:,2]) - (data[:,1]))/dt))

		# project first two snapshots
		data_reduced[:,:2] = (U[:,:r].T) @ data[:,:2]


		if (numtimesteps > 2) :
			if numtimesteps == 3:
				#low order derivative
				xdot_reduced[:,2] = ((U[:,:r].T) @ (((data[:,2]) - (data[:,1]))/dt))
				data_reduced[:,2] = (U[:,:r].T) @ data[:,2]

			elif numtimesteps == 4:
				#low order derivate for last two
				xdot_reduced[:,2] = ((U[:,:r].T) @ (((data[:,2]) - (data[:,1]))/dt))
				data_reduced[:,2] = (U[:,:r].T) @ data[:,2]

				xdot_reduced[:,3] = ((U[:,:r].T) @ (((data[:,3]) - (data[:,2]))/dt))
				data_reduced[:,3] = (U[:,:r].T) @ data[:,3]

			else:
				if chunck == 0:
					# less than 1000
					xdot_reduced[:,2:-2] = ((U[:,:r].T) @ compute_xdot(data,dt))
					data_reduced[:,:] = (U[:,:r].T) @ data

					# backward difference for last two
					xdot_reduced[:,-2] = ((U[:,:r].T) @ (((data[:,-2]) - (data[:,-3]))/dt))
					xdot_reduced[:,-1] = ((U[:,:r].T) @ (((data[:,-1]) - (data[:,-2]))/dt))

				else:
					# more than 1000
					if remainder != 0:
						#not a multiple of 1000

						#first 1000 - excluding the first 2
						current_snap = data[:,:1002]
						data_reduced[:,2:1000] = (U[:,:r].T) @ current_snap[:,2:-2]
						xdot_reduced[:,2:1000] = (U[:,:r].T) @ compute_xdot(current_snap,dt)

						for ii in range(1,chunck):
							current_snap = data[:,(ii*1000)-2 : (ii+1)*1000 + 2] #grab 1004 snapshots because xdot chops off first 2 and last 2
							data_reduced[:,ii*1000:(ii+1)*1000] = (U[:,:r].T) @ current_snap[:,2:-2]
							xdot_reduced[:,ii*1000:(ii+1)*1000] = (U[:,:r].T) @ compute_xdot(current_snap,dt)

						if remainder == 1:
							#just one off (1001,2001,...)
							xdot_reduced[:,-1] = ((U[:,:r].T) @ (((data[:,-1]) - (data[:,-2]))/dt))
							data_reduced[:,-1] = (U[:,:r].T) @ data[:,-1]

						elif remainder == 2:
							# two off (1002,2002,...)
							xdot_reduced[:,-2] = ((U[:,:r].T) @ (((data[:,-2]) - (data[:,-3]))/dt))
							xdot_reduced[:,-1] = ((U[:,:r].T) @ (((data[:,-1]) - (data[:,-2]))/dt))
							data_reduced[:,-2] = (U[:,:r].T) @ data[:,-2]
							data_reduced[:,-1] = (U[:,:r].T) @ data[:,-1]

						else:
							# more than 2 we can use higher order xdot
							xdot_reduced[:,chunck*1000:-2] = (U[:,:r].T) @ compute_xdot(data[:,1000*chunck-2:],dt)
							# and project data
							data_reduced[:,chunck*1000:] = (U[:,:r].T) @ data[:,1000*chunck:]

							#low order for last two 
							xdot_reduced[:,-2] = ((U[:,:r].T) @ (((data[:,-2]) - (data[:,-3]))/dt))
							xdot_reduced[:,-1] = ((U[:,:r].T) @ (((data[:,-1]) - (data[:,-2]))/dt))



					else:
						#a multiple of 1000 snapshots

						#first 1000 - excluding the first 2
						current_snap = data[:,:1002]
						data_reduced[:,2:1000] = (U[:,:r].T) @ current_snap[:,2:-2]
						xdot_reduced[:,2:1000] = (U[:,:r].T) @ compute_xdot(current_snap,dt)

						#compute high order derivatives 1000 at a time
						for ii in range(1,chunck-1):
							current_snap = data[:,(ii*1000)-2 : (ii+1)*1000 + 2] #grab 1004 snapshots because xdot chops off first 2 and last 2
							data_reduced[:,ii*1000:(ii+1)*1000] = (U[:,:r].T) @ current_snap[:,2:-2]
							xdot_reduced[:,ii*1000:(ii+1)*1000] = (U[:,:r].T) @ compute_xdot(current_snap,dt)

						#deal with last 1000
						current_snap = data[:,numtimesteps-1000-2:numtimesteps]
						data_reduced[:,-1000:numtimesteps] = (U[:,:r].T) @ current_snap[:,-1000:]

						# high order xdot for last 998
						xdot_reduced[:,numtimesteps-1000:-2] = (U[:,:r].T) @ compute_xdot(current_snap,dt)
						
						#first order xdot for last two
						xdot_reduced[:,-2] = ((U[:,:r].T) @ (((data[:,-2]) - (data[:,-3]))/dt))
						xdot_reduced[:,-1] = ((U[:,:r].T) @ (((data[:,-1]) - (data[:,-2]))/dt))

	

		print("Saving..")
		fname1 = "%sdata_reduced_%d.h5" %(projected_data_folder,r)
		savexf = h5py.File(fname1,'w')
		savexf.create_dataset('data',data = data_reduced)
		savexf.close()

		fname2 = "%sxdot_reduced_%d.h5" %(projected_xdot_folder,r)
		savexdotf = h5py.File(fname2,'w')
		savexdotf.create_dataset('xdot',data = xdot_reduced)
		savexdotf.close()

#some error stuff
def mape(truth,pred):
	#mean absolute prediction error
	mask = (truth > 1e-12)
	return (((np.fabs(truth-pred)/truth)[mask]).sum()+pred[~mask].sum())/len(truth)

def rpe(truth,pred):

	#relative prediction error
	mask = (abs(truth) > 1e-10)
	rpe_mat = np.zeros(mask.shape)
	rpe_mat[mask] = (np.fabs(truth-pred)/abs(truth))[mask]
	rpe_mat[~mask] = abs(pred[~mask])
	return rpe_mat


def get_bool(prompt):
    while True:
        try:
           return {"true":True,"false":False}[input(prompt).lower()]
        except KeyError:
           print("Invalid input please enter True or False!")

def get_int(prompt):
    while True:
        try:
           return int(input(prompt))
        except ValueError:
           print("Thats not an integer silly!")
def main():
	r_vals = [17,29]


	trainsize = get_int("How many snapshots?")
	comp_svd = get_bool("Compute SVD? (True or False -- Type False if it exists in a file 'svd_nt%d.h5')")


	if comp_svd:
		truncation_value = get_int("Enter the truncation value for randomized SVD (if unsure type 500):")
		print("Computing svd...")
		fulldatapath = "OpInfdata/data_minmax.h5"
		hf = h5py.File(fulldatapath,'r')
		data = hf['data'][:,:]
		hf.close()
		U,_ = compute_randomized_svd(data[:,:trainsize],"OpInfdata/svd_nt%d.h5" %trainsize)

	else: 
		print("loading svd...")
		hfs = h5py.File("OpInfdata/svd_nt%d.h5" %trainsize, 'r')
		U = hfs['U'][:,:]

		fulldatapath = "OpInfdata/data_minmax.h5"
		hf = h5py.File(fulldatapath,'r')
		data = hf['data'][:,:]
		hf.close()

	print("Computing xdot and reducing data...")
	project_data(data[:,:trainsize], "OpInfdata/reducedData/data_reduced_minmax_nt%d/" %trainsize, "OpInfdata/reducedData/xdot_reduced_minmax_nt%d/" %trainsize, U, 1e-7,r_vals)


if __name__ == '__main__':
	main()






