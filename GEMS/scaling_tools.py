'''Scaling functions '''

import numpy as np
from numba import jit
from sklearn.externals import joblib 



@jit()
def minmax_scaler_predefined_scale(data,varnum,mini=-1,maxi=1):
	scaling_params = np.load("OpInfdata/Scaling_parameters.npz")
	mins = scaling_params['mins']
	maxs = scaling_params['maxs']
	scales = scaling_params['scales']

	return scales[varnum]*data + mini - mins[varnum]*scales[varnum] 

@jit()
def minmax_scaler(data,mini,maxi,N,scale,varnum=0):
	'''
	scales data to the range [mini,maxi] where each column has variables stacked every N rows.
	for example variable 3 in column 2 is data[2*N:3*N,1]. So N/len(data[:,0]) should be an integer

	Also unscales back to original variables 

	scaling algorithm follows this: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
	
	...............................................
	INPUT
	...............................................

	data - The dataset to be scaled - as described above 
		   ndarray (Nfull x numsnaps)

    mini,maxi - the taget range for each variable
    			int,int

    N - the number of elements of each variable in each column (GEMS data N = 38523)
    	int
    
    scale - (1) scale to [mini,maxi] or (0) unscale from [mini,maxi] 
    			boolean
	...............................................
	OUTPUT
	...............................................	

	data - the scaled data - each variable is in interval [mini,maxi]

	'''

	#make sure the data is 2d even if its just a vector (for indexing below - doesnt change any values) 
	if len(data.shape) <2:
		data = np.reshape(data,(-1,1))

	# the total number of variables should be the length of each column divided by N
	numVars = 8 

	if scale:
		#We are scaling the data
		mins = np.zeros(numVars)
		maxs = np.zeros(numVars)
		scales = np.zeros(numVars)


		for kk in range(numVars):	
			mins[kk] = np.min(data[kk*N:(kk+1)*N,:])
			maxs[kk] = np.max(data[kk*N:(kk+1)*N,:])
			scales[kk] = (maxi-mini)/(maxs[kk] - mins[kk])
			data[kk*N:(kk+1)*N,:] = scales[kk]*data[kk*N:(kk+1)*N,:] + mini - mins[kk]*scales[kk] 

		np.savez("OpInfdata/Scaling_parameters", mins = mins, maxs = maxs, scales = scales)
		print('---------------------\n')
		print('Min -------- Max\n')
		print('---------------------\n')
		print("Pressure\n")
		print("%.3f ------ %.3f" %(mins[0],maxs[0]))
		print('---------------------\n')

		print("x - velocity\n")
		print("%.3f ------ %.3f" %(mins[1],maxs[1]))
		print('---------------------\n')
		print("y - velocity\n")
		print("%.3f ------ %.3f" %(mins[2],maxs[2]))
		print('---------------------\n')
		print("Specific Volume\n")
		print("%.3f ------ %.3f" %(mins[3],maxs[3]))
		print('---------------------\n')
		print("CH4 molar \n")
		print("%.3f ------ %.3f" %(mins[4],maxs[4]))
		print('---------------------\n')
		print("O2 molar\n")
		print("%.3f ------ %.3f" %(mins[5],maxs[5]))
		print('---------------------\n')
		print("CO2 molar\n")
		print("%.3f ------ %.3f" %(mins[6],maxs[6]))
		print('---------------------\n')
		print("H2O molar\n")
		print("%.3f ------ %.3f" %(mins[7],maxs[7]))
		print('---------------------\n')


	if not(scale):
		#We are unscaling 
		scaling_params = np.load("OpInfdata/Scaling_parameters.npz")
		mins = scaling_params['mins']
		maxs = scaling_params['maxs']
		scales = scaling_params['scales']
		# print("Mins = ", mins)
		# print("\nMaxs = ", maxs)
		# print("\nScales = ", scales)



		for jj in range(numVars):
			data[jj*N:(jj+1)*N,:] = (data[jj*N:(jj+1)*N,:] - mini + (mins[jj]*scales[jj]))/scales[jj]

	return data


