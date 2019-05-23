'''
This file contains functions to helper do operator inference 
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import csr_matrix
import chemistry_conversions as chem
from sklearn.linear_model import SGDRegressor,LinearRegression,Ridge
from scipy.optimize import lsq_linear


def normal_equations(D,r,k,num):
	'''
	Solves the normal equations corresponding to the least squares problem
	min ||Do - r|| + k||Fo||
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	INPUT
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	D - (nd array) Data matrix K x n+1+s
	r - (nd array) X dot data reduced K x 1
	k - (float) regularization parameter 
	num - (int) number of ls problem we are solving [1..r]
	offdiagonals - (bool) minimize only off diags of A (TRUE) or minimize all values of A,F,c (FALSE)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	OUTPUT
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	o - the least squares solution
	'''
	K,rps = D.shape

	F = np.eye(rps)
	F[num,num] = 0

	pseudo = k*F
	rhs = np.zeros((rps))

	Dplus = np.vstack((D,pseudo))
	Rplus = np.vstack((r.reshape((-1,1)),rhs.reshape((-1,1))))

	return np.linalg.lstsq(Dplus,Rplus,rcond=None)[0]


def get_x_sq(X):
	N,w = X.shape
	S = w*(w+1)/2
	Xi = np.zeros((N,int(S)))
	c = 0
	for i in range(w):
		Xi[:,c:c+(w-i)] = np.multiply(np.tile(X[:,i].reshape(N,1),[1,w-i]),X[:,i:w])
		c = c+(w-i)
	return Xi

def F2H(F):
	n,_ = F.shape

	jj,ii = np.nonzero(F.T)

	FT = F.T
	vv = np.ravel(FT[jj,ii])
	jj = jj+1
	ii = ii+1
	iH = []
	jH = []
	vH = []
	# print(type(iH))
	bb = 0
	for i in range(1,n+1):
		cc = bb+n+1-i
		# print(jj)
		sel = (jj>bb) & (jj <=cc)
		# print(sel)
		itemp = ii[sel]
		jtemp = jj[sel]
		vtemp = vv[sel]
		for j in range(1,len(jtemp)+1):
			sj = jtemp[j-1] - bb  
			if sj == 1 :
				iH.append([itemp[j-1]])
				jH.append([(i-1)*n+i+(sj)-1])
				vH.append([vtemp[j-1]])
			else:
				iH.append([itemp[j-1]])
				iH.append([itemp[j-1]])
				jH.append([(i-1)*n+(i)+(sj)-1])
				jH.append([(i+(sj)-2)*n+i])
				vH.append([vtemp[j-1]/2])
				vH.append([vtemp[j-1]/2])
		bb = cc
	# print(np.array(vH).shape)
	# print(np.array(iH).shape)
	# print(np.array(jH).shape)
	vHa = np.ndarray.flatten(np.array(vH))

	iHa = np.ndarray.flatten(np.array(iH))
	iHa = iHa-1
	jHa = np.ndarray.flatten(np.array(jH))
	jHa = jHa-1
	# print(vH)
	# print(vHa)
	H = csr_matrix((vHa,(iHa,jHa)),shape=(n,n**2)).toarray()
	return H
