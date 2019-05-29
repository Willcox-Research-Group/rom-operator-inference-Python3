import numpy as np
from scipy.sparse import csr_matrix
from operator_inference import opinf_helper
from operator_inference import integration_helpers


class model:

	def __init__(self, degree = 'LQ', inp = True):

		'''
		Initalize operator inference model
			\dot{x} = f(x) + Bu(t)

		--------------------
		Parameters
		--------------------

		Degree: {'L','Lc','LQ','LQc','Q','Qc'}, optional, default 'LQ'
				'L' 
					a linear model f(x) = Ax
				'Lc' 
					a linear model with a constant f(x) = Ax + c
				'LQ' 
					a linear quadratic model f(x) = Ax + Fx^2
				'LQc'
					a linear quadratic model with a constant f(x) = Ax + Fx^2 + c
				'Q' 
					a strictly quadratic model f(x) = Fx^2
				'Qc' 
					a strictly quadratic model with a constant f(x) = Fx^2 + c
 
		Inp: bool, optional, default = True
				True 
					assumes the system has an additive input term u(t) != 0
				False 
					assumes the system does not have an input u(t) = 0
		''' 
		self.degree = degree
		self.input = inp

	def fit(self,r,reg,Xdot,X,U = None):
		'''
		Solves for the operators

		--------------------
		Parameters
		--------------------	
		r: int 
			The basis size (number of basis vectors)

		reg: float
			L2 regularization penalty. Can be imposed if this is non-zero
			solves min ||Do - r||_2 + k || reg ||_2
		
		Xdot: array
			The velocity training data in an array of size [n_variables, n_samples] 
		
		X: array 
			The snapshot training data in an array of size [n_variables, n_samples]

		U: array, optional, default = None
			The input training data in an array of size [n_parameters, n_samples] (only if input == True)

		--------------------
		Output
		--------------------
		None

		'''
		K = X.shape[1] #number of timesteps
		p = 0
		s = int(r*(r+1)/2)
		self.r = r


		# Linear Quadratic'''
		if self.degree == 'LQ':
			X2 = opinf_helper.get_x_sq(X.T)
			D = np.hstack((X.T,X2))
			oshape = r + s

		#Linear Quadratic with a constant
		elif self.degree == 'LQc':
			X2 = opinf_helper.get_x_sq(X.T)
			D = np.hstack((X.T,X2, np.ones((K,1))))	
			p += 1
			oshape = r + s

		# '''Linear'''
		elif self.degree == 'L':
			D = X.T
			s = 0
			oshape = r

		# ''' Linear with a constant'''
		elif self.degree == 'Lc':
			D = np.hstack((X.T,np.ones((K,1))))
			p += 1
			oshape = r 

		# ''' Strictly Quadratic'''
		elif self.degree == 'Q':
			D = opinf_helper.get_x_sq(X.T)
			oshape = s
		
		# ''' Strictly Quadratic with a constant '''
		elif self.degree == 'Qc':
			D = np.hstack((opinf_helper.get_x_sq(X.T), np.ones((K,1))))
			p += 1
			oshape = s

		else:
			print("Error - incorrect degree type. Options are {'L','Lc','LQ','LQc','Q','Qc'}. Please redefine your model")
			exit()


		if self.input:

			U = np.atleast_2d(U)
			D = np.hstack((D,U.T))
			p += U.shape[0]
			


		''' Solve for the operators!'''
		O = np.zeros((oshape+p,r))

		#Solve for the operators!
		for it in range(r):
			O[:,it] = np.ravel(opinf_helper.normal_equations(D,Xdot[it,:],reg,it))
		O = O.T
		''''''''
		self.residual = np.linalg.norm(D@O.T - Xdot.T,2)**2
		self.solution = np.linalg.norm(O.T,2)**2

		# ''' Linear Quadratic'''
		if self.degree == 'LQ':
			if self.input:
				# A H B
				self.F = O[:,r:r+s]
				self.A, self.H, self.B = O[:,:r],opinf_helper.F2H(O[:,r:r+s]),O[:,r+s:r+s+p]
			else:
				# A H
				self.A, self.H= O[:,:r],opinf_helper.F2H(O[:,r:r+s])
				self.B = np.zeros((r,1))

		# ''' Linear Quadratic with a constant'''
		elif self.degree == 'LQc':
			if self.input:
				# A H c B
				self.A, self.H, self.c, self.B = O[:,:r],opinf_helper.F2H(O[:,r:r+s]),O[:,r:r+1],O[:,r+s+1:r+s+p+1]

			else:
				# A H c
				self.B = np.zeros((r,1))
				self.A, self.H, self.c = O[:,:r],opinf_helper.F2H(O[:,r:r+s]),O[:,r:r+1],

		# '''Linear'''
		elif self.degree == 'L':
			if self.input:
				# A B
				self.A, self.B = O[:,:r],O[:,r:r+p]
			else:
				# A 
				self.A = O[:,:r]
				self.B = np.zeros((r,1))

		# ''' Linear with a constant'''
		elif self.degree == 'Lc':
			if self.input:
				# A c B
				self.A, self.c, self.B = O[:,:r],O[:,r:r+1],O[:,r+1:r+1+p]
			else:
				# A c
				self.A, self.c = O[:,:r],O[:,r:r+1]
				self.B = np.zeros((r,1)) 
		# ''' Strictly Quadratic'''
		elif self.degree == 'Q':
			if self.input:
				# H B
				self.H, self.B = opinf_helper.F2H(O[:,:s]),O[:,s:s+p]
			else:
				# H
				self.H = opinf_helper.F2H(O[:,:s])
				self.B = np.zeros((r,1))

		# ''' Strictly Quadratic with a constant '''
		elif self.degree == 'Qc':
			if self.input:
				# H c B
				self.H, self.c, self.B = opinf_helper.F2H(O[:,:s]),O[:,s:s+1],O[:,s+1:s+1+p]
			else:
				# H c
				self.H, self.c = opinf_helper.F2H(O[:,:s]),O[:,s:s+1]
				self.B = np.zeros((r,1))

	def predict(self, init, n_timesteps, dt, u = None):

		'''
		Simulate the learned model

		--------------------
		Parameters
		--------------------	
		init: array
			the initial state vector to begin simulation in an array of size [n_variables, 1]

		n_timesteps: int
			number of time steps to simulate
		
		dt: float 
			time step size 
		
		u: array, optional, default = None
			The input for each time step in an array of size [n_parameters, n_timesteps]

		--------------------
		Output
		--------------------
		None

		'''
		assert init.shape[0] == self.r, "initial state size is invalid, must be [%d, 1]" %r
		
		u = np.atleast_2d(u)
		
		if u.any() != None:
			assert u.shape[1] == n_timesteps, "Input shape is not equal to number of timesteps"
		else:
			u = np.zeros((1,n_timesteps))


		projected_state = np.zeros((self.r,n_timesteps))
		projected_state[:,0] = init

		# Linear Quadratic'''
		if self.degree == 'LQ':
			for i in range(1,n_timesteps):
				projected_state[:,i] = integration_helpers.rk4advance_LQ(projected_state[:,i-1],dt,self.A,self.H,self.B,u[:,i])
				if np.any(np.isnan(projected_state[:,i])):
					print("NaNs enountered at step ", i)
					break
			return projected_state,i

		#Linear Quadratic with a constant
		elif self.degree == 'LQc':
			for i in range(1,n_timesteps):
				projected_state[:,i] = integration_helpers.rk4advance_LQc(projected_state[:,i-1],dt,self.A,self.H,self.c[:,0],self.B,u[:,i])
				if np.any(np.isnan(projected_state[:,i])):
					print("NaNs enountered at step ", i)
					break
			return projected_state,i
		# '''Linear'''
		elif self.degree == 'L':
			for i in range(1,n_timesteps):
				projected_state[:,i] = integration_helpers.rk4advance_L(projected_state[:,i-1],dt,self.A,self.B,u[:,i])
				if np.any(np.isnan(projected_state[:,i])):
					print("NaNs enountered at step ", i)
					break
			return projected_state,i

		# ''' Linear with a constant'''
		elif self.degree == 'Lc':
			for i in range(1,n_timesteps):
				projected_state[:,i] = integration_helpers.rk4advance_Lc(projected_state[:,i-1],dt,self.A,self.c[:,0],self.B,u[:,i])
				if np.any(np.isnan(projected_state[:,i])):
					print("NaNs enountered at step ", i)
					break
			return projected_state,i

		# ''' Strictly Quadratic'''
		elif self.degree == 'Q':
			for i in range(1,n_timesteps):
				projected_state[:,i] = integration_helpers.rk4advance_Q(projected_state[:,i-1],dt,self.H,self.B,u[:,i])
				if np.any(np.isnan(projected_state[:,i])):
					print("NaNs enountered at step ", i)
					break
			return projected_state,i
		
		# ''' Strictly Quadratic with a constant '''
		elif self.degree == 'Qc':
			for i in range(1,n_timesteps):
				projected_state[:,i] = integration_helpers.rk4advance_Qc(projected_state[:,i-1],dt,self.H,self.c[:,0],self.B,u[:,i])
				if np.any(np.isnan(projected_state[:,i])):
					print("NaNs enountered at step ", i)
					break
			return projected_state,i


	def get_residual_norm(self):
		return self.residual, self.solution
	def get_operators(self):

		'''
		Return the operators of the fitted model. 

		--------------------
		Parameters
		--------------------
		None

		--------------------
		Output
		--------------------
		(ops,):  a tuple containing each operator (nd array) as defined by degree of the model

		'''

		ops = ()

		if self.degree == 'L':
			ops += (self.A,)
		elif self.degree == 'Lc':
			ops += (self.A,self.c)
		elif self.degree == 'LQ':
			ops += (self.A, self.H)
		elif self.degree == 'LQc':
			ops += (self.A, self.H, self.c)
		elif self.degree == 'Q':
			ops += (self.H)
		else:
			ops += (self.H,self.c)

		if self.input:
			ops += (self.B,)

		return ops
