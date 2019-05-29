from operator_inference import OpInf
from operator_inference import opinf_helper
import numpy as np
import matplotlib.pyplot as plt
import h5py


# load some heat data (1D snapshots)
hf = h5py.File('OpInfdata/heat_opinf_data.h5','r') 
train_snapshots = hf['train_snapshots'][:,:]
original_data = hf['snapshots'][:,:]
part_test_l, part_test_r, part_train_l, part_train_r = hf['particular_test_left'][:,:], hf['particular_test_right'][:,:], hf['particular_train_left'][:,:], hf['particular_train_right'][:,:]


# total number of snapshots
n_t = original_data.shape[1]

# length of each snapshot
d = original_data.shape[0]

# range of x 
x = np.linspace(0,4,d)

# time step size
dt = 0.01

# number of training snapshots
train_size = train_snapshots.shape[1]

# number of testing snapshots
test_size = n_t - train_size



# compute POD basis
[U,S,V] = np.linalg.svd(train_snapshots, full_matrices = False)
r=15
print("Dimension of POD basis", r)

# project data onto POD basis
xhat = (U[:,:r].T)@train_snapshots

# compute the reduced time derivative 
xdot = np.zeros(xhat.shape)
xdot[:,2:-2] = (xhat[:,0:-4] - 8*xhat[:,1:-3] + 8*xhat[:,3:-1] - xhat[:,4:])/(12*dt)
xdot[:,0] = ((xhat[:,1] - xhat[:,0])/dt).reshape((r,))
xdot[:,1] = ((xhat[:,2] - xhat[:,1])/dt).reshape((r,))
xdot[:,-1] = ((xhat[:,-1] - xhat[:,-2])/dt).reshape((r,))
xdot[:,-2] = ((xhat[:,-2] - xhat[:,-3])/dt).reshape((r,))
'''
------------------------------------------------------
Using the operator_inference module
------------------------------------------------------
'''
#define the model 
mymodel = OpInf.model('Lc',False) # a linear quadratic with no input

#fit the model
mymodel.fit(r,0,xdot,xhat) #0 is the L2 regularization param

#simulate the model for train and test time steps
xr,break_point = mymodel.predict(np.ravel(xhat[:,0]), n_t, dt)

#reconstruct the predictions
xr_rec = U[:,:r]@xr

#add the particular back
xr_rec[:,train_size:] = xr_rec[:,train_size:] + part_test_l + part_test_r
xr_rec[:,:train_size] = xr_rec[:,:train_size] + part_train_l+ part_train_r


'''
------------------------------------------------------
Plot results
------------------------------------------------------
'''

plt.rcParams.update({'font.size': 30}) 


alphs = np.linspace(1,0.5,int((n_t)))
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=-3, vmax=3))
time = np.linspace(0,2*np.math.pi,n_t)
plt.figure(5)
for jj in range(d):
    plt.scatter( x[jj]*np.ones(len(time)),time, c = xr_rec[jj,:],s=5,cmap = plt.cm.jet,vmin=-3,vmax = 3 )
plt.plot(x,np.ones(d)*np.math.pi, color = 'k',linewidth = 2)
plt.xlabel("Spatial Location",fontsize = 35)
plt.ylabel("Time", fontsize = 35)
plt.yticks([0,np.math.pi/3,2*np.math.pi/3,np.math.pi,4*np.math.pi/3,5*np.math.pi/3,2*np.math.pi],['0',r'$\frac{\pi}{3}$',r'$\frac{2\pi}{3}$',r'$\pi$',r'$\frac{4\pi}{3}$',r'$\frac{5\pi}{3}$', r'$2\pi$'],fontsize = 30)
plt.xticks([0,2,4],['0',r'$\frac{L}{2}$','L'],fontsize = 30)
plt.title("Predicted")
sm._A = []
plt.colorbar(sm,label = "Temperature")
plt.subplots_adjust(left=0.12, bottom = .15, right = 0.95)



plt.figure(6)
for jj in range(d):
    plt.scatter( x[jj]*np.ones(len(time)),time, c = original_data[jj,:],s=5,cmap = plt.cm.jet,vmin=-3,vmax = 3 )
plt.plot(x,np.ones(d)*np.math.pi, color = 'k',linewidth = 2)

plt.xlabel("Spatial Location",fontsize = 35)
plt.ylabel("Time", fontsize = 35)
plt.yticks([0,np.math.pi/3,2*np.math.pi/3,np.math.pi,4*np.math.pi/3,5*np.math.pi/3,2*np.math.pi],['0',r'$\frac{\pi}{3}$',r'$\frac{2\pi}{3}$',r'$\pi$',r'$\frac{4\pi}{3}$',r'$\frac{5\pi}{3}$', r'$2\pi$'],fontsize = 30)
plt.xticks([0,2,4],['0',r'$\frac{L}{2}$','L'],fontsize = 30)
plt.title("True")
sm._A = []
plt.colorbar(sm,label = "Temperature")
plt.subplots_adjust(left=0.12, bottom = .15, right = 0.95)



plt.figure(7)
for jj in range(d):
    plt.scatter( x[jj]*np.ones(len(time)),time, c = abs(xr_rec[jj,:] - original_data[jj,:]),s=5,cmap = plt.cm.jet,vmin=0,vmax = .15 )
plt.plot(x,np.ones(d)*np.math.pi, color = 'k',linewidth = 2)

plt.xlabel("Spatial Location",fontsize = 35)
plt.ylabel("Time", fontsize = 35)
plt.yticks([0,np.math.pi/3,2*np.math.pi/3,np.math.pi,4*np.math.pi/3,5*np.math.pi/3,2*np.math.pi],['0',r'$\frac{\pi}{3}$',r'$\frac{2\pi}{3}$',r'$\pi$',r'$\frac{4\pi}{3}$',r'$\frac{5\pi}{3}$', r'$2\pi$'],fontsize = 30)
plt.xticks([0,2,4],['0',r'$\frac{L}{2}$','L'],fontsize = 30)
plt.title("Absolute Error")
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=.15))
sm._A = []
plt.colorbar(sm,label = "Absolute Error")
plt.subplots_adjust(left=0.12, bottom = .15, right = 0.95)

plt.show()
