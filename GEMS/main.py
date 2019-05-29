import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
from sklearn import preprocessing
import os
from operator_inference import integration_helpers
import scaling_tools as scale_tool
import chemistry_conversions as chem
import h5py_cache as h5c
from operator_inference import OpInf
import projection_helpers
import time
from operator_inference import opinf_helper
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
cm = plt.cm.get_cmap('RdYlBu')

def main():
	'''
	Constructs the operator inference problem. Solves for operators and integrates resulting ODE. Computes error
	for different basis sizes and regularization parameters
	'''

	''' 
	--------------------------------------------
	USER INPUTS
	--------------------------------------------

	k_ridge: list of floats
		The value or values to use for regularization parameter

	r_vals : list of ints
		the POD basis dimensions 

	train_time: int 
		number of snapshots for training (doing opinf)

	forecast_time: int
		number of timesteps passed the training set to predict 

	plot_svd : bool
		True = plot singular value decay   
		False = dont plot 

	compute_species_integral : bool
		True = compute sum of each species at each time step and saves to file
		False = Dont compute sum 

	plot_forecast: bool
		True = plot the time trace of the forecast
	    False = dont plot it

	element_to_plot: int 
		which element to plot time trace for

	output_filename : string
		filename to save the timetrace to 

	save_forecast: bool
		True = save data for prediction in text file
		False = dont save data

	save_data: bool
		True = save the recovered,re dimensionalized solution and error at time "time_to_save"
		False = dont save

	time_to_save: int
		timestep to save 

	monitor_unphysical_values: bool
		whether to count the negative values occuring in each variable

	datafilefolder: string (hdf5 filename)
		folder which contains the projected data , 
		Files inside folder should be called "data_reduced_%d.h5" %d is the dimension number
		Should have datasets 'data'

	xdotfilefolder: string (hdf5 filename)
		folder which contains the projected xdot , 
		Files inside folder should be called "xdot_reduced_%d.h5" %d is the dimension number
		Should have dataset 'xdot' 

	fulldatapath: string (hdf5 filename)
		full filepath for the full dimensional data 
		should be in dataset 'data'

	svd_filepath: string (hdf5 filename)
		where the svd lives - should be in a data set 'U' and 'S'
	--------------------------------------------

	--------------------------------------------
	'''


	#--------------------------------------------
	#	    	USER INPUTS BEGIN
	#--------------------------------------------
	k_ridge = [5e4]
	colors = plt.cm.jet(np.linspace(0,1,len(k_ridge)))
	r_vals = [9]#,30,35,40,45]

	train_time = 5000
	forecast_time = 20000-train_time

	plot_svd = False

	compute_species_integral = False

	plot_forecast = False
	element_to_plot = [36916-1,37892-1,10518-1,6296-1] #this is (0,22.5), (60,22.5), (0, 12), (-20,10) -- [22953-1,37458-1,34536-1,4354-1]
	output_filename = "_train%d_test%d_pbackInput.pdf" %(train_time,forecast_time)

	save_forecast = False
	time_trace_folders = ["Monitor_Location_0_225","Monitor_Location_60_225","Monitor_Location_0_12","Monitor_Location_n20_10"]


	save_data = True
	time_to_save = 9999

	monitor_unphysical_values = True

	datafilefolder = "OpInfdata/reducedData/data_reduced_minmax_nt%d" %train_time
	xdotfilefolder = "OpInfdata/reducedData/xdot_reduced_minmax_nt%d" %train_time

	fulldatapath = "OpInfdata/data_minmax.h5"
	svd_filepath = 'OpInfdata/svd_nt%d.h5' %train_time


	#--------------------------------------------
	#			USER INPUTS END
	#--------------------------------------------

	Nfull = 308184 		#length of full state
	N = 38523			#length of each variable

	total_time = np.linspace(0.015,0.017,20000) #Time steps for all 20000 
	dt = 1e-7


	hf = h5py.File(svd_filepath)
	Und = hf['U'][:,:max(r_vals)] 			#Singular vectors
	Snd = hf['S'][:]			#Singular values
	hf.close()


	'''plot singular value decay '''
	if plot_svd:
		plt.figure(2)
		plt.semilogy(Snd[:150]/Snd[0],marker = 'o',label = 'standard',linewidth=2)
		plt.grid()
		plt.xlabel("Singular Value Index")
		plt.ylabel("Normalized Singular Value")
		plt.show()


	# define arrays for saving error
	relative_error_over_domain = np.zeros((len(r_vals),len(k_ridge),8))

	log_res = np.zeros((len(k_ridge),len(r_vals)))
	log_sol = np.zeros((len(k_ridge),len(r_vals)))


	#input function
	U = lambda t: 1e6*(1+0.1*np.sin(2*np.math.pi*5000*t))	

	for kidx,k in enumerate(k_ridge):
		
		print("Regularization parameter = %1.1f" %k)

		for idx,r in enumerate(r_vals):

			print("POD basis dimension = ", r)



			fdata = h5py.File("%s/data_reduced_%d.h5" %(datafilefolder,r),'r')
			xhat = fdata['data'][:,:train_time]

			fxdot = h5py.File("%s/xdot_reduced_%d.h5" %(xdotfilefolder,r),'r')
			xdot = fxdot['xdot'][:,:train_time]
			fdata.close()
			fxdot.close()

			'''
			------------------------
			Define the model 
			------------------------
			'''
			mymodel = OpInf.model('LQ',True) 					# a linear quadratic with input


			'''
			------------------------
			Fit the model 
			------------------------
			'''
			start = time.time()
			mymodel.fit(r,k,xdot,xhat,U(total_time[:train_time]))
			# A,H,B = mymodel.get_operators()
			end = time.time()
			time1 = end-start
			print("Operators found in %.4f seconds" %(end-start))

		

			'''
			------------------------
			Simulate the model 
			------------------------
			'''

			init = xhat[:,0]
			xhat,xdot = None,None

			start = time.time()
			xr,break_point = mymodel.predict(init, train_time+forecast_time, dt, U(total_time[:train_time+forecast_time]))
			end = time.time()
			time2 = end-start
			print("Simulation done in %.4f seconds" %(end-start))

			'''
			------------------------
			Unscale the data 
			------------------------
			'''
			print("Und shape - ", Und.shape)

			start = time.time()

			xr_element_to_plot = np.zeros((train_time+forecast_time,len(element_to_plot)))
			for ii,jj in enumerate(element_to_plot):
				for tt in range(train_time+forecast_time):
					xr_element_to_plot[tt,ii] = Und[jj,:r]@xr[:,tt]

				xr_element_to_plot[:,ii] = scale_tool.minmax_scaler(xr_element_to_plot[:,ii],-1,1,N,0).reshape((-1,))
			end = time.time()
			time3 = end-start
			print("Reconstruction of timetraces in %.4f seconds " %(end-start))

			start = time.time()
			xr_time_to_save = Und[:,:r]@xr[:,time_to_save]
			xr_time_to_save = scale_tool.minmax_scaler(xr_time_to_save,-1,1,N,0)
			xr_time_to_save = xr_time_to_save[:,0]

			end = time.time()
			time4 = end-start
			print("Reconstruction of fields in %.4f seconds " %(end-start))

			del mymodel 

			if compute_species_integral:
				print("Reconstructing and unscaling solutions")
				xr = Und[:,:r]@xr[:,:10000]
				xr = scale_tool.minmax_scaler(xr,-1,1,N,0)
				print("Computing integral...")
				integral_species = np.zeros((4,10000))
				for jj in range(4):
					for ii in range(10000):
						integral_species[jj,ii] = np.sum(xr[4*N + (jj*N): 4*N + ((jj+1)*N), ii])
				np.savetxt('results/integrated_species/integral_species_r%d_nt%d_l%d' %(r,train_time,k), integral_species)
				del xr

			if save_data:


				hfdata = h5py.File('OpInfdata/data_unscaled.h5')
				# hfdata = h5py.File(fulldatapath)
				data = hfdata['data'][:,time_to_save]
				hfdata.close()

				error = np.zeros((Nfull))
				error[:N] = projection_helpers.rpe(data[:N],xr_time_to_save[:N])


				hft = h5py.File('OpInfdata/temperature.h5','r')
				true_T = hft['T'][:,time_to_save]
				hft.close()

				

				# replace specific volume with temperature 
				print("Computing temperature at last timestep")
				predicted_T = chem.compute_temperature(xr_time_to_save[:N],xr_time_to_save[3*N:4*N],xr_time_to_save[4*N:5*N],xr_time_to_save[5*N:6*N],xr_time_to_save[6*N:7*N],xr_time_to_save[7*N:8*N])
				error[3*N:4*N] = abs(true_T-predicted_T)/abs(true_T)

				print("Minimum predicted Temp = ", np.min(predicted_T))

				#replace specific volume error with temperature error 
				data[3*N:4*N]  = true_T
				xr_time_to_save[3*N:4*N] = predicted_T

				abs_error = np.reshape(abs(data - xr_time_to_save),(Nfull,1))
				abs_error = np.reshape(abs_error, (N,8),order = 'F')
				error[N:2*N]   = abs_error[:,1]
				error[2*N:3*N] = abs_error[:,2]
				error[4*N:5*N] = abs_error[:,4]/max(data[4*N:5*N])
				error[5*N:6*N] = abs_error[:,5]/max(data[5*N:6*N])
				error[6*N:7*N] = abs_error[:,6]/max(data[6*N:7*N])
				error[7*N:8*N] = abs_error[:,7]/max(data[7*N:8*N])

				error = np.reshape(error, (Nfull,1))

				error = np.reshape(error, (N,8),order = 'F')
				norm_error = np.mean(error,axis=0)

				#relative error for P and T
				relative_error_over_domain[idx,kidx,[0,3]] = norm_error[[0,3]] 

				#u and v have absolute error
				relative_error_over_domain[idx,kidx,[1,2]] = np.mean(abs_error[:,[1,2]],axis = 0)

				# species are absolute error over max true
				relative_error_over_domain[idx,kidx,4] = np.mean(abs_error[:,4]/max(data[4*N:5*N]))
				relative_error_over_domain[idx,kidx,5] = np.mean(abs_error[:,5]/max(data[5*N:6*N]))
				relative_error_over_domain[idx,kidx,6] = np.mean(abs_error[:,6]/max(data[6*N:7*N]))
				relative_error_over_domain[idx,kidx,7] = np.mean(abs_error[:,7]/max(data[7*N:8*N]))
				print("\n-----------------------------------------")
				print("Average error over domain \n-----------------------------------------\n")
				print("Pressure : ", relative_error_over_domain[idx,kidx,0])
				print("Temperature : ", relative_error_over_domain[idx,kidx,1])
				print("X velocity : ", relative_error_over_domain[idx,kidx,2])
				print("Y velocity : ", relative_error_over_domain[idx,kidx,3])
				print("CH4 molar conc : ", relative_error_over_domain[idx,kidx,4])
				print("O2 molar conc : ", relative_error_over_domain[idx,kidx,5])
				print("CO2 molar conc : ", relative_error_over_domain[idx,kidx,6])
				print("H2O molar conc : ", relative_error_over_domain[idx,kidx,7])
				print("-----------------------------------------")


				np.savetxt('results/field_data/relerror_t%d_r%d_l%d_nt%d.csv' %(time_to_save,r,k,train_time), error,delimiter = ",")
				np.savetxt('results/field_data/abserror_t%d_r%d_l%d_nt%d.csv' %(time_to_save,r,k,train_time), abs_error,delimiter = ",")
				np.savetxt('results/field_data/xr_t%d_r%d_l%d_nt%d.csv' %(time_to_save,r,k,train_time), np.reshape(xr_time_to_save,(N,8),order = 'F'),delimiter = ",")
				
			if save_forecast:
				plot_time = train_time+forecast_time
				for jj_idx in range(len(element_to_plot)):
					i = 0 #pressure
					np.savetxt('results/timetraces/%s/nt%d/r%d_k%d.txt' %(time_trace_folders[jj_idx],train_time,r,k),xr_element_to_plot[:plot_time,jj_idx])

			if plot_forecast:
				data = np.loadtxt('chengs_data/TimeTraceData/point_mon_7.txt', delimiter = ",")
				time_steps = data[:train_time+forecast_time,0]

				if break_point < train_time+forecast_time - 1:
					plot_time = break_point - 20
				else:
					plot_time = train_time+forecast_time

				varnames = ["Pressure","Temperature"]
				for idx,jj in enumerate(element_to_plot):
					c = 0
					for i in [0]: #Pressure
						truth = data[:plot_time,i+1].reshape((-1,1))
						pred = xr_element_to_plot[:plot_time,idx]

						plt.figure(idx)
						plt.plot(time_steps[:plot_time], xr_element_to_plot[:plot_time,idx],linestyle = '--', linewidth=2,label='ROM, r = %d' %r)
						plt.plot(time_steps[:train_time+forecast_time], data[:train_time+forecast_time,i+1],linewidth=2,label = "TRUE")
						plt.plot(np.ones(300)*time_steps[train_time-1], np.linspace(min(np.minimum(truth,pred)[0]), max(np.maximum(truth,pred)[0]),300),color='black')
						plt.xlabel("Time (s)",fontsize = 25)
						plt.xticks(fontsize = 35)
						plt.yticks(fontsize = 35)
						plt.ylabel(varnames[c],  fontsize = 25)
						plt.ticklabel_format(style='sci',axis = 'y',scilimits=(0,0))
						plt.legend(fontsize=20)
						# plt.savefig('results/timetraces/%s/nt%d/r%d_k%d_%s_%s' %(time_trace_folders[idx],train_time,r,k,varnames[c],output_filename),bbox_inches="tight", format='pdf',dpi=300)
						# plt.clf()
						
						c += 1
				plt.show()
			
			if monitor_unphysical_values:
				unphys_timestep = np.zeros((8,train_time+forecast_time))
				min_unp = np.zeros(8)
				print("Reconstructing the whole thing...")
				
				for ii in range(train_time+forecast_time):
					full_timestep = Und[:,:r]@xr[:,ii]
					full_timestep = scale_tool.minmax_scaler(full_timestep,-1,1,N,0)
					for var in range(8):
						unphys_timestep[var,ii] = np.sum(full_timestep[var*N:(var+1)*N] < 0)

				np.savetxt("results/unphysical_values/Unphysical_Values_Timestep", unphys_timestep)

			return time1,time2,time3,time4

if __name__ == '__main__':
	
	main()
	

