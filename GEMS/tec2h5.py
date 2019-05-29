import os
import numpy as np
import glob
import chemistry_conversions as chem
import scaling_tools as scale_tool
import h5py
import h5py_cache as h5c


'''
-------------------------------------------------------------------------
BEGIN USER INPUTS
-------------------------------------------------------------------------
'''
 # folder containing chengs "ncons" files - this folder should ONLY contain those files
folder_path_data = 'OriginalGEMS/'

#The following will be created if they dont already exist.
h5_path_out = 'OpInfdata/' #file path to save data (scaled and unscaled) as hdf5 files

#No. of elements in the spatial domain.
#This can be found in the header of the data files
numElements = 38523

#No. of variables
#This can be found in the header of the data files
numVars = 8

'''
-------------------------------------------------------------------------
END USER INPUTS
-------------------------------------------------------------------------
'''

# Create target Directory if it doesnt exist
if not os.path.exists(h5_path_out): os.mkdir(h5_path_out)



def sortKeyFunc(s):
    #function for sorting chengs files by timestep in the filename
    # file name format : 'test_file_ncons_150000.dat'
    return int(os.path.basename(s)[-10:-4])


''''''''' Read in variable data '''''''''

fnames = glob.glob(os.path.join(folder_path_data, '*'))
fnames.sort(key = sortKeyFunc) #sort based on timestep in filename
full_dataset = np.zeros((numElements*numVars,len(fnames)))


for idx,data_file in enumerate(fnames):
    data_name = data_file[len(folder_path_data):-4]

    datContent = [i.strip().split() for i in open("./%s%s.dat" %(folder_path_data,data_name)).readlines()] #Each element in the list is a line

    data_begins = 15
    numCols = 4
    
    solutiontime = float(datContent[10][1][13:])
    print("Solution time = ", solutiontime)
    lines_per_var = np.int(((numElements - 3)/numCols) + 1)

    #pressure
    data1 = []
    for line1 in range(data_begins,data_begins+lines_per_var):
        for ii in range(len(datContent[line1][:])):
            data1.append(datContent[line1][ii])

    #x velocity
    data2 = []
    for line2 in range(line1+1,line1+1+lines_per_var):
        for ii in range(len(datContent[line2][:])):
            data2.append(datContent[line2][ii]) 
  
    #y velocity
    data3 = []
    for line3 in range(line2+1,line2+1+lines_per_var):
        for ii in range(len(datContent[line3][:])):
            data3.append(datContent[line3][ii])   

    # temperature
    data4 = []
    for line4 in range(line3+1,line3+1+lines_per_var):
        for ii in range(len(datContent[line4][:])):
            data4.append(datContent[line4][ii]) 

    # CH4 mass fraction
    data5 = []
    for line5 in range(line4+1,line4+1+lines_per_var):
        for ii in range(len(datContent[line5][:])):
            data5.append(datContent[line5][ii])

    # O2 mass fraction
    data6 = []
    for line6 in range(line5+1,line5+1+lines_per_var):
        for ii in range(len(datContent[line6][:])):
            data6.append(datContent[line6][ii])    

    #H2O mass fraction
    data7 = []
    for line7 in range(line6+1,line6+1+lines_per_var):
        for ii in range(len(datContent[line7][:])):
            data7.append(datContent[line7][ii])  

    #CO2 mass fraction
    data8 = []
    for line8 in range(line7+1,line7+1+lines_per_var):
        for ii in range(len(datContent[line8][:])):
            data8.append(datContent[line8][ii])    


    full_dataset[:numElements,idx]      = data1
    full_dataset[numElements:2*numElements,idx]   = data2
    full_dataset[2*numElements:3*numElements,idx] = data3
    full_dataset[3*numElements:4*numElements,idx] = data4
    full_dataset[4*numElements:5*numElements,idx] = data5
    full_dataset[5*numElements:6*numElements,idx] = data6

    #i want h2o to come last
    full_dataset[6*numElements:7*numElements,idx] = data8
    full_dataset[7*numElements:8*numElements,idx] = data7

    ''''''''''''''''''''''''''''''''''''''''''


'''
LIFT AND SAVE DATA
'''  
#compute specific gas constant  
R_specific = np.zeros(len(fnames))
for tt in range(len(fnames)):
    R_specific[tt] = chem.compute_specific_gas(np.vstack((full_dataset[4*numElements:5*numElements,tt], full_dataset[5*numElements:6*numElements,tt],full_dataset[6*numElements:7*numElements,tt],full_dataset[7*numElements:8*numElements,tt])).T,[16.04,32.0,18.0,44.01])

R_specific = np.mean(R_specific)
print("Specific gas constant = ", R_specific)

#replace mass fractions with molar concentrations and temperature with density
full_dataset[4*numElements:5*numElements,:],full_dataset[5*numElements:6*numElements,:],full_dataset[6*numElements:7*numElements,:],full_dataset[7*numElements:8*numElements,:],full_dataset[3*numElements:4*numElements,:] = chem.massfractions_2_molarconcentrations(full_dataset[4*numElements:5*numElements,:], full_dataset[5*numElements:6*numElements,:],full_dataset[6*numElements:7*numElements,:],full_dataset[7*numElements:8*numElements,:], full_dataset[:numElements,:], full_dataset[3*numElements:4*numElements,:], R_specific)

#replace density with specific volume
full_dataset[3*numElements:4*numElements,:] = 1/full_dataset[3*numElements:4*numElements,:]

if len(fnames) < 1000:
    print("saving unscaled data...")
    hf = h5py.File("%s/data_unscaled.h5" %h5_path_out,'w')
    d = hf.create_dataset('data', data = full_dataset)
    hf.close()

    print("scaling...")
    #scale to [-1,1] range
    full_dataset = scale_tool.minmax_scaler(full_dataset,-1,1,numElements,1)

    print("saving scaled data...")
    hf = h5py.File("%s/data_minmax.h5" %h5_path_out,'w')
    d = hf.create_dataset('data', data = full_dataset)
    hf.close()

else:
    #save unscaled data
    chunk_shape = (100,1000)
    print("saving unscaled data...")
    hf = h5c.File("%s/data_unscaled.h5" %h5_path_out,'w', chunk_cache_mem_size = 1024**2*4000)
    d = hf.create_dataset('data', full_dataset.shape,dtype = 'f',chunks = chunk_shape,compression = "lzf")

    for i in range(full_dataset.shape[1]):
        d[:,i] = full_dataset[:,i]
    hf.close()

    print("scaling...")
    #scale to [-1,1] range
    full_dataset = scale_tool.minmax_scaler(full_dataset,-1,1,numElements,1)

    #save scaled data
    print("saving scaled data...")
    hf = h5c.File("%s/data_minmax.h5" %h5_path_out,'w', chunk_cache_mem_size = 1024**2*4000)
    d = hf.create_dataset('data', full_dataset.shape,dtype = 'f',chunks = chunk_shape,compression = "lzf")

    for i in range(full_dataset.shape[1]):
        d[:,i] = full_dataset[:,i]
    hf.close()





