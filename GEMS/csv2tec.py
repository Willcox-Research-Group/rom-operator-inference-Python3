import numpy as np
import csv
import os
import pandas as pd

def csv2tec(file_to_read,file_path_grid,tecplot_path_out):
    '''
    Convert 'file_to_read' to tecplot file

    INPUTS
        file_to_read - string containing the CSV file path (numElements x numVariables)
        file_path_grid - string containing file path to grid.dat
        tecplot_path_out - folder where you want to save the tecplot file (it will be created if it doesnt exist)
    OUTPUTS
        saved file to tecplot_path_out with the same name as file_to_read with .dat extension
    '''
    # Create target Directory if it doesnt exist
    if not os.path.exists(tecplot_path_out): os.mkdir(tecplot_path_out)


    solutiontime = 0.0150000
    ''''''''' Read in data '''''''''


    numElements = 38523

    data = np.loadtxt(file_to_read,delimiter = ",")


    ''''''''' Read in grid data '''''''''
    datContent_xy = [i.strip().split() for i in open("%s" %file_path_grid).readlines()] #Each element in the list is a line

    data_begins_xy = 9 #First line of data is line 9, start from 0.

    numNodes = 39065
    num_columns = len(datContent_xy[data_begins_xy])

    numVars = 3
    lines_per_var = np.int(numNodes/num_columns)

    # connectivity info from the bottom of the files
    connectivity = np.array(datContent_xy[data_begins_xy+lines_per_var*numVars:]) 

    x = np.zeros((numNodes),dtype = float)
    y = np.zeros((numNodes),dtype = float)
    count = 0
    for line in range(data_begins_xy,lines_per_var+data_begins_xy): 
        x[count:count+num_columns] = datContent_xy[line] #[float(i) for i in datContent_xy[line]] 
        count += num_columns
  
    count = 0
    for line in range(lines_per_var+data_begins_xy,2*lines_per_var+data_begins_xy):                 # x coords 
        y[count:count+num_columns] = datContent_xy[line]# y coords
        count += num_columns

    ''''''''''''''''''''''''''''''''''''''''''


    ''''''''' Write matlab data and grid data to tecplot '''''''''
    title = "TITLE =\"GEMS DATA\" \n"
    variables = "VARIABLES =\"P\"\n\"U\"\n\"V\"\n\"q\" \n\"CH4_mc\" \n\"O2_mc\" \n\"H2O_mc\" \n\"CO2_mc\"  \n"
    junk = "ZONE T=\"zone 1\" \n STRANDID=0, SOLUTIONTIME=%f \n Nodes=%d, Elements=%d, ZONETYPE=FEQuadrilateral\n DATAPACKING=BLOCK\n VARLOCATION=([1-8]=CELLCENTERED) \n DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n" %(solutiontime,numNodes,numElements)
    numCols = 4
    numVars = 8
    lines_per_var = int(1+((numElements-(numCols-1))/4))

    with open("%s%s.dat"%(tecplot_path_out,file_to_read[len(file_to_read) - file_to_read[::-1].find('/'):-4]), "w") as f:
        writer = csv.writer(f, delimiter = " ")
        f.write(title)
        f.write(variables)
        f.write(junk)

        for var in range(numVars):
            c = 0
            for i in range(lines_per_var-1):
                writer.writerow(data[c:c+numCols,var])
                c += numCols
            writer.writerows([data[c:c+numCols-1,var]])
            
        for line in range(numElements):
            writer.writerow(connectivity[line,:])

import matlab.engine
eng = matlab.engine.start_matlab()


def main():
    '''
    ------------------------
    Begin User Input
    ------------------------
    '''
    train_time = 5000  # number of snapshots for learning operators
    k_ridge = [5e4]     # regularization parameter
    r_vals = [9]       # POD basis size
    save_time = 9999    # timestep where data was saved
    tecplot_path_out = 'results/tecdata/' #folder to store plotable tecplot data 
    '''
    ------------------------
    End User Input
    ------------------------
    '''

    file_path_grid = 'grid.dat' # file path for grid data
    for k in k_ridge:
        for r in r_vals:
            file_to_read = 'results/field_data/relerror_t%d_r%d_l%d_nt%d.csv' %(save_time,r,k,train_time)
            csv2tec(file_to_read,file_path_grid,tecplot_path_out)
            a = eng.save_as_tec("%s%s"%(tecplot_path_out,file_to_read[len(file_to_read) - file_to_read[::-1].find('/'):-4]))

            file_to_read = 'results/field_data/data_t%d.csv' %save_time  
            csv2tec(file_to_read,file_path_grid,tecplot_path_out)
            a = eng.save_as_tec("%s%s"%(tecplot_path_out,file_to_read[len(file_to_read) - file_to_read[::-1].find('/'):-4]))

            file_to_read = 'results/field_data/abserror_t%d_r%d_l%d_nt%d.csv' %(save_time,r,k,train_time)
            csv2tec(file_to_read,file_path_grid,tecplot_path_out)
            a = eng.save_as_tec("%s%s"%(tecplot_path_out,file_to_read[len(file_to_read) - file_to_read[::-1].find('/'):-4]))

            file_to_read = 'results/field_data/xr_t%d_r%d_l%d_nt%d.csv' %(save_time,r,k,train_time)
            csv2tec(file_to_read,file_path_grid,tecplot_path_out)
            a = eng.save_as_tec("%s%s"%(tecplot_path_out,file_to_read[len(file_to_read) - file_to_read[::-1].find('/'):-4]))

if __name__ == '__main__':
    main()