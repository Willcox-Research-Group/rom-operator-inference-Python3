''' 
This file contains functions to do chemistry related conversions of data

1) compute the temperature given pressure, density and molar concentrations
2) compute the specific gas constant of a mixture for a given set of mass fractions
3) convert mass fractions to molar concentrations
4) convert molar concentrations to mass fractions
'''

import numpy as np

def compute_temperature(P,q,CH4_molar,O2_molar,CO2_molar,H2O_molar, return_extras = False):
	'''
	Compute temperature from given, predicted state(s) using the ideal gas law 
	...............................................
	INPUT:
	...............................................
	CH4, O2, CO2, H2O - species molar concentrations 
						(4) ndarrays 38523 x numsnaps
	P - Pressure (Pa = kg/ms^2)
		ndarray 38523 x numsnaps
	q - specific volume (1/density)
						ndarray of size 38523 x numsnaps
	...............................................
	OUTPUT:
	...............................................	
	T - temperature 
		ndarray 38523 x numsnaps
	'''

	#get mass fractions
	CH4,O2,CO2,H2O = molarconcentrations_2_massfractions(CH4_molar,O2_molar,CO2_molar,H2O_molar,q)

	if len(P.shape) > 1:
		#number of snapshots
		numsnaps = CH4_molar.shape[1]

		R_specific = np.zeros(numsnaps)
		#get specific gas constant for each snapshot
		for kk in range(numsnaps): 
			R_specific[kk] = compute_specific_gas(np.vstack((CH4[:,kk],O2[:,kk],CO2[:,kk],H2O[:,kk])).T, [16.04,32.0,18.0,44.01])
	
	else:
		#get specific gas constant 
		R_specific = compute_specific_gas(np.vstack((CH4,O2,CO2,H2O)).T, [16.04,32.0,18.0,44.01])
	
	#specific gas is the mean value 
	R_specific = np.mean(R_specific)
	
	#compute temperature 
	T = P*q/R_specific 
	if return_extras:
		return T,R_specific,np.vstack((CH4.T,O2.T,CO2.T,H2O.T)).T
	else:
		return T

def compute_specific_gas(mass_fractions,molar_masses):
	'''
	Compute the specific gas constant for a given state of mass fractions.

	R_specific = R/M_avg    --- R : universal gas constant (8.314 J/mol K)
	1/M_avg = sum Y_i/M_i 	--- Y_i are species mass fractions and M_i are corresponding molar mass (g/mol)

	...............................................
	INPUT:
	...............................................
	mass_fractions - mass fractions for each species in the domain (each column is a single mass fraction, each row is an element in the domain)
					 ndarray (number of elements x number of species)
	molar_masses   - the molar masses corresponding to the species mass fractions (g/mol)
					 ndarray (number of species x 1)
	...............................................
	OUTPUT:
	...............................................
	R_specific - specific gas constant J/(kg K)
				 ndarray (number of elements x 1)

	'''
	# print("R_specific input size = ", mass_fractions.shape)
	
	R_universal = 8.3144598 #J/mol K

	number_of_species = len(molar_masses)

	#average molar mass
	M_avg=0
	for i in range(number_of_species):
		M_avg += mass_fractions[:,i]/molar_masses[i]

	R_specific = R_universal*(M_avg)
	return 1000*np.mean(R_specific)

def massfractions_2_molarconcentrations(CH4,O2,CO2,H2O,P,T,R):
	'''
	Convert mass fractions to molar concentrations following https://en.wikipedia.org/wiki/Molar_concentration
	...............................................
	INPUT:
	...............................................
	CH4, O2, CO2, H2O - species mass fractions 
						(4) ndarrays 38523 x numsnaps 
	P - Pressure (Pa = kg/ms^2)
		ndarray 38523 x numsnaps
	T - Temperature (Kelvin)
		ndarray 38523 x numsnaps
	R - Average specific gas constant (J/kgK)
		float 
	...............................................
	OUTPUT
	...............................................
	CH4_molar,O2_molar,CO2_molar,H2O_molar - species molar concentrations (mol/m^3)
											 (4) ndarrays 38523 x numsnaps
	density - density of the mixture (kg/m^3)
			  ndarray 38523 x numsnaps
	'''

	#molar mass in kg/kmol
	mole_mass_CH4 = 16.04
	mole_mass_O2 = 32.0
	mole_mass_H2O = 18.0
	mole_mass_CO2 = 44.01

	if P.shape != T.shape:
		print("Pressure and Temperature not of correct shape!")
		print("Pressure shape = ", P.shape)
		print("Temperature shape = ", T.shape)
		print("Terminating!")
		exit()

	#compute density of mixture (ideal gas law)
	density = P/(R*T) #(kg)/(m^3)

	#convert mass fractions to molar concentrations (mol/m^3)
	CH4_molar = (density)*CH4/(mole_mass_CH4)
	O2_molar = (density)*O2/(mole_mass_O2)
	CO2_molar = (density)*CO2/(mole_mass_CO2)
	H2O_molar = (density)*H2O/(mole_mass_H2O)
	return CH4_molar,O2_molar,CO2_molar,H2O_molar,density

def molarconcentrations_2_massfractions(CH4_molar,O2_molar,CO2_molar,H2O_molar,q):
	'''
	Convert molar concentrations to mass fractions following https://en.wikipedia.org/wiki/Molar_concentration
	...............................................
	INPUT:
	...............................................
	CH4_molar,O2_molar,CO2_molar,H2O_molar - species molar concentrations (kmol/m^3)
						(4) ndarrays 38523 x numsnaps 
	q - specific volume (1/density)
						ndarray of size 38523 x numsnaps

	...............................................
	OUTPUT
	................. ..............................
	CH4,O2,CO2,H2O - species mass fractions
					 (4) ndarrays 38523 x numsnaps
	'''

	#molar mass in kg/kmol
	mole_mass_CH4 = 16.04
	mole_mass_O2 = 32.0
	mole_mass_H2O = 18.0
	mole_mass_CO2 = 44.01

	#convert molar concentrations to mass fractions 
	CH4 = CH4_molar*mole_mass_CH4*q
	O2  = O2_molar*mole_mass_O2*q
	H2O = H2O_molar*mole_mass_H2O*q
	CO2 = CO2_molar*mole_mass_CO2*q

	return CH4,O2,CO2,H2O


