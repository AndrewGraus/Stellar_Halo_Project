#!/usr/bin/env python3

## job name
#PBS -N Regressors_test
## queue: devel <= 2 hr, normal <= 8 hr, long <= 5 day
#PBS -q long
#PBS -l select=1:ncpus=20:mpiprocs=1:ompthreads=1:model=ivy
#PBS -l walltime=120:00:00
## combine stderr & stdout into one file
#PBS -j oe
## output file name
#PBS -o regression.txt
#PBS -M agraus@utexas.edu
#PBS -m bae
#PBS -V
#PBS -W group_list=s1542

#Gaussian process is a resource hog so I'm trying some other regressors to 
#see if I can get an answer without taking up too many resources

import numpy as np
import h5py
from sklearn.gaussian_process import GaussianProcess
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.svm import NuSVR
from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import BaggingRegressor
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split

#For this module it seems like you simply put in an n dimensional training array X
#and the training answers y and then do a fit to get a function gp
#then use that to predict stuff 
# 
# gp = GaussianProcess(corr='squared_exponential')
# gp.fit(X,y)
#
# y_pred, dy_pred = gp.predict(X_new,eval_MSE=True)
#
# In my case X is the input parameters (r-r_host, and v-v_host) for the training
# dm particles and y is the mass fraction for those particles
#
# X_new is the r-r_host and v-v_host for the disk data set
# and will give y_pred or the predicted mass fractions (I think)
#
# so just implement it and then see what we get:

f_halo = h5py.File('../m12i_res_7100_cdm/halo_600.hdf5')

pos_halo = f_halo['position'][:]
mass_halo = f_halo['mass'][:]
radius_halo = f_halo['radius'][:]
vel_halo = f_halo['velocity'][:]

#identify the host                                                                                                    
host_id = np.argmax(mass_halo)
host_mass = mass_halo[host_id]
host_pos = pos_halo[host_id]
host_vel = vel_halo[host_id]

f = h5py.File('DM_data_w_stars_training.hdf5')

coords = f['PartType1']['Coordinates'][:]
vel = f['PartType1']['Velocities'][:]
mass_ratio = f['PartType1']['Mass_Ratio'][:]

#1) attempt to save memory by reducing number of points
#   remove everything outside 400 kpc (these shouldn't)
#   factor in anyways

#I think I need to merge coords and vel into an array

phase_space_coords =  np.concatenate((coords-host_pos,vel-host_vel),axis=1)


#use this handy module to splot my phase space coords into a training and test
#sample

X_train, X_test, y_train, y_test = train_test_split(phase_space_coords,mass_ratio,
                                                    test_size=0.5,random_state=102)

#Some simple linear regressors (These are too simple)
#LR = LinearRegression()
#LR.fit(X_train,mass_ratio)

f.close()
f_halo.close()

print('running Vector regressor')

SVR_func = NuSVR(verbose=1)
SVR_func.fit(X_train,y_train)
mass_ratio_SVR = SVR_func.predict(X_test)

print('running Gradient Descent regressor')

SGD_func = SGDRegressor(verbose=1)
SGD_func.fit(X_train,y_train)
mass_ratio_SGD = SGD.predict(X_test)

output_array = np.zeros((len(mass_ratio_SVR),3))
output_array[:,0] = y_test
output_array[:,1] = mass_ratio_SVR
output_array[:,2] = mass_ratio_SGD

np.savetxt('./outputs.txt',output_array)

print('finished')
