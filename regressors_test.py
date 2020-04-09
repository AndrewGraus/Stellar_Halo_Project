#!/usr/bin/env python3

## job name
#PBS -N Regressors_test
## queue: devel <= 2 hr, normal <= 8 hr, long <= 5 day
#PBS -q normal
#PBS -l select=1:ncpus=24:mpiprocs=1:ompthreads=1:model=has
#PBS -l walltime=8:00:00
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
from sklearn.ensemble import BaggingRegressor
from sklearn import preprocessing
from sklearn import utils

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

#I think I need to merge coords and vel into a 
#2X6 array

#Now this matrix is so large we run into memory errors
#
#A few potential solutions:
# 1) reduce the number of input points (limit to dm part within 400 kpc?) [NOPE]
# 2) change algorithm (some might require less memory or some option)
#    may reduce memory
#    - Try using GaussianProccessRegressor module [X]
#    - Try "bagging" 
# 3) run on a processor with more memory or figure out how to parallelize

X_train =  np.concatenate((coords-host_pos,vel-host_vel),axis=1)

print(X_train.shape)
print(X_train.shape[0]/1000)
#Now I can use the GaussianProcess to predict

#gp = GaussianProcess(corr='squared_exponential')
#gp = GaussianProcessRegressor()
#gp.fit(X_train,mass_ratio)

#Try a bagging regressor
#which subsamples and then averages (I think)

LR = LinearRegression()
LR.fit(X_train,mass_ratio)

LassoR = Lasso()
LassoR.fit(X_train,mass_ratio)

RidgeR = Ridge()
RidgeR.fit(X_train,mass_ratio)

f.close()
f_halo.close()

#Lets test it on itself

mass_ratio_LR = LR.predict(X_train)
mass_ratio_Lasso = LassoR.predict(X_train)
mass_ratio_Ridge = RidgeR.predict(X_train)

print(mass_ratio,np.min(mass_ratio), np.max(mass_ratio))
print(mass_ratio_LR,np.min(mass_ratio_LR), np.max(mass_ratio_LR))
print(mass_ratio_Lasso,np.min(mass_ratio_Lasso), np.max(mass_ratio_Lasso))
print(mass_ratio_Ridge,np.min(mass_ratio_Ridge), np.max(mass_ratio_Ridge))

h_disk = 0.6751

f_disk = h5py.File('../../z13_disk/halo_1107_Z13_disk_cat.hdf5')
part_pos_disk = f_disk['PartType1']['Coordinates'][:]/h_disk
part_vel_disk = f_disk['PartType1']['Velocities'][:]
part_mass_disk = f_disk['PartType1']['Masses'][:]*1.0e10/h_disk
part_ids_disk = f_disk['PartType1']['ParticleIDs'][:]

halo_cat = f_disk['Snapshot00152']['HaloCatalog_Rockstar']
halo_pos = halo_cat['Center'][:]/h_disk
halo_vel = halo_cat['Velocity'][:]
halo_mass = halo_cat['Mvir'][:]/h_disk

host_id = np.argmax(mass_halo)
host_mass = halo_mass[host_id]
host_pos = halo_pos[host_id]
host_vel = halo_vel[host_id]

X_test = np.concatenate((part_pos_disk-host_pos,part_vel_disk-host_vel),axis=1)

'''mass_ratio_LR = LR.predict(X_test)
mass_ratio_Lasso = LassoR.predict(X_test)
mass_ratio_Ridge = RidgeR.predict(X_test)

print(mass_ratio,np.min(mass_ratio), np.max(mass_ratio))
print(mass_ratio_LR,np.min(mass_ratio_LR), np.max(mass_ratio_LR))
print(mass_ratio_Lasso,np.min(mass_ratio_Lasso), np.max(mass_ratio_Lasso))
print(mass_ratio_Ridge,np.min(mass_ratio_Ridge), np.max(mass_ratio_Ridge))
'''
#okay now output the data to a file
#
#f_write = h5py.File('DM_data_ML_halo_1386.hdf5','w')
#
#f_write.create_dataset("PartType1/Coordinates",data=part_pos_disk)
#f_write.create_dataset("PartType1/Velocities",data=part_vel_disk)
#f_write.create_dataset("PartType1/ParticleIDs",data=part_ids_disk)
#f_write.create_dataset("PartType1/Masses",data=part_mass_disk)
#f_write.create_dataset("PartType1/Stellar_Masses",data=total_M_star_sorted)
#f_write.create_dataset("PartType1/Mass_Ratio",data=mass_ratio_trained)
#f_write.close()

print('finished')
