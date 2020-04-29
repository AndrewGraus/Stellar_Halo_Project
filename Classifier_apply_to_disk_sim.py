#!/usr/bin/env python3

## job name
#PBS -N CL_disk
## queue: devel <= 2 hr, normal <= 8 hr, long <= 5 day
#PBS -q normal
#PBS -l select=1:ncpus=20:mpiprocs=1:ompthreads=1:model=ivy
#PBS -l walltime=8:00:00
## combine stderr & stdout into one file
#PBS -j oe
## output file name
#PBS -o classifier_test.txt
#PBS -M agraus@utexas.edu
#PBS -m bae
#PBS -V
#PBS -W group_list=s1542

#The classifier has been trained, now I want to test it on the test split 

import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from subprocess import call

f = h5py.File('./disk_files_full/'+data_file)
#halo data
snap = f['Snapshot00152']['HaloCatalog_Rockstar']
mass_halo = snap['Mvir'][:]
pos_halo = snap['Center'][:]
vel_halo = snap['Velocity']

#part data
parts = f['PartType1']
coords = parts['Coordinates'][:]/h
vel = parts['Velocities'][:]
mass= parts['Masses'][:]/h

#identify the host                                                                                                    
host_id = np.argmax(mass_halo)
host_mass = mass_halo[host_id]
host_pos = pos_halo[host_id]
host_vel = vel_halo[host_id]

#check that I trained on cosmology corrected coordinates
phase_space_coords =  np.concatenate((coords-host_pos,vel-host_vel),axis=1)

f.close()

print('loading classifier model')

model_output = model.predict(phase_space_coords)

print(model_output)
#now I want to save the model as an hdf5

#if path.exists('/nobackupp8/agraus/'):
#    call(['conda','deactivate'])
#    call(['module','unload','miniconda3/v4'])
#    call(['module','load','python3/Intel_Python_3.6_2018.3.222'])


print('finished')
