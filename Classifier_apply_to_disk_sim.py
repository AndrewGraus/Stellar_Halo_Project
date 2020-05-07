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

f = h5py.File('/home5/agraus/data/disk_runs/z13_disk/halo_1107_Z13_disk_cat.hdf5')

h = f['Header'].attrs['HubbleParam']

#halo data
snap = f['Snapshot00152']['HaloCatalog_Rockstar']
mass_halo = snap['Mvir'][:]/h
pos_halo = snap['Center'][:]/h
vel_halo = snap['Velocity'][:]

#part data
parts = f['PartType1']
coords = parts['Coordinates'][:]/h
vel = parts['Velocities'][:]
mass= parts['Masses'][:]*1.0e10/h
ids = parts['ParticleIDs'][:]

#identify the host                                                                                                   
host_id = np.argmax(mass_halo)
host_mass = mass_halo[host_id]
host_pos = pos_halo[host_id]
host_vel = vel_halo[host_id]

print(host_id)
print(host_mass)
print(host_pos)
print(host_vel)

#check that I trained on cosmology corrected coordinates
phase_space_coords =  np.concatenate((coords-host_pos,vel-host_vel),axis=1)

f.close()

print('loading classifier model')

model = tf.keras.models.load_model('./saved_models/Classifier_biased_sigmoid.h5')

print('applying model to data')

model_output = model.predict(phase_space_coords)

#now I want to save the model as an hdf5

#Now I need to output the data
#I guess I could output another file that has the particle data from part 2, but JUST the
#particle data

print(model_output)
print(np.sum(model_output>0.0))

print('saving output')

f_write = h5py.File('./halo_1107_Z13_particles.hdf5')
f_write.create_dataset("PartType1/Coordinates",data=coords)
f_write.create_dataset("PartType1/Velocities",data=vel)
f_write.create_dataset("PartType1/ParticleIDs",data=ids)
f_write.create_dataset("PartType1/Masses",data=mass)
f_write.create_dataset("PartType1/Mass_Ratio",data=model_output)

print('finished')
