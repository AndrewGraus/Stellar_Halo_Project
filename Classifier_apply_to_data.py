#!/usr/bin/env python3

## job name
#PBS -N CL_test
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

phase_space_coords =  np.concatenate((coords-host_pos,vel-host_vel),axis=1)

#use this handy module to splot my phase space coords into a training and test
#sample

X_train, X_test, y_train, y_test = train_test_split(phase_space_coords,mass_ratio,
                                                    test_size=0.5,random_state=102)

f.close()
f_halo.close()

print('loading classifier model')

#For training a classifier I just need an interger based on the number
#of labels, since I have two labels (zero and non-zero) so I just need
#to take every zero value and assign it zero, and every non-zero
#value and assign it one

#first do a mask that is y_train !=0 to give an array of TRUE FALSE
#where 0.0 is false and > 0.0 is true. Then convert the True-False
#to 0 and 1

y_train_classifier = np.array((y_train!=0),dtype=int)
y_test_classifier  = np.array((y_test!=0),dtype=int)

#now I can load up the model I've already trained
model = tf.keras.models.load_model('saved_models/Classifier_test_sigmoid.h5')

test_loss, test_acc = model.evaluate(X_test,y_test_classifier,verbose=2)

model_output = model.predict(X_test)

print(model_output)
#now I want to save the model as an hdf5

#if path.exists('/nobackupp8/agraus/'):
#    call(['conda','deactivate'])
#    call(['module','unload','miniconda3/v4'])
#    call(['module','load','python3/Intel_Python_3.6_2018.3.222'])


print('finished')
