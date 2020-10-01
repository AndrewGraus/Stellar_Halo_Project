#!/usr/bin/env python3

#This is a basic attempt to use the tensorflow regressors since
#the sklearn regressors take too long to run

import numpy as np
import h5py

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from subprocess import call

#import tensorflow_docs as tfdocs

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

f_halo = h5py.File('../m12i_res_7100_cdm/halo/halo_600.hdf5')

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

dm_mass = f['PartType1']['Masses'][:]
stellar_mass = f['PartType1']['Stellar_Masses'][:]
coords = f['PartType1']['Coordinates'][:]
vel = f['PartType1']['Velocities'][:]
mass_ratio = stellar_mass/dm_mass[0]

phase_space_coords =  np.concatenate((coords-host_pos,vel-host_vel),axis=1)

non_zero_mask = (mass_ratio>0.0) #only want to train on things that are non-zero

labels_non_zero = mass_ratio[non_zero_mask]
phase_space_non_zero = phase_space_coords[non_zero_mask]

print('training set:')
print('labels: {}'.format(labels_non_zero.shape))
print('data: {}'.format(phase_space_non_zero.shape))

f.close()
f_halo.close()

#This is just an example from the tensorflow website that I hacked to take in 6d data
#No clue if this will work
def build_model():
    model = keras.Sequential([layers.Flatten(input_shape=(6,)),
                              layers.Dense(64,activation='relu'),
                              layers.Dense(64,activation='relu'),
                              layers.Dense(1)
                              ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer,metrics=['mae','mse'])

    return model

print('building model')

model = build_model()

print(model.summary())

EPOCHS = 1000

print('training model')

history = model.fit(phase_space_non_zero,labels_non_zero,epochs=EPOCHS, validation_split=0.2, verbose=1)

model.save('saved_models/regressor_test_epoch1000.h5')

print('finished')
