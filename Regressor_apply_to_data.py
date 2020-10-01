#!/usr/bin/env python3

import numpy as np
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

f_halo = h5py.File('../m12i_res_7100_cdm/halo/halo_600.hdf5')

pos_halo = f_halo['position'][:]
mass_halo = f_halo['mass'][:]
radius_halo = f_halo['radius'][:]
vel_halo = f_halo['velocity'][:]

f_predict = h5py.File('./halo_1107_Z13_particles.hdf5')

h = 0.675

coord_predict = f_predict['PartType1']['Coordinates'][:]
vel_predict = f_predict['PartType1']['Velocities'][:]
id_predict = f_predict['PartType1']['ParticleIDs'][:]

host_cen = np.array([37.53820323, 34.52403051, 37.02327211])*1000.0
host_vel = np.array([ 57.161152, -83.112122, -17.277088])

coord_diff_predict = coord_predict - host_cen
vel_diff_predict = vel_predict - host_vel

print('running Vector regressor')

phase_space_coords =  np.concatenate((coord_diff_predict,vel_diff_predict),axis=1)

model = tf.keras.models.load_model('./saved_models/regressor_test.h5')

model_output_10 = model.predict(phase_space_coords)

model = tf.keras.models.load_model('./saved_models/regressor_test_epoch100.h5')
model_output_100 = model.predict(phase_space_coords)

f_write = h5py.File('./predictions_from_regressor.hdf5')
f_write.create_dataset("PartType1/ParticleIDs",data=id_predict)
f_write.create_dataset("PartType1/mass_ratio_10",data=model_output_10)
f_write.create_dataset("PartType1/mass_ratio_100",data=model_output_100)


print('finished')
