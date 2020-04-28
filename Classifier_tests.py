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

#Because regressors assume a continuous output It cannot correctly handle
#a continuous distribution with zeros mixed in
#
#I think what I need to do is first tran a classifier, which will classify
#zeros and non-zeros, and then after that is working take the non-zero
#values and train a regressor to get the predicted mass fractions
#
#This program is to test that first part.

import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from subprocess import call

#This Classifier requires the use of tf2 which is not natively implemented
#in any of the Pleiades modules, so I need to load conda and then
#switch to the tf2 environment and then unload it at the end
#
#my attempt to use subprocess call to load the tf2 environ seems to not
#work because call doesn't recognize module. so I need to simply make
#a submission script for this and run it that way
#
#I'll leave the code blocks here for refernce in case I come back to this
#method

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

print('running tf')

#For training a classifier I just need an interger based on the number
#of labels, since I have two labels (zero and non-zero) so I just need
#to take every zero value and assign it zero, and every non-zero
#value and assign it one

#first do a mask that is y_train !=0 to give an array of TRUE FALSE
#where 0.0 is false and > 0.0 is true. Then convert the True-False
#to 0 and 1

y_train_classifier = np.array((y_train!=0),dtype=int)
y_test_classifier  = np.array((y_test!=0),dtype=int)

#Now I need to build my classifier model
#So this is two layers with 64 nodes the 3rd layer should return
#just one number between 0 and 1, but its not going to be EXACTLY
#zero or 1 it seems to represent a probablility the more you
#train it the sharper the transiton between zero and 1 should be
#maybe?
def build_model():
    #model = keras.Sequential([layers.Dense(64, activation='relu', 
    #                        input_shape=[len(X_train[0])]),
    #                          layers.Dense(64, activation='relu'),
    #                          layers.Dense(2)
    #                         ])
    
    model = keras.Sequential([layers.Flatten(input_shape=(6,)),
                              layers.Dense(64, activation='relu'),
                              layers.Dense(64, activation='relu'),
                              layers.Dense(1, activation='sigmoid')
                              ])
                            

    optimizer = 'adam'
    
    #Can use BinaryCrossentropy because I have only two labels
    loss  = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
    
    return model

print('I guess we build this thing')
model = build_model()

print('Lets train this thing')
EPOCHS = 5

print(X_train.shape, y_train_classifier.shape)

print(y_train_classifier)

history = model.fit(X_train, y_train_classifier, epochs=EPOCHS, verbose=1)
print('Lets test this thing')

test_loss, test_acc = model.evaluate(X_test,y_test_classifier,verbose=2)

#now I want to save the model as an hdf5

model.save('saved_models/Classifier_test_sigmoid.h5')

print('finished')
