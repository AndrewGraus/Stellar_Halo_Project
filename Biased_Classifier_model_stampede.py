#!/usr/bin/env python3

#SBATCH --job-name=run_classifier
##SBATCH --partition=skx-dev    # SKX node: 48 cores, 4 GB per core, 192 GB total
##SBATCH --partition=skx-normal    # SKX node: 48 cores, 4 GB per core, 192 GB total
#SBATCH --partition=normal    ## KNL node: 64 cores x 2 FP threads, 1.6 GB per core, 96 GB total
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=1    ## OpenMP threads per MPI task
#SBATCH --time=48:00:00
#SBATCH --output=sklearn_job_%j.txt
#SBATCH --mail-user=agraus@utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140080

#Update Dec 10:
#I want to try to refine this to make it look less like the DM halo
#Things to do:
# 1) normalize the data
# 2) change layers 

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

print('loading data')

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

print(host_id)
print(host_mass)
print(host_pos)
print(host_vel)

f = h5py.File('DM_data_w_stars_training.hdf5')

coords = f['PartType1']['Coordinates'][:]
vel = f['PartType1']['Velocities'][:]
M_star = f['PartType1']['Stellar_Masses'][:]

#I think I need to merge coords and vel into an array

diff_coord = coords-host_pos
diff_vel = vel - host_vel

dist  = np.linalg.norm(diff_coord,axis=1)
gal_select = (dist<100.0)

diff_coord_gal = diff_coord[gal_select]
diff_vel_gal = diff_vel[gal_select]
M_star_gal = M_star[gal_select]

print(M_star_gal)

phase_space_coords =  np.concatenate((diff_coord_gal,diff_vel_gal),axis=1)

#use this handy module to splot my phase space coords into a training and test
#sample

X_train, X_test, y_train, y_test = train_test_split(phase_space_coords,M_star_gal,
                                                    test_size=0.2,random_state=102)


sc = StandardScaler()
sc_fit(X_train)
X_train = sc.transform(X_train)
sc_fit(X_test)
X_test = sc.transform(X_test)

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

print(y_train_classifier)

neg, pos = np.bincount(y_train_classifier)
initial_bias = np.log(float(pos)/float(neg))

#Now I need to build my classifier model
#So this is two layers with 64 nodes
#
#This also needs to be built to take into account very biased data
#because the data is 1% stars
#
#so in order to do that I need to change a few things from a default
#neural network
#
#First add in a dropout layer, I believe a drop out removes some
#of the nodes each epoch so the NN doesn't overfit
#
#and then add a bias to the final layer that takes in the log(pos/neg)
#where pos is the number of positive values (dm particles with associated stars)
#and neg is the dm particles without associated stars
#
#initialize the optimizer such that it has lr=12-3
#remove the from_logits=True from the BinaryCorssentropy
#
#Now I'm not sure which of these makes a difference, but the resulting 
#NN actually seems to work, so I'm going with it
#
#Oh also you need to increase the batch size becaues by default its 32(!)
#because you want to guarantee the batch has some non-zero values

def build_model(output_bias=None):
    
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]

    output_bias = tf.keras.initializers.Constant(output_bias)
    
    model = Sequential()
    model.add(Dense(7, input_dim=6, kernal_initializer='normal', activation='relu'))
    model.add(Dense(10000, activation='relu'))
    model.add(Dense(1,activation='sigmoid',bias_initializer=output_bias))

    '''model = keras.Sequential([layers.Flatten(input_shape=(6,)),
                              layers.Dense(64, activation='relu'),
                              layers.Dense(64, activation='relu'),
                              layers.Dropout(0.5),
                              layers.Dense(1, activation='sigmoid',bias_initializer=output_bias)
                              ])'''
                            

    optimizer = keras.optimizers.Adam(lr=1e-3)
    
    #Can use BinaryCrossentropy because I have only two labels
    loss  = tf.keras.losses.BinaryCrossentropy()

    model.compile(loss=loss,optimizer=optimizer,metrics=METRICS)
    
    return model

print('I guess we build this thing')
model = build_model(output_bias=initial_bias)

print('Lets train this thing')
EPOCHS = 100
BATCH_SIZE=2048

history = model.fit(X_train, y_train_classifier, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
print('Lets test this thing')

test_loss = model.evaluate(X_test,y_test_classifier,batch_size=BATCH_SIZE,verbose=2)

print(test_loss)

#now I want to save the model as an hdf5

model.save('saved_models/Classifier_biased_sigmoid_norm.h5')

print('finished')
