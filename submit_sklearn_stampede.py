#!/usr/bin/env python3

#SBATCH --job-name=run_gp
##SBATCH --partition=skx-dev    # SKX node: 48 cores, 4 GB per core, 192 GB total
#SBATCH --partition=skx-normal    # SKX node: 48 cores, 4 GB per core, 192 GB total
##SBATCH --partition=normal    ## KNL node: 64 cores x 2 FP threads, 1.6 GB per core, 96 GB total
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=1    ## OpenMP threads per MPI task
#SBATCH --time=48:00:00
#SBATCH --output=sklearn_job_%j.txt
#SBATCH --mail-user=agraus@utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140080

# system ----
import os, h5py, psutil
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import utils
# local ----

f_halo = h5py.File('halo_600.hdf5')

pos_halo = f_halo['position'][:]
mass_halo = f_halo['mass'][:]
radius_halo = f_halo['radius'][:]
vel_halo = f_halo['velocity'][:]

#identify the host                                                                                                   

host_id = np.argmax(mass_halo)
host_mass = mass_halo[host_id]
host_pos = pos_halo[host_id]
host_vel = vel_halo[host_id]

f = h5py.File('./DM_data_w_stars_training.hdf5')

dm_mass = f['PartType1']['Masses'][:]
stellar_mass = f['PartType1']['Stellar_Masses'][:]
coords = f['PartType1']['Coordinates'][:]
vel = f['PartType1']['Velocities'][:]
mass_ratio = stellar_mass/dm_mass[0]

phase_space_coords =  np.concatenate((coords-host_pos,vel-host_vel),axis=1)

non_zero_mask = (mass_ratio>0.0) #only want to train on things that are non-zero                                      

labels_non_zero = mass_ratio[non_zero_mask]
phase_space_non_zero = phase_space_coords[non_zero_mask]

f.close()

dm_mass, stellar_mass, coords, vel, mass_ratio = None,None,None,None,None

X_train, X_test, y_train, y_test = train_test_split(phase_space_non_zero, labels_non_zero,
                                                   test_size=0.2, random_state=0)

sc_x = StandardScaler()
sc_y = preprocessing.MaxAbsScaler()

sc_x.fit(X_train)
X_train = sc_x.transform(X_train)
sc_x.fit(X_test)
X_test = sc_x.transform(X_test)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
sc_y.fit(y_test)
y_test = sc_y.transform(y_test)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

#bg_gp = BaggingRegressor(base_estimator=GaussianProcessRegressor(),n_estimators=1000
#                         ,max_samples=1000,bootstrap=False,verbose=True)
#
#Okay from reading up a little I might be doing this wrong n_samples is number of
#data points (maximum is 1073770)
#So to cover all the data I need roughly 1000 estimators and 1000 samples without
#replacement
#
#Lets try increasing samples lowering estimators and adding in bootstrapping

#process = psutil.Process(os.getpid())
#print('memory usage: {}'.format(process.memory_info().rss))

bg_gp = BaggingRegressor(base_estimator=GaussianProcessRegressor(),n_estimators=800
                         ,max_samples=2500,bootstrap=True,verbose=False)

bg_gp.fit(X_train,y_train)

pred_train = bg_gp.predict(X_train)
pred_test = bg_gp.predict(X_test)

print(pred_train.shape)
print(pred_test.shape)
print(X_train.shape)
print(X_test.shape)

f_write = h5py.File('gp_data_bootstrapped_800_scaled.hdf5')
f_write.create_dataset("y_train",data=pred_train)
f_write.create_dataset("y_test",data=pred_test)
f_write.create_dataset("X_train",data=X_train)
f_write.create_dataset("X_test",data=X_test)
f_write.close()
