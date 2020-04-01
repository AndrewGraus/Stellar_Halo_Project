#Now I need to agree on a machine learning algorithm to use
#I think I can just use gaussian process but this is to test to 
#see if this gives us something reasonable

import numpy as np
import h5py
from sklearn.gaussian_process import GaussianProcess

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


