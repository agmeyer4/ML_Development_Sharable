#IMPORT REQUIRED PACKAGES
import tensorflow as tf
from tensorflow import keras
import CO2_functions
import CO2_Processing
import pandas as pd
from CO2_functions import *
from CO2_Processing import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os
from keras.models import model_from_json
import json
from keras.wrappers.scikit_learn import KerasRegressor

df = pd.read_pickle('PN4_10S_downsample.pkl') # This is data resampled at 100 seconds
df = df.resample('10T').mean()
#TIME LAG
df_to_use = df

n_seconds = 1 #how many periods to lag
n_features= len(df_to_use.columns)-1 #how many features exist in the feature matrix (number of cols - target col)
time_lagged = series_to_supervised(df_to_use,n_in=0,n_out=n_seconds,dropnan=False) #lag function (source CO2_Processing)
time_lagged_reframed = delete_unwanted_cols(time_lagged) #delete unneccesary columns (source CO2_Processing)


#Make mass flux at t the last column
loc = time_lagged_reframed.columns.get_loc('m_dot(t)')
cols = time_lagged_reframed.columns.tolist()
cols = cols[0:loc]+cols[(loc+1):]+[cols[loc]]
time_lagged_reframed = time_lagged_reframed[cols]

#Setup ML Architecture

values = time_lagged_reframed.dropna().values #Convert to numpy for processing
min_max_scalar = preprocessing.MinMaxScaler() #setup scaling
values_scaled = min_max_scalar.fit_transform(values) #scale all values from 0 to 1

#Set train size. Because time is a factor, we do not choose randomly, but chronologically
percent_train = 0.7
train_size = int(len(values)*percent_train) 
train = values_scaled[:train_size,:]  #Get train/test arrays
test = values_scaled[train_size:,:]

X_train,y_train = train[:,:-1], train[:,-1] #Split into feature/target arrays: target array = m_dot(t) (last column)
X_test, y_test = test[:,:-1], test[:,-1]

#Store shapes prior to 3D reshape such that they can be "unreshaped" and unscaled for representative fit/test plotting
orig_X_train_shape = X_train.shape
orig_X_test_shape = X_test.shape
orig_y_train_shape = y_train.shape
orig_y_test_shape = y_test.shape

#Reshape time lagged arrays based on number of seconds lagged: 
# X shape = (rows,lagged features,columns)
# y shape = (rows,)
X_train = X_train.reshape((X_train.shape[0], n_seconds, n_features)) 
X_test = X_test.reshape((X_test.shape[0], n_seconds, n_features))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

from model_builder import create_model

model = KerasRegressor(build_fn=create_model,verbose=10)

batch_size = [5,10,20]#,40,60,80,100,200]
epochs = [1,5,10,20]#,50,100
learn_rate = [0.001]#,0.01,0.1]
activation = ['softmax','relu']#,'tanh','sigmoid']
dropout_rate = [0.0,0.1,0.2,0.5]
neurons = [32,64]#,128]

param_grid = dict(batch_size=batch_size, epochs=epochs,learn_rate=learn_rate,activation=activation,dropout_rate=dropout_rate,neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3,verbose=10)
grid_result = grid.fit(X_train, y_train,verbose=1)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
