import tensorflow as tf
from tensorflow import keras
# import os
# os.environ['KERAS_BACKEND'] = 'theano'
# import keras


import CO2_functions
import CO2_Processing
import pandas as pd
from CO2_functions import *
from CO2_Processing import *
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os
from keras.models import model_from_json
import json
from keras import backend as K
import gc

class ML_Model_Builder:
    def __init__(self,activation,neurons,dropout_rate,learn_rate,decay,batch_size,epochs):
        self.activation = activation
        self.neurons = neurons
        self.dropout_rate = dropout_rate
        self.learn_rate = learn_rate
        self.decay = decay
        self.batch_size = batch_size
        self.epochs = epochs

    def _create_model(self):
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
        
        model = Sequential()
        model.add(LSTM(self.neurons,activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1,activation = 'sigmoid'))
        
        opt = keras.optimizers.Adam(lr=self.learn_rate,decay=self.decay)
        self.opt_string = 'tf.keras.optimizers.Adam(lr=self.learn_rate,decay=self.decay)'
        
        self.loss = 'mse'
        
        model.compile(loss=self.loss,optimizer=opt,metrics=[rmse])
        self.model = model
        
    def _train_model(self,data):
        self.position_number = data.position_number
        self.periods_to_lag = data.periods_to_lag
        self.downsample_sec = data.downsample_sec
        self.feature_columns = data.feature_columns
        self.tower = data.tower
        self.train_percent = data.train_percent
        #if (hasattr(data,'logfile')) & (not hasattr(self,'logfile')):
         #   self.logfile = data.logfile
        if not hasattr(self,'model'):
            self._create_model()
        
        print(f"Downsampling = {self.downsample_sec}\nLag Periods = {self.periods_to_lag}\
        \nactivation={self.activation}\nneurons={self.neurons}\ndropout_rate={self.dropout_rate}\
        \nlearn_rate={self.learn_rate}\ndecay={self.decay}\nbatch size={self.batch_size}\nepochs={self.epochs}")
        
        self.history = self.model.fit(data.X_train,data.y_train,epochs=self.epochs,batch_size=self.batch_size,\
                                      validation_data=(data.X_test,data.y_test),verbose=1)

        #del self.model
        #gc.collect()
        #K.clear_session()
        #tf.compat.v1.reset_default_graph() # TF graph isn't same as Keras graph
        #print(f"Final RMSE for this model: {self.history.history['rmse'][-1]}")
    
    def _del_and_clear(self):
        K.clear_session()
        #tf.reset_default_graph()
        #tf.contrib.keras.backend.clear_session()
    
    def _fit_data(self,data):
        print(f"Fitting data from X_test")
        data.y_fit = self.model.predict(data.X_test)