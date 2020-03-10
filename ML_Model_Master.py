import tensorflow as tf
from tensorflow import keras
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
from keras import backend

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
            return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
        
        model = Sequential()
        model.add(LSTM(self.neurons,activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1,activation = 'sigmoid'))
        opt = tf.keras.optimizers.Adam(lr=self.learn_rate,decay=self.decay)

        model.compile(loss='mse',optimizer=opt,metrics=[rmse])
        self.model = model
        
    def _train_model(self,data):
        self.periods_to_lag = data.periods_to_lag
        self.downsample_sec = data.downsample_sec
        self.feature_columns = data.feature_columns
        self.tower = data.tower
        self.train_percent = data.train_percent
        if not hasattr(self,'model'):
            self._create_model()
        
        print(f"Training Model with Parameters:\nDownsampling = {self.downsample_sec}\nLag Periods = {self.periods_to_lag}\nActivation = {self.activation}\nNeurons = {self.neurons}\nDropout Rate = {self.dropout_rate}\nLearning Rate = {self.learn_rate}\nDecay = {self.decay}\nEpochs={self.epochs}\nBatch Size={self.batch_size}")
        self.history = self.model.fit(data.X_train,data.y_train,epochs=self.epochs,batch_size=self.batch_size,\
                                      validation_data=(data.X_test,data.y_test),verbose=1)
    
    def _fit_data(self,data):
        print(f"Fitting data from X_test")
        data.y_fit = self.model.predict(data.X_test)