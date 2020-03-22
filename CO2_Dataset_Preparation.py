import CO2_functions
import CO2_Processing
import pandas as pd
from CO2_functions import *
from CO2_Processing import *
import pickle
import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self,data_path,position_number,**kwargs):
        for key,value in kwargs.items():
            if key == 'logfile':
                self.logfile = value
        self.position_number = position_number
        self.data_path = data_path
    
    def _get_date_range(self):
        if self.position_number == 4:
            self.start_date = '2019-09-24'
            self.end_date = '2019-10-03'
        elif self.position_number == 6:
            self.start_date = '2019-11-06'
            self.end_date = '2019-11-27'
    
    def _data_retrieve(self):
        first_go = True #flag for first time through
        self._get_date_range()
        for single_date in daterange(self.start_date,self.end_date): #"daterange()" from CO2_processing
            if os.path.exists("{}/{}.pickle".format(self.data_path,single_date)):
                print_log_flush("Retrieving data for {}".format(single_date),self.logfile)
                if first_go: #open and create the data structure
                    with open('{}/{}.pickle'.format(self.data_path,single_date), 'rb') as handle:
                        data = pickle.load(handle)
                    first_go=False
                else: #append the next day to the dta structure
                    with open('{}/{}.pickle'.format(self.data_path,single_date), 'rb') as handle:
                        new_data = pickle.load(handle)
                    for key in data:
                        if key in new_data.keys():
                            data[key] = pd.concat([data[key],new_data[key]])
            else: 
                print_log_flush("No data found for {}".format(single_date),self.logfile)
                print_log_flush(os.listdir(self.data_path),self.logfile)
                continue
        return data
        
    def _preprocess(self):
        data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),self._data_retrieve()) #CO2_Processing
        print_log_flush('Removing Impulses',self.logfile)
        print_log_flush('Downsampling and Concatenating',self.logfile)
        data = downsample_and_concatenate(data) #CO2_Processing
        
        if self.position_number == 4:
            data = sept24_26_correction(data) #CO2_Processing
            data = combine_vent_data(data,1) #Combine LI_Vent and Vent_Anem_Temp into a single df by sampling rate #CO2_processing
            data['Vent_Mass'] = moving_mass_flow(data['Vent_Mass']) #Add the moving mass flow rate based on function developed.  #CO2_processing
            data['Vent_Mass'] = pd.concat([\
                                   data['Vent_Mass'].loc[(data['Vent_Mass'].index>'2019-09-24 08:57:00')&\
                                                         (data['Vent_Mass'].index<'2019-09-26 08:00:00')],\
                                   data['Vent_Mass'].loc[(data['Vent_Mass'].index>'2019-09-26 12:00:00')&\
                                                         (data['Vent_Mass'].index<'2019-10-03 13:00:00')].interpolate()])
        elif self.position_number == 6:
            #Processing for position 6
            data = combine_vent_data(data,1) #Combine LI_Vent and Vent_Anem_Temp into a single df by sampling rate 
            data['Vent_Mass'] = moving_mass_flow(data['Vent_Mass']) #Add the moving mass flow rate based on function developed. 
            for key in data:
                data[key] = pd.concat([\
                               data[key].loc[(data[key].index>'2019-11-06 00:00:00')&(data[key].index<'2019-11-25 12:00:00')],\
                               data[key].loc[(data[key].index>'2019-11-25 17:00:00')&(data[key].index<'2019-11-27 10:28:00')]])

            data['Multi'] = data['Multi'].loc[data['Multi']['CO2_3']<600]
            data['Multi'] = data['Multi'].loc[data['Multi']['Wind_Velocity']>1.0]

        self.data = data
        

class ML_Data:
    def __init__(self,feature_columns,downsample_sec,periods_to_lag,tower,train_percent):
        self.feature_columns = feature_columns
        self.downsample_sec = downsample_sec
        self.periods_to_lag = periods_to_lag
        self.tower = tower
        self.train_percent = train_percent
    
    def _prepare_and_downsample(self,data):
        #get the correct data from the tower (multi or picarro)
        self.position_number = data.position_number
        if self.tower == 'Multi':
            tower = data.data['Multi']
        elif self.tower == 'Picarro':
            tower=data.data['Picarro']
        else:
            raise NameError('tower_id must be a valid tower, either "Multi" or "Picarro"')
        vent=data.data['Vent_Mass']
        tower_proc = dwn_sample(tower,self.downsample_sec) #CO2_Processing
        vent_proc = dwn_sample(vent,self.downsample_sec) #CO2_Processing
        df = pd.concat([tower_proc,vent_proc],axis=1)
        #Concatenate and add wind speed & direction if picarro data
        if self.tower == 'Picarro':
            df = wind_add(df,'ANEM_X','ANEM_Y') #CO2_functions
        #Drop columns
        if 'm_dot' not in self.feature_columns:
            self.feature_columns.append('m_dot')
        df = df[self.feature_columns]

        #Make mass flux the last column
        loc = df.columns.get_loc('m_dot')
        cols = df.columns.tolist()
        cols = cols[0:loc]+cols[(loc+1):]+[cols[loc]]
        df = df[cols]   
        
        self.df_preprocessed = df
    
    def _ML_Process(self,data):
        if not hasattr(self,'df_preprocessed'):
            self._prepare_and_downsample(data)
        if (hasattr(data,'logfile')) & (not hasattr(self,'logfile')):
            self.logfile = data.logfile
        n_periods = self.periods_to_lag#how many periods to lag
        n_features = len(self.df_preprocessed.columns)-1#how many features exist in the feature matrix (number of cols - target col)
        time_lagged = series_to_supervised(self.df_preprocessed.dropna(),n_in=0,n_out=n_periods,dropnan=False) #CO2_Processing
        time_lagged_reframed = delete_unwanted_cols(time_lagged) #delete unneccesary columns #CO2_Processing

        
        #Make mass flux at t the last column
        loc = time_lagged_reframed.columns.get_loc('m_dot(t)')
        cols = time_lagged_reframed.columns.tolist()
        cols = cols[0:loc]+cols[(loc+1):]+[cols[loc]]
        time_lagged_reframed = time_lagged_reframed[cols]
        
        values = time_lagged_reframed.dropna().values #Convert to numpy for processing
        min_max_scalar = preprocessing.MinMaxScaler() #setup scaling
        values_scaled = min_max_scalar.fit_transform(values) #scale all values from 0 to 1
        self.min_max_scalar = min_max_scalar
        
        train_size = int(len(values)*self.train_percent) 
        train = values_scaled[:train_size,:]  #Get train/test arrays
        test = values_scaled[train_size:,:]
                
        X_train,y_train = train[:,:-1], train[:,-1] #Get feature/target arrays
        X_test, y_test = test[:,:-1], test[:,-1]
        
        self.orig_X_train_shape = X_train.shape
        self.orig_X_test_shape = X_test.shape
        self.orig_y_train_shape = y_train.shape
        self.orig_y_test_shape = y_test.shape
        
        X_train = X_train.reshape((X_train.shape[0], n_periods, n_features)) 
        X_test = X_test.reshape((X_test.shape[0], n_periods, n_features))
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test   