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
        if self.position_number == 1:
            self.start_date = '2019-08-15'
            self.end_date = '2019-08-21'            
        elif self.position_number == 2: 
            self.start_date = '2019-08-28'
            self.end_date = '2019-09-12'
        elif self.position_number == 3:
            self.start_date = '2019-09-12'
            self.end_date = '2019-09-23'
        elif self.position_number == 4:
            self.start_date = '2019-09-24'
            self.end_date = '2019-10-03'
        elif self.position_number == 5:
            self.start_date = '2019-10-16'
            self.end_date = '2019-11-04'
        elif self.position_number == 6:
            self.start_date = '2019-11-06'
            self.end_date = '2019-11-27'
        elif self.position_number == 'all':
            self.start_date = '2019-08-15'
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
        self.data = data
        
    def _hard_filter(self,df_key,filter_dict):
        print(f"Hard Filtering {df_key}")
        if  not hasattr(self,'filters_applied'):
            self.filters_applied = {}
        self.filters_applied['hard_filter'] = [df_key,filter_dict]
        df = self.data[df_key].copy()
        filter_cols = filter_dict.keys()
        allcols = df.columns
        new_df = df.reset_index()[['Corrected_DT']]
        for col in allcols:
            if col in filter_cols:
                print(f"Binding {col} between {filter_dict[col][0]} and {filter_dict[col][1]}")
                d = df.reset_index()[col].dropna()
                mask = (d<filter_dict[col][0])|(d>filter_dict[col][1])
                drop_col = d[mask].index

                new_col = df.reset_index()[col]
                new_col = new_col.drop(drop_col)
                new_df = pd.concat([new_df,new_col],axis = 1)
            else:
                new_col = df.reset_index()[col]
                new_df = pd.concat([new_df,new_col],axis = 1)

        self.data[df_key] = new_df.set_index('Corrected_DT',drop=True)
        
    def _median_filter(self,df_key,filter_dict):
        def med_mask(data_col,window,sigma):
            base = data_col.rolling(window).median()
            noise = data_col-base
            thresh = sigma*np.std(noise)
            mask = np.abs(noise)>thresh
            return mask
        print(f"Median Filtering {df_key}") 
        if  not hasattr(self,'filters_applied'):
            self.filters_applied = {}
        self.filters_applied['median_filter'] = [df_key,filter_dict]
        df = self.data[df_key].copy()
        filter_cols = filter_dict.keys()
        allcols = df.columns
        new_df = df.reset_index()[['Corrected_DT']]

        for col in allcols:
            if col in filter_cols:
                window = filter_dict[col][0]
                sigma = filter_dict[col][1]
                print(f"Filtering {col} with window {window} and sigma {sigma}")
                d = df.reset_index()[col].dropna()
                mask = med_mask(d,window,sigma)
                drop_col = d[mask].index

                new_col = df.reset_index()[col]
                new_col = new_col.drop(drop_col)
                new_df = pd.concat([new_df,new_col],axis = 1)
            else:
                new_col = df.reset_index()[col]
                new_df = pd.concat([new_df,new_col],axis = 1)

        self.data[df_key] = new_df.set_index('Corrected_DT',drop=True)
    
    def _apply_hampel_filter(self,df_key_array):
        print(f"Applying Hampel Filter to {df_key_array}")
        self.data = hampel_filter(self.data,df_key_array)
        
    def _remove_low_wind(self,df_key_array,min_vel):
        for key in df_key_array:
            if key == 'Multi':
                data[key] = data[key].loc[data[key]['Wind_Velocity']>min_vel]
            elif key == 'Picarro':
                data[key] = data[key].loc[data[key]['ws']>min_vel]
                
    def _preprocess(self):
        self._data_retrieve()

        
        if self.position_number == 1:
            self.data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),self.data) #CO2_Processing
            self.data = downsample_and_concatenate(self.data) #CO2_Processing
            self.data = aug15_21_correction(self.data)            
            self.data = combine_vent_data(self.data,1)
            self.data['Vent_Mass'] = moving_mass_flow(self.data['Vent_Mass'])
            filter_dict = {'CO2_1':[390,650],'CO2_2':[390,650],'CO2_3':[390,650],'Temp':[20,100]}
            self._hard_filter('Multi',filter_dict)
        elif self.position_number == 2:
            self.data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),self.data) #CO2_Processing
            self.data = downsample_and_concatenate(self.data) #CO2_Processing
            self.data = aug28_sept12_correction(self.data)
            self.data = combine_vent_data(self.data,1)
            self.data['Vent_Mass'] = moving_mass_flow(self.data['Vent_Mass'])
            filter_dict = {'CO2_1':[390,600],'CO2_2':[390,600],'CO2_3':[390,600],'Rotations':[0,20],'Wind_Velocity':[0,20]}
            self._hard_filter('Multi',filter_dict)
            filter_dict = {'Temp':[100,5]}
            self._median_filter('Multi',filter_dict)
            filter_dict = {'ANEM_X':[-20,20],'ANEM_Y':[-20,20],'ANEM_Z':[-10,15]}
            self._hard_filter('Picarro',filter_dict)
        elif self.position_number == 3:
            self.data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),self.data) #CO2_Processing
            self.data = downsample_and_concatenate(self.data) #CO2_Processing
            self.data = sept12_sept23_correction(self.data)
            self.data = combine_vent_data(self.data,1)
            self.data['Vent_Mass'] = moving_mass_flow(self.data['Vent_Mass'])
            filter_dict = {'Pic_CO2':[380,700],'ANEM_X':[-10,20],'ANEM_Y':[-15,20],'ANEM_Z':[-10,10]}
            self._hard_filter('Picarro',filter_dict)
            filter_dict = {'CO2_1':[390,550],'CO2_2':[390,550],'CO2_3':[390,550]}
            self._hard_filter('Multi',filter_dict)
            filter_dict = {'Temp':[100,5]}
            self._median_filter('Multi',filter_dict)
        elif self.position_number == 4:
            self.data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),self.data) #CO2_Processing
            self.data = downsample_and_concatenate(self.data) #CO2_Processing
            self.data = sept24_26_correction(self.data) #CO2_Processing
            self.data = combine_vent_data(self.data,1) #Combine LI_Vent and Vent_Anem_Temp into a single df by sampling rate 
            self.data['Vent_Mass'] = moving_mass_flow(self.data['Vent_Mass']) #Add the moving mass flow rate  
            filter_dict = {'Pic_CH4':[0,15],'ANEM_X':[-10,20],'ANEM_Y':[-20,20]}
            self._hard_filter('Picarro',filter_dict)
            filter_dict = {'CO2_3':[390,800]}
            self._hard_filter('Multi',filter_dict)
        elif self.position_number == 5:
            self.data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),self.data) #CO2_Processing
            self.data = downsample_and_concatenate(self.data) #CO2_Processing
            self.data = oct16_nov04_correction(self.data)
            self.data = combine_vent_data(self.data,1) #Combine LI_Vent and Vent_Anem_Temp into a single df by sampling rate 
            self.data['Vent_Mass'] = moving_mass_flow(self.data['Vent_Mass']) #Add the moving mass flow rate  
            filter_dict = {'m_dot':[-1,15]}
            self._hard_filter('Vent_Mass',filter_dict)
            filter_dict = {'CO2_2':[390,700]}
            self._hard_filter('Multi',filter_dict)
            filter_dict = {'Pic_CO2':[380,800],'Pic_CH4':[0,4],'ANEM_Z':[-10,20]}
            self._hard_filter('Picarro',filter_dict)
            #Tubes were switched, switch back
            d1 = self.data['Multi'].loc[self.data['Multi'].index < '2019 10-29 00:00:00'].rename(columns = {'CO2_1':'CO2_3','CO2_3':'CO2_1'})
            d2 = self.data['Multi'].loc[self.data['Multi'].index > '2019 10-29 00:00:00']
            self.data['Multi'] = pd.concat([d1,d2])
            
        elif self.position_number == 6:
            self.data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),self.data) #CO2_Processing
            self.data = downsample_and_concatenate(self.data) #CO2_Processing
            self.data = nov04_nov27_correction(self.data)
            self.data = combine_vent_data(self.data,1)
            self.data['Vent_Mass'] = moving_mass_flow(self.data['Vent_Mass'])
            filter_dict = {'m_dot':[-1,15]}
            self._hard_filter('Vent_Mass',filter_dict)
            filter_dict = {'CO2_1':[390,650],'CO2_2':[390,700],'CO2_3':[390,650],'Temp':[-20,100]}
            self._hard_filter('Multi',filter_dict)
            filter_dict = {'Pic_CO2':[380,800]}
            self._hard_filter('Picarro',filter_dict)
            self.data['Picarro'] = self.data['Picarro'].loc[(self.data['Picarro'].index < '2019-11-25 12:00:00')|(self.data['Picarro'].index > '2019-11-25 17:00:00')].resample("0.1S").mean()
        elif self.position_number == 10:
            data = combine_vent_data(data,1) #Combine LI_Vent and Vent_Anem_Temp into a single df by sampling rate 
            data['Vent_Mass'] = moving_mass_flow(data['Vent_Mass']) #Add the moving mass flow rate based on function developed. 
            for key in data:
                data[key] = data[key].loc[(data[key].index>'2019-09-11 10:00:00')&(data[key].index<'2019-09-11 14:00:00')]

class Processed_Set:
    def __init__(self,tower,position_number,excess_rolls,**kwargs):
        self.position_number = position_number
        #self.vent_bool = vent_bool
        self.tower = tower
        self.excess_rolls = excess_rolls
        for key,value in kwargs.items():
            if key == 'vent_bool':
                self.vent_bool = value
            elif key == 'wbb_bool':
                self.wbb_bool = value
        if self.tower == 'Multi':
            if self.position_number == 1:
                self.date_ranges = {0:['2019-08-15','2019-08-21'],1:['2019-10-22','2019-10-30'],2:['2019-11-06','2019-11-27']}
            if self.position_number == 2:
                self.date_ranges = {0:['2019-08-29','2019-09-19']}
            if self.position_number == 3:
                self.date_ranges = {0:['2019-09-30','2019-10-03']}
        elif self.tower == 'Picarro':
            if self.position_number == 1:
                self.date_ranges = {0:['2019-08-15','2019-08-21']}
            if self.position_number == 2:
                self.date_ranges = {0:['2019-08-28','2019-09-12']}
            if self.position_number == 3:
                self.date_ranges = {0:['2019-09-12','2019-09-23']}    
            if self.position_number == 4:
                self.date_ranges = {0:['2019-09-24','2019-10-03']}  
            if self.position_number == 5:
                self.date_ranges = {0:['2019-10-16','2019-11-04']}  
            if self.position_number == 6:
                self.date_ranges = {0:['2019-11-05','2019-11-27']}  
        else: 
            raise NameError('Tower must be "Picarro" or "Multi')
    
    def _retrieve_data(self,data_path):
        firstgo = True
        self.data_path = data_path
        for i in range(0,len(self.date_ranges)):
            for date in daterange(self.date_ranges[i][0],self.date_ranges[i][1]):
                if firstgo:
                    with open(f'{data_path}/{self.tower}/{date}_PN{self.position_number}.pkl','rb') as handle:
                        tower_df = pickle.load(handle)
                    if self.vent_bool:
                        with open(f'{data_path}/Vent/{date}.pkl','rb') as handle:
                            vent_df = pickle.load(handle)
                    if self.wbb_bool:
                        with open(f'{data_path}/WBB_Weather/{date}.pkl','rb') as handle:
                            wbb_df = pickle.load(handle)
                    firstgo=False
                else:
                    with open(f'{data_path}/{self.tower}/{date}_PN{self.position_number}.pkl','rb') as handle:
                        tower_df = pd.concat([tower_df,pickle.load(handle)])
                    if self.vent_bool:
                        with open(f'{data_path}/Vent/{date}.pkl','rb') as handle:
                            vent_df = pd.concat([vent_df,pickle.load(handle)])
                    if self.wbb_bool:
                        with open(f'{data_path}/WBB_Weather/{date}.pkl','rb') as handle:
                            wbb_df = pd.concat([wbb_df,pickle.load(handle)])
        if self.tower == 'Multi':
            tower_df = multi_direction_correction(tower_df)
            tower_df.rename(columns={'Wind_Velocity':'ws','Wind_Direction':'wd'},inplace=True)

        self.data = {f'{self.tower}':tower_df}
        if self.vent_bool:
            self.data['Vent_Mass'] = vent_df
        if self.wbb_bool:
            self.data['WBB_Weather'] = wbb_df
        
    def _apply_excess(self):
        print(f"Applying excess using minimum on windows: {self.excess_rolls}")
        for roll in self.excess_rolls:
            if self.tower=='Picarro':
                pollutant_cols = ['Pic_CO2','Pic_CH4']
                self.save_cols = ['Pic_CO2','Pic_CH4','Pic_Loc','ANEM_X','ANEM_Y','ANEM_Z']
            else:
                pollutant_cols = ['CO2_1','CO2_2','CO2_3']
                self.save_cols = ['CO2_1', 'CO2_2', 'CO2_3', 'Temp', 'ws', 'wd']
            for col in pollutant_cols:
                self.data[self.tower][f'min_r{roll}_{col}'] = self.data[self.tower][col].rolling(roll,center=True,min_periods=1).min()

        
        for roll in self.excess_rolls:
            for col in pollutant_cols:
                self.data[self.tower][f'excess_r{roll}_{col}'] = self.data[self.tower][col]-self.data[self.tower][f'min_r{roll}_{col}']
                self.save_cols.append(f'excess_r{roll}_{col}')
                #self.feature_columns.append(f'excess_r{roll}_{col}')

        self.data[self.tower] = self.data[self.tower][self.save_cols]
    
    def _combine_vent_tower(self,downsample):
        print("combining vent and tower data into a dataframe")
        if downsample == 0:
            self.df = self.data[self.tower]
        else:
            tower_proc = dwn_sample(self.data[self.tower],downsample)
            vent_proc = dwn_sample(self.data['Vent_Mass'],downsample)
            self.df = pd.concat([tower_proc,vent_proc['m_dot']],axis=1)
        
    def _add_rolling_wind(self,rolls,delete_anem_bool):
        print(f"rolling wind with {rolls} size windows") 
        if self.tower != 'Picarro':
            raise NameError('Tower must be Picarro for this operation')
        
        for col in ['ANEM_X','ANEM_Y']:
            for roll in rolls:
                self.df[f'roll_{roll}_{col}'] = self.df[col].rolling(roll,center=True,min_periods=1).mean()

        w = wind_add(self.df,'ANEM_X','ANEM_Y').copy()[['ws','wd']]
        self.df[f'ws'] = w['ws']
        self.df[f'wd'] = w['wd']

        for roll in rolls:
            print(f'Wind Add with {roll} rolled timesteps')
            w = wind_add(self.df,f'roll_{roll}_ANEM_X',f'roll_{roll}_ANEM_Y').copy()[['ws','wd']]
            self.df[f'roll_{roll}_ws'] = w['ws']
            self.df[f'roll_{roll}_wd'] = w['wd']
        
        if delete_anem_bool:
            if self.tower == 'Picarro':
                self.df = self.df.filter(regex='ws|wd|Pic')
            elif self.tower =='Multi':
                self.df = self.df.filter(regex='ws|wd|CO2')
            
            
    def _column_shifter(self,shift_list,**kwargs):
        print(f"shifting wind columns by {shift_list}")
        if self.tower == 'Picarro':
            cols = self.df.columns.drop(self.df.filter(regex='Pic').columns)
        elif self.tower=='Multi':
            cols = self.df.columns.drop(self.df.filter(regex='CO2').columns)  
        for key,value in kwargs.items():
            if key == 'delete':
                del_col = value
            else:
                del_col = False
        
        for shift_num in shift_list:
            for col in cols:
                self.df[f'{col}(t-{shift_num})'] = self.df[col].shift(periods=shift_num)    
        if del_col:
            self.df.drop(cols,axis=1,inplace=True)

    def _vent_on_only(self):
        self.df = self.df.loc[self.df['m_dot']>0]


        
#######========================================================================================================####################
            
class ML_Data:
    def __init__(self,downsample_sec,periods_to_lag,tower,train_percent,**kwargs):
        for key,value in kwargs.items():
            if key == 'feature_columns':
                self.feature_columns = value
        #self.feature_columns = feature_columns
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
        
        if not hasattr(self,'feature_columns'):        
            self.feature_columns = data.save_cols.copy()
        if 'ws' not in self.feature_columns:
            self.feature_columns.append('ws')
        elif 'wd' not in self.feature_columns:
            self.feature_columns.append('wd')
        
        #Drop columns
        self.feature_and_target = self.feature_columns.copy()
        if 'm_dot' not in self.feature_and_target:
            self.feature_and_target.append('m_dot')
        df = df[self.feature_and_target]

        #Make mass flux the last column
        loc = df.columns.get_loc('m_dot')
        cols = df.columns.tolist()
        cols = cols[0:loc]+cols[(loc+1):]+[cols[loc]]
        df = df[cols]   
        
        df = df.drop(df.filter(regex='Loc').columns,axis=1)
        self.df_preprocessed = df
    
    def _ML_Process(self,data,**kwargs):
        for key,value in kwargs.items():
            if key == 'lag_how':
                self.lag_how = value
        
        if not hasattr(self,'df_preprocessed'):
            self._prepare_and_downsample(data)
        if (hasattr(data,'logfile')) & (not hasattr(self,'logfile')):
            self.logfile = data.logfile
        n_periods = self.periods_to_lag#how many periods to lag
        n_features = len(self.df_preprocessed.columns)-1#how many features exist in the feature matrix (number of cols - target col)
        
        
#         if self.lag_how == 'forward':
#             time_lagged = series_to_supervised(self.df_preprocessed.dropna(),n_in=n_periods,n_out=1,dropnan=False) #CO2_Processing
#             period_reshape = n_periods+1
#         elif self.lag_how == 'backward':
        time_lagged = series_to_supervised(self.df_preprocessed.dropna(),n_in=0,n_out=n_periods,dropnan=False) #CO2_Processing
        period_reshape = n_periods
#         elif self.lag_how == 'both':
#             time_lagged = series_to_supervised(self.df_preprocessed.dropna(),n_in=n_periods,n_out=n_periods,dropnan=False) 
#             period_reshape = n_periods*2

        
        time_lagged_reframed = delete_unwanted_cols(time_lagged) #delete unneccesary columns #CO2_Processing

        
        #Make mass flux at t the last column
        loc = time_lagged_reframed.columns.get_loc('m_dot(t)')
        cols = time_lagged_reframed.columns.tolist()
        cols = cols[0:loc]+cols[(loc+1):]+[cols[loc]]
        time_lagged_reframed = time_lagged_reframed[cols]
        self.time_lagged_reframed = time_lagged_reframed
        
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
        
        X_train = X_train.reshape((X_train.shape[0], period_reshape, n_features)) 
        X_test = X_test.reshape((X_test.shape[0], period_reshape, n_features))
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test   