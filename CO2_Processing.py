# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:10:24 2020

@author: agmey
"""
def retrieve_data_from_folder(data_path,start_date,end_date):
    import pickle
    import os
    import pandas as pd 
    
    
    if start_date == 'all':
        start_date = '2019-08-15'
        end_date = '2019-11-27'
    
    first_go = True
    for single_date in daterange(start_date, end_date):
        if os.path.exists("{}/{}.pickle".format(data_path,single_date)):
            print("Retrieving data for {}".format(single_date))
            if first_go:
                with open('{}/{}.pickle'.format(data_path,single_date), 'rb') as handle:
                    data = pickle.load(handle)
                first_go=False
            else:
                with open('{}/{}.pickle'.format(data_path,single_date), 'rb') as handle:
                    new_data = pickle.load(handle)
                for key in data:
                    if key in new_data.keys():
                        data[key] = pd.concat([data[key],new_data[key]])
        else: 
            print("No data found for {}".format(single_date))
            print(os.listdir(data_path))
            continue
    return data

#==============================================================================================================#


def downsample_and_concatenate(dict_of_df):
    data = dict_of_df.copy() #hello
    return_data = {}
    if not data['Picarro_CO2'].empty:
        return_data['Picarro'] = concat_pic(data)

    if not data['Multiplexer_CO2_1'].empty:
        return_data['Multi'] = concat_multi(data)
        

    return_data['LI'] = data['LI_Vent'].drop(['EPOCH_TIME','Corrected_ET'],axis=1).set_index('Corrected_DT',drop=False).resample('1S').mean()
    
    data['Vent_Anem_Temp']['DOW'] = data['Vent_Anem_Temp']['Corrected_DT'].dt.dayofweek
    data['Vent_Anem_Temp'] = set_vent_zeros(data['Vent_Anem_Temp'])
    return_data['Vent'] = data['Vent_Anem_Temp'].drop(['EPOCH_TIME','Corrected_ET'],axis=1).set_index('Corrected_DT').resample('10S').mean()
    return_data['Vent'].interpolate(limit=1,inplace=True)
    
    
    return_data['WBB_CO2'] = data['WBB_CO2'].set_index('Corrected_DT').resample('10S').mean()
    return_data['WBB_Weather'] = data['WBB_Weather'].set_index('Corrected_DT').resample('T').mean()
    for key in return_data:
        return_data[key]['DOW'] = return_data[key].index.dayofweek

    return return_data

#==============================================================================================================#
def set_vent_zeros(vent_df):
    print('setting night vent data to zero')
    vent = vent_df.copy()
    vent.loc[(vent['Rotations']<80)&((vent.Corrected_DT.dt.hour<10)|(vent.Corrected_DT.dt.hour>17)|(vent.DOW == 5)|\
                                     (vent.DOW == 6)), ['Rotations','Velocity']]=0.0
    return vent
#==============================================================================================================#
def concat_pic(data_dict):
    ######################################################################
    # Function to concatenate all of the picarro dataframes from         #
    # a dictionary of split dataframes. Resamples at 0.1 second intervals#
    # by mean. Returns the concatenated dataframe                        #
    ######################################################################
    import pandas as pd
    print("Concatenating Picarro Data")
    co2_resample = data_dict['Picarro_CO2'].drop(['EPOCH_TIME','Corrected_ET'],axis=1)\
    .set_index('Corrected_DT',drop=False).resample('0.1S').mean() #resample from corrected data
    anem_resample = data_dict['Picarro_ANEM'].drop(['EPOCH_TIME','Corrected_ET','Pic_Loc'],axis=1)\
    .set_index('Corrected_DT',drop=False).resample('0.1S').mean() #resample from corrected data
    return pd.concat([co2_resample,anem_resample],axis=1)   #concatenate and return
#==============================================================================================================#
def concat_multi(data_dict):
    ######################################################################
    # Function to concatenate all of the multiplexer dataframes from     #
    # a dictionary of split dataframes. Resamples at 1 second intervals  #
    # by mean. Returns the concatenated dataframe                        #
    ######################################################################
    import pandas as pd
    print("Concatenating Multi Data")
    Multi1_resample = data_dict['Multiplexer_CO2_1'].drop(['EPOCH_TIME','Corrected_ET'],axis=1).set_index('Corrected_DT',drop=False).resample('1S').mean() #resample from corrected data
    Multi2_resample = data_dict['Multiplexer_CO2_2'].drop(['EPOCH_TIME','Corrected_ET','Multi_Loc'],axis=1).set_index('Corrected_DT',drop=False).resample('1S').mean()#resample from corrected data
    Multi3_resample = data_dict['Multiplexer_CO2_3'].drop(['EPOCH_TIME','Corrected_ET','Multi_Loc'],axis=1).set_index('Corrected_DT',drop=False).resample('1S').mean()#resample from corrected data
    MultiWeather_resample = data_dict['Multiplexer_Weather'].drop(['EPOCH_TIME','Corrected_ET'],axis=1).set_index('Corrected_DT',drop=False).resample('1S').mean()#resample from corrected data
    concat = pd.concat([Multi1_resample,Multi2_resample,Multi3_resample,MultiWeather_resample],axis=1)#concatenate and return
    #concat['DOW'] = concat.Datetimeindex.dayofweek
    
    return concat
#==============================================================================================================#
def find_timesteps(df):
    from collections import Counter
    data = df.copy()
    if 'Corrected_DT' in data.columns:
        diff = data['Corrected_DT']-data['Corrected_DT'].shift(1)
    else:
        diff = data.reset_index()['Corrected_DT']-data.reset_index()['Corrected_DT'].shift(1)    
    c = Counter(diff)
    return c.most_common()[0][0].total_seconds()
#==============================================================================================================#
def moving_average(df,time_window):
    data = df.copy()
    t_step = find_timesteps(data)
    roll_num = int(time_window//t_step)
    print("Applying a central moving average of {} seconds".format(time_window))
    
    if 'Corrected_DT' in data.columns:
        data.set_index('Corrected_DT',inplace=True)
    return data.rolling(roll_num,center=True).mean()
#==============================================================================================================#
def dwn_sample(df,time_window):
    data = df.copy()
    print("Downsampling by mean at {} seconds".format(time_window))

    if 'Corrected_DT' in data.columns:
        result = data.set_index('Corrected_DT').resample('{}S'.format(time_window)).mean() 
    else :
        result =  data.resample('{}S'.format(time_window)).mean() 
    return result
#==============================================================================================================#
def combine_vent_data(dict_of_dfs,sample_rate):
    import pandas as pd
    print("Combining vent data")
    data = dict_of_dfs.copy()
    
    if sample_rate < 1:
        raise ValueError('Cannot sample below 1 second, as LI data is at this rate')
    elif sample_rate == 1:
        data['Vent_Mass'] = pd.concat([data['LI'],data['Vent'].drop('DOW',axis=1).resample('{}S'.format(sample_rate)).interpolate(limit=10),data['WBB_CO2'].drop(['EPOCH_TIME'],axis=1).resample('{}S'.format(sample_rate)).interpolate(limit=10)],axis=1)
    elif sample_rate < 10:
        data['Vent_Mass'] = pd.concat([data['LI'].resample('{}S'.format(sample_rate)).mean(),data['Vent'].drop('DOW',axis=1).resample('{}S'.format(sample_rate)).interpolate(),data['WBB_CO2'].drop(['EPOCH_TIME'],axis=1).resample('{}S'.format(sample_rate)).interpolate()],axis=1) 
    else:
        data['Vent_Mass'] = pd.concat([data['LI'].resample('{}S'.format(sample_rate)).mean(),data['Vent'].drop('DOW',axis=1).resample('{}S'.format(sample_rate)).mean(),data['WBB_CO2'].drop(['EPOCH_TIME'],axis=1).resample('{}S'.format(sample_rate)).mean()],axis=1) 

        
    data['Vent_Mass']['Q'] = float('NaN')
    data['Vent_Mass'].loc[data['Vent_Mass']['Velocity']>0.0,['Q']]=2.5
    data['Vent_Mass'].loc[data['Vent_Mass']['Velocity']==0.0,['Q']]=0.0
    del data['Vent']
    del data['LI']
    return data
#==============================================================================================================#

def moving_mass_flow(concat_df):
    import pandas as pd
    print("Adding Mass Flow")
    df = concat_df.copy()
    R = 8.3145
    P = 85194.46
    T = df['Temp_1']+273
    df['Excess'] = df['LI_CO2'] - df['WBB_CO2'].interpolate()
    Excess = df['Excess']
    C_v = Excess*10**(-6)
    M_m = 44.01
    df['C_m'] = P*C_v*M_m/(R*T)
    Q = df['Q']
    df['m_dot'] = df.apply(lambda row: 0.0 if row['Q'] == 0.0 else row['Q']*row['C_m'],axis=1)
    df.drop(['C_m'],axis=1,inplace=True)
    return df
#==============================================================================================================#

def sept24_26_correction(data):
    import numpy as np
    import pandas as pd
    print("Applying vent corrections")
    cdt = ['2019-09-24 08:57:00','2019-09-24 08:57:10','2019-09-24 19:35:00','2019-09-24 19:35:10','2019-09-25 08:46:00','2019-09-25 08:46:10','2019-09-25 18:47:00',\
           '2019-09-25 18:47:10','2019-09-26 08:00:00']
    r = [0,100,100,0,0,100,100,0,0]
    v = [0,12,12,0,0,12,12,0,0]
    t1 = np.zeros(len(cdt))
    t2 = np.zeros(len(cdt))
    d = np.zeros(len(cdt))

    app = pd.DataFrame({'Corrected_DT':cdt,'Rotations':r,'Velocity':v,'Temp_1':t1,'Temp_2':t2,'DOW':d})
    app['Corrected_DT'] = pd.to_datetime(app['Corrected_DT'])
    app.set_index('Corrected_DT',inplace=True)

    app = app.resample('10S').mean().interpolate()
    
    data['Vent'] = data['Vent'].interpolate()

    data['Vent'] = pd.concat([app,data['Vent']]).resample('10S').mean()
    return data
#==============================================================================================================#
def night_vel_zeroing(dt1,dt2):
    import pandas as pd
    import numpy as np
    cdt = [dt1,dt2]
    r = np.zeros(len(cdt))
    v = np.zeros(len(cdt))
    t1 = np.zeros(len(cdt))
    t2 = np.zeros(len(cdt))
    day = np.zeros(len(cdt))
    d1 = pd.DataFrame({'Corrected_DT':cdt,'Rotations':r,'Velocity':v,'Temp_1':t1,'Temp_2':t2,'DOW':day})
    d1['Corrected_DT'] = pd.to_datetime(d1['Corrected_DT'])
    d1.set_index('Corrected_DT',inplace=True)
    d1 =d1.resample('10S').interpolate()
    
    return d1
#==============================================================================================================#
def aug15_21_correction(data):
    import numpy as np
    import pandas as pd
    print("Applying vent corrections")
    
    data['Vent'] = data['Vent'].loc[data['Vent'].index < '2019-08-21 13:00:00'] #Delete anything after 13:00 (weird spike)
    d = data['Vent'].loc[data['Vent'].index < '2019-08-15 20:00:00'] # First day

    #build night df for setting zeros
    cdt = ['2019-08-15 20:00:00','2019-08-15 20:00:10','2019-08-16 08:30:00']
    r = np.zeros(len(cdt))
    v = np.zeros(len(cdt))
    t1 = np.zeros(len(cdt))
    t2 = np.zeros(len(cdt))
    day = np.zeros(len(cdt))
    d1 = pd.DataFrame({'Corrected_DT':cdt,'Rotations':r,'Velocity':v,'Temp_1':t1,'Temp_2':t2,'DOW':day})
    d1['Corrected_DT'] = pd.to_datetime(d1['Corrected_DT'])
    d1.set_index('Corrected_DT',inplace=True)
    d1 =d1.resample('10S').interpolate()

    d = pd.concat([d,d1])

    #Second Day
    d1 = data['Vent'].loc[(data['Vent'].index > '2019-08-16 08:30:00')&(data['Vent'].index < '2019-08-16 20:00:10')]
    d = pd.concat([d,d1])

    #Builld night/weekend df
    cdt = ['2019-08-16 20:00:00','2019-08-16 20:00:10','2019-08-19 08:38:00']
    r = np.zeros(len(cdt))
    v = np.zeros(len(cdt))
    t1 = np.zeros(len(cdt))
    t2 = np.zeros(len(cdt))
    day = np.zeros(len(cdt))
    d1 = pd.DataFrame({'Corrected_DT':cdt,'Rotations':r,'Velocity':v,'Temp_1':t1,'Temp_2':t2,'DOW':day})
    d1['Corrected_DT'] = pd.to_datetime(d1['Corrected_DT'])
    d1.set_index('Corrected_DT',inplace=True)
    d1 =d1.resample('10S').interpolate()

    d = pd.concat([d,d1])

    #Third day
    d1 = data['Vent'].loc[(data['Vent'].index >= '2019-08-19 08:38:10')&(data['Vent'].index < '2019-08-19 20:00:00')]
    d = pd.concat([d,d1])

    #Another Night
    cdt = ['2019-08-19 20:00:00','2019-08-20 08:20:40']
    r = np.zeros(len(cdt))
    v = np.zeros(len(cdt))
    t1 = np.zeros(len(cdt))
    t2 = np.zeros(len(cdt))
    day = np.zeros(len(cdt))
    d1 = pd.DataFrame({'Corrected_DT':cdt,'Rotations':r,'Velocity':v,'Temp_1':t1,'Temp_2':t2,'DOW':day})
    d1['Corrected_DT'] = pd.to_datetime(d1['Corrected_DT'])
    d1.set_index('Corrected_DT',inplace=True)
    d1 =d1.resample('10S').interpolate()

    d = pd.concat([d,d1])

    #Final Day
    d1 = data['Vent'].loc[(data['Vent'].index>='2019-08-20 08:20:50')&(data['Vent'].index<'2019-08-21 20:20:00')].interpolate()

    d = pd.concat([d,d1])

    #Extra part of final day with average values
    cdt = ['2019-08-21 13:00:00','2019-08-21 18:15:00']
    r = 102.134473
    v = 10.272954
    t1 = 49.162694
    t2 = 31.979362
    day = np.zeros(len(cdt))
    d1 = pd.DataFrame({'Corrected_DT':cdt,'Rotations':r,'Velocity':v,'Temp_1':t1,'Temp_2':t2,'DOW':day})
    d1['Corrected_DT'] = pd.to_datetime(d1['Corrected_DT'])
    d1.set_index('Corrected_DT',inplace=True)
    d1 =d1.resample('10S').interpolate()

    d = pd.concat([d,d1])

    data['Vent'] = d
    data['Vent'] = data['Vent'].loc[~data['Vent'].index.duplicated(keep='first')]
    
    return data
#==============================================================================================================#
def aug28_sept12_correction(data):
    import pandas as pd
    d = data['Vent'].loc[data['Vent'].index < '2019-08-29 20:00:00']

    d1 = night_vel_zeroing('2019-08-29 20:00:00','2019-08-30 07:00:00')
    d = pd.concat([d,d1])

    d1 = data['Vent'].loc[(data['Vent'].index>'2019-08-30 07:00:00')&(data['Vent'].index<'2019-08-30 20:00:00')]
    d = pd.concat([d,d1])

    d1 = night_vel_zeroing('2019-08-30 20:00:00','2019-09-02 07:00:00')
    d = pd.concat([d,d1])

    d1 = data['Vent'].loc[(data['Vent'].index>'2019-09-02 07:00:00')&(data['Vent'].index<'2019-09-04 20:00:00')]
    d = pd.concat([d,d1])

    d1 = night_vel_zeroing('2019-09-04 20:00:00','2019-09-05 08:37:00')
    d = pd.concat([d,d1])

    d1 = data['Vent'].loc[(data['Vent'].index>'2019-09-05 08:37:00')&(data['Vent'].index<'2019-09-06 18:18:00')]
    d = pd.concat([d,d1])

    d1 = night_vel_zeroing('2019-09-06 18:18:00','2019-09-09 07:00:00')
    d = pd.concat([d,d1])

    d1 = data['Vent'].loc[(data['Vent'].index>'2019-09-09 07:00:00')&(data['Vent'].index<'2019-09-10 18:54:30')].interpolate()
    d = pd.concat([d,d1])

    d1 = night_vel_zeroing('2019-09-10 18:54:30','2019-09-11 08:44:50')
    d = pd.concat([d,d1])

    d1 = data['Vent'].loc[(data['Vent'].index>'2019-09-11 08:44:50')].interpolate()
    d = pd.concat([d,d1])

    data['Vent'] = d
    
    return data
#==============================================================================================================#
def sept12_sept23_correction(data):
    import pandas as pd
    print("Applying vent correction for PN3")
    d = data['Vent'].loc[data['Vent'].index < '2019-09-14 00:00:00']
    d1 = night_vel_zeroing('2019-09-14 00:00:00','2019-09-16 07:00:00')
    d2 = data['Vent'].loc[(data['Vent'].index > '2019-09-16 07:00:00')&(data['Vent'].index < '2019-09-16 20:00:00')]
    d3 = data['Vent'].loc[(data['Vent'].index > '2019-09-16 20:00:00')&(data['Vent'].index < '2019-09-17 18:00:00')].interpolate()
    d4 = data['Vent'].loc[(data['Vent'].index > '2019-09-17 18:00:00')&(data['Vent'].index < '2019-09-17 20:00:00')]
    d5 = night_vel_zeroing('2019-09-17 20:00:00','2019-09-18 07:00:00')
    d6 = data['Vent'].loc[(data['Vent'].index > '2019-09-18 07:00:00')&(data['Vent'].index < '2019-09-18 20:00:00')]
    d7 = night_vel_zeroing('2019-09-18 20:00:00','2019-09-19 07:00:00')
    d8 = data['Vent'].loc[data['Vent'].index>'2019-09-19 07:00:00']

    data['Vent'] = pd.concat([d,d1,d2,d3,d4,d5,d6,d7,d8])
    
    return data
#==============================================================================================================#
def oct16_nov04_correction(data):
    import pandas as pd
    d1 = data['Vent'].loc[data['Vent'].index < '2019-10-16 20:00:00']
    d2 = night_vel_zeroing('2019-10-16 20:00:00','2019-10-17 07:00:00')
    d3 = data['Vent'].loc[(data['Vent'].index > '2019-10-17 07:00:00')&(data['Vent'].index < '2019-10-18 20:00:00')]
    d4 = night_vel_zeroing('2019-10-18 20:00:00','2019-10-21 07:00:00')
    d5 = data['Vent'].loc[(data['Vent'].index > '2019-10-21 07:00:00')&(data['Vent'].index < '2019-10-22 20:00:00')]
    d6 = night_vel_zeroing('2019-10-22 20:00:00','2019-10-23 09:00:00')
    d7 = data['Vent'].loc[(data['Vent'].index > '2019-10-23 09:00:00')&(data['Vent'].index < '2019-10-24 19:00:00')].interpolate()
    d8 = data['Vent'].loc[(data['Vent'].index > '2019-10-24 19:00:00')&(data['Vent'].index < '2019-10-24 20:00:00')]
    d9 = night_vel_zeroing('2019-10-24 20:00:00','2019-10-25 07:00:00')
    d10 = data['Vent'].loc[(data['Vent'].index > '2019-10-25 07:00:00')&(data['Vent'].index < '2019-10-25 20:00:00')]
    d11 = night_vel_zeroing('2019-10-25 20:00:00','2019-10-28 07:00:00')
    d12 = data['Vent'].loc[(data['Vent'].index > '2019-10-28 07:00:00')]
    
    data['Vent'] = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12])
    return data
#==============================================================================================================#
def nov04_nov27_correction(data):
    import pandas as pd
    d1 = data['Vent'].loc[data['Vent'].index < '2019-11-06 20:00:00']
    d2 = night_vel_zeroing('2019-11-06 20:00:00','2019-11-07 07:00:00')
    d3 = data['Vent'].loc[(data['Vent'].index > '2019-11-07 07:00:00')&(data['Vent'].index < '2019-11-08 20:00:00')]
    d4 = night_vel_zeroing('2019-11-08 20:00:00','2019-11-11 07:00:00')
    d5 = data['Vent'].loc[(data['Vent'].index > '2019-11-11 07:00:00')\
                                &(data['Vent'].index < '2019-11-11 19:15:00')\
                                &((data['Vent']['Velocity']>0)|(data['Vent']['Velocity'].isnull()))]
    d6 = data['Vent'].loc[(data['Vent'].index > '2019-11-11 19:15:00')&(data['Vent'].index < '2019-11-11 20:00:00')]
    d7 = night_vel_zeroing('2019-11-11 20:00:00','2019-11-12 07:00:00')
    d8 = data['Vent'].loc[(data['Vent'].index > '2019-11-12 07:00:00')\
                                &(data['Vent'].index < '2019-11-12 12:00:00')\
                                &((data['Vent']['Velocity']>8)|(data['Vent']['Velocity'].isnull()))]
    d9 = data['Vent'].loc[(data['Vent'].index > '2019-11-12 12:00:00')&(data['Vent'].index < '2019-11-14 12:00:00')]
    d10 = data['Vent'].loc[(data['Vent'].index > '2019-11-14 12:00:00')\
                                &(data['Vent'].index < '2019-11-14 19:45:00')\
                                &((data['Vent']['Velocity']>8)|(data['Vent']['Velocity'].isnull()))]
    d11 = data['Vent'].loc[(data['Vent'].index > '2019-11-14 19:45:00')&(data['Vent'].index < '2019-11-15 20:00:00')]
    d12 = night_vel_zeroing('2019-11-15 20:00:00','2019-11-18 07:00:00')
    d13 = data['Vent'].loc[(data['Vent'].index > '2019-11-18 07:00:00')&(data['Vent'].index < '2019-11-22 20:00:00')]
    d14 = night_vel_zeroing('2019-11-22 20:00:00','2019-11-25 07:00:00')
    d15 = data['Vent'].loc[data['Vent'].index > '2019-11-25 07:00:00']


    data['Vent'] = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15])
    return data
#==============================================================================================================#
#SET UP TIME LAGGING FUNTION
#Courtesy of Jason Brownlee
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
from pandas import DataFrame
from pandas import concat
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (df.columns[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (df.columns[j], i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
#==============================================================================================================#

#We need to delete some of the columns generated by the series_to_supervised function
def delete_unwanted_cols(initial_lagged_df):
    #We want to delete all of the values of LI_CO2 EXCEPT the t=t case (very last one)
    s = "m_dot" 
    drop_cols = [] #initialize columns to be dropped
    for column in initial_lagged_df:
#         if s not in column and "(t)" in column:
#             drop_cols.append(column)     #Drop all columns except the LI_CO2 column at time t (only want the lagged variables)
        if s in column and "(t)" not in column:
            drop_cols.append(column)     #Drop all columns with LI_CO2 that arent the last one
    return initial_lagged_df.drop(drop_cols,axis = 1)  
#==============================================================================================================#
def process_for_ML_test(cols,downsample_secs,lag_secs,tower_id,pn,percent_train):
    import tensorflow as tf
    from tensorflow import keras
    import CO2_functions
    import pandas as pd
    from CO2_functions import retrieve_data_from_folder,remove_spikes,wind_add
    import matplotlib.pyplot as plt
    import pickle
    import numpy as np
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
        
    print("position number = {}".format(pn))
    if pn == 4:
        #Get the data from CHPC
        data_orig = retrieve_data_from_folder('../CO2_Data_Final','2019-09-24','2019-10-03')
        print("Removing Impulses")
        data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),data_orig) #remove impulses to prevent skew
        data = downsample_and_concatenate(data) #sample each instrument to its respective sampling rate such that everything is\
                                                #equally sampled after correcting DT
        print("Correcting data from sept 24-26")
        data = sept24_26_correction(data) #add vent vel when not monitored (instrument off)
        data = combine_vent_data(data,1) #Combine LI_Vent and Vent_Anem_Temp into a single df by sampling rate 
        data['Vent_Mass'] = moving_mass_flow(data['Vent_Mass']) #Add the moving mass flow rate based on function developed. 
        data['Vent_Mass'] = pd.concat([\
                                   data['Vent_Mass'].loc[(data['Vent_Mass'].index>'2019-09-24 08:57:00')&\
                                                         (data['Vent_Mass'].index<'2019-09-26 08:00:00')],\
                                   data['Vent_Mass'].loc[(data['Vent_Mass'].index>'2019-09-26 12:00:00')&\
                                                         (data['Vent_Mass'].index<'2019-10-03 13:00:00')].interpolate()])
    elif pn == 6:
        #Processing for position 6
        data_orig = retrieve_data_from_folder('../CO2_Data_Final','2019-11-06','2019-11-27')
        data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),data_orig) #remove impulses to prevent skew
        data = downsample_and_concatenate(data) #sample each instrument to its respective sampling rate such that everything is\
                                                #equally sampled after correcting DT
        data = combine_vent_data(data,1) #Combine LI_Vent and Vent_Anem_Temp into a single df by sampling rate 
        data['Vent_Mass'] = moving_mass_flow(data['Vent_Mass']) #Add the moving mass flow rate based on function developed. 
        for key in data:
            data[key] = pd.concat([\
                                       data[key].loc[(data[key].index>'2019-11-06 00:00:00')&\
                                                     (data[key].index<'2019-11-25 12:00:00')],\
                                       data[key].loc[(data[key].index>'2019-11-25 17:00:00')&\
                                                     (data[key].index<'2019-11-27 10:28:00')]])
    
    #get the correct data from the tower (multi or picarro)
    if tower_id == 'Multi':
        tower = data['Multi']
    elif tower_id == 'Picarro':
        tower=data['Picarro']
    else:
        raise NameError('tower_id must be a valid tower, either "Multi" or "Picarro"')
        
    vent=data['Vent_Mass']

    tower_proc = dwn_sample(tower,downsample_secs)
    vent_proc = dwn_sample(vent,downsample_secs)
    
    df = pd.concat([tower_proc,vent_proc],axis=1)

    #Concatenate and add wind speed & direction if picarro data
    if tower_id == 'Picarro':
        df = wind_add(df,'ANEM_X','ANEM_Y')

    #Drop columns
    if 'm_dot' not in cols:
        cols.append('m_dot')
    df = df[cols]
    
    #Make mass flux the last column
    loc = df.columns.get_loc('m_dot')
    cols = df.columns.tolist()
    cols = cols[0:loc]+cols[(loc+1):]+[cols[loc]]
    df = df[cols]
    

    #TIME LAG
    df_to_use = df

    n_seconds = 10 #how many periods to lag
    n_features= len(df_to_use.columns)-1 #how many features exist in the feature matrix (number of cols - target col)
    time_lagged = series_to_supervised(df_to_use,n_in=0,n_out=n_seconds,dropnan=False) #lag function
    time_lagged_reframed = delete_unwanted_cols(time_lagged) #delete unneccesary columns

    #Make mass flux at t the last column
    loc = time_lagged_reframed.columns.get_loc('m_dot(t)')
    cols = time_lagged_reframed.columns.tolist()
    cols = cols[0:loc]+cols[(loc+1):]+[cols[loc]]
    time_lagged_reframed = time_lagged_reframed[cols]
    print("columns fed to numpy: ",time_lagged_reframed.columns)
    
    values = time_lagged_reframed.dropna().values #Convert to numpy for processing
    min_max_scalar = preprocessing.MinMaxScaler() #setup scaling
    values_scaled = min_max_scalar.fit_transform(values) #scale all values from 0 to 1

    #Set train size. Because time is a factor, we do not choose randomly, but chronologically
    print("Train/test split: {}%".format(percent_train*100))
    train_size = int(len(values)*percent_train) 
    train = values_scaled[:train_size,:]  #Get train/test arrays
    test = values_scaled[train_size:,:]

    X_train,y_train = train[:,:-1], train[:,-1] #Get feature/target arrays
    X_test, y_test = test[:,:-1], test[:,-1]
    
    #Store shapes prior to 3D reshape such that they can be "unreshaped" and unscaled for representative fit/test plotting
    orig_X_train_shape = X_train.shape
    orig_X_test_shape = X_test.shape
    orig_y_train_shape = y_train.shape
    orig_y_test_shape = y_test.shape
    print("Shapes prior to 3d reshape: \nX_train = {}\nX_test = {}\ny_train = {}\ny_test =\
    {}".format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
    
    X_train = X_train.reshape((X_train.shape[0], n_seconds, n_features)) 
    X_test = X_test.reshape((X_test.shape[0], n_seconds, n_features))

    return X_train,X_test,y_train,y_test,min_max_scalar,orig_X_train_shape,orig_X_test_shape,orig_y_train_shape,orig_y_test_shape

#################################################################################
def print_log_flush(string,logfile):
    print(string,flush=True)
    if logfile is not None:
        logfile.write(string+'\n')
        logfile.flush()

        
##########################################        
def hampel_filter(data_dict,key_dict):
    import numba
    from numba import jit
    import numpy as np
    import pandas as pd
    @jit(nopython=True)
    def hampel_filter_forloop_numba(input_series, window_size, n_sigmas=3):
        #Taken from https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d

        n = len(input_series)
        new_series = input_series.copy()
        k = 1.4826 # scale factor for Gaussian distribution
        indices = []

        # possibly use np.nanmedian 
        for i in range((window_size),(n - window_size)):
            x0 = np.median(input_series[(i - window_size):(i + window_size)])
            S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
            if (np.abs(input_series[i] - x0) > n_sigmas * S0):
                new_series[i] = x0
                indices.append(i)

        return new_series, indices
       
    
    for key,var in key_dict.items():
        if key not in data_dict:
            continue
        df = data_dict[key]
        allcols = df.columns
        filter_cols = var[0] 
        window = var[1]
        sigma = var[2]
        first_go = True
        for col in allcols:
            print(col)
            if col in filter_cols:
                filtered,ind = hampel_filter_forloop_numba(np.array(df[col]),window,n_sigmas=sigma)
                if first_go:
                    new_df = df[col].reset_index().drop(ind).set_index('Corrected_DT')
                    first_go = False
                else:
                    new_col = df[col].reset_index().drop(ind).set_index('Corrected_DT')
                    new_df = pd.concat([new_df,new_col],axis=1)
                print(f"Column {col} filtered via hampel. Removed {len(ind)} rows of data")
            else:
                if first_go:
                    new_df = df[col]
                    first_go = False
                else:
                    new_col = df[col]
                    new_df = pd.concat([new_df,new_col],axis=1)
        data_dict[key] = new_df
    return data_dict
         

    
    
    
    