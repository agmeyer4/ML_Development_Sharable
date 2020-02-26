# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#=========================================================================================================#
def get_date_range():
    #Ask the user for a date range and return the results
    date1=input("Enter Start Date YYYY-mm-DD: ")
    date2=input("Enter End Date YYYY-mm-DD: ") 
    return date1,date2

#=========================================================================================================#
    
def sql_connect():
    ######################################################################
    # Function to get exclusively "Spike Necessary" LI_8100 Data         # 
    # from SQL. Input the SQL Tablename, and date range between which    #
    # data will be fetched. For one day's worth of data, enter the same  #
    # Date.                                                              #
    ######################################################################
    import pymysql.cursors
    
    #Connect to SQL database with username and pw
    mydb = pymysql.connect(
        host='155.98.6.253',
        user='EddyFlux',
        passwd = 'UvTrhM_yFo71X2',
        database = 'CO2_Eddy'
        )
    
    #Set up cursor (allows navigation through SQL syntax)
    mycursor = mydb.cursor()
    
    return mycursor

#=========================================================================================================#

def get_LI_data(tablename,date1,date2):
    ######################################################################
    # Function to get necessary LI_8100 Data                             # 
    # from SQL. Input the SQL Tablename, and date range between which    #
    # data will be fetched. For one day's worth of data, enter the same  #
    # Date.                                                              #
    # Inputs:   tablename = name of SQL table to query, input as string  #
    #           date1 = start data for query, as string                  #
    #           date2 = end date for query, as string                    #
    ######################################################################
    import pandas as pd

    mycursor = sql_connect()
    mycursor.execute("SELECT Local_DT, EPOCH_TIME, Cdry\
                        FROM {}\
                        WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                        order by EPOCH_TIME asc;".format(tablename,date1,date2)) #SQL statement
    data = mycursor.fetchall() #fetch the data
    LI_vent = pd.DataFrame(list(data)) #convert imported data to dataframe
    LI_vent.columns = ['Local_DT','EPOCH_TIME','LI_CO2'] #name columns
    cols = LI_vent.columns.drop('Local_DT') #get all column names beside date column
    LI_vent[cols]=LI_vent[cols].apply(pd.to_numeric,errors='coerce') #change all but date to floats
    
    return LI_vent
#=========================================================================================================#
    
def get_multiplexer_data(tablename,date1,date2,split_or_concat,i):
    ######################################################################
    # Function to get exclusively "Spike Necessary" Multiplexer Data     # 
    # from SQL. Input the SQL Tablename, and date range between which    #
    # data will be fetched. For one day's worth of data, enter the same  #
    # Date.                                                              #
    ######################################################################
    
    import pandas as pd
    
    #Connect to SQL
    mycursor = sql_connect()   
    
    if split_or_concat == 'split':
        if i < 3:
            mycursor.execute("SELECT Local_DT,EPOCH_TIME,CO2_{},Location_Multi\
                        FROM {}\
                        WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                        order by EPOCH_TIME asc;".format(i,tablename,date1,date2)) #SQL query
            data = mycursor.fetchall() #fetch the data
            Multiplexer = pd.DataFrame(list(data)) #convert imported data into a dataframe
            Multiplexer.columns = ['Local_DT','EPOCH_TIME','CO2_{}'.format(i),'Multi_Loc'] #name columns
        elif i == 3:
            mycursor.execute("SELECT Local_DT,EPOCH_TIME,CO2_{},Rotations, Wind_Velocity,Wind_Direction,Temp,Location_Multi\
                        FROM {}\
                        WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                        order by EPOCH_TIME asc;".format(i,tablename,date1,date2)) #SQL query
            data = mycursor.fetchall() #fetch the data
            Multiplexer = pd.DataFrame(list(data)) #convert imported data into a dataframe
            Multiplexer.columns = ['Local_DT','EPOCH_TIME','CO2_{}'.format(i),'Rotations','Wind_Velocity','Wind_Direction','Temp','Multi_Loc'] #name columns
    elif split_or_concat == 'concat':
        mycursor.execute("SELECT *\
                    FROM {}\
                    WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                    order by EPOCH_TIME asc;".format(tablename,date1,date2)) #SQL query
        data = mycursor.fetchall() #fetch the data
        Multiplexer = pd.DataFrame(list(data)) #convert imported data into a dataframe
        Multiplexer.columns = ['Local_DT','EPOCH_TIME','CO2_1','CO2_2','CO2_3','Rotations','Wind_Velocity','Wind_Direction','Temp','Multi_loc'] #name columns
    else:
        raise ValueError('Input "split" or "concat" as the last argument')
        
    
    
    cols = Multiplexer.columns.drop('Local_DT') #get all column names but date column
    Multiplexer[cols]=Multiplexer[cols].apply(pd.to_numeric,errors='coerce') #change all but date to floats
    
    return Multiplexer
#=========================================================================================================#
    
def get_vent_anem_temp_data(tablename,date1,date2):
    ######################################################################
    # Function to get exclusively "Spike Necessary" Vent_Anem_Temp Data  # 
    # from SQL. Input the SQL Tablename, and date range between which    #
    # data will be fetched. For one day's worth of data, enter the same  #
    # Date.                                                              #
    ######################################################################
    import pandas as pd
   
    #Connect to SQL
    mycursor = sql_connect()
    mycursor.execute("SELECT *\
                    FROM {}\
                    WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                    order by EPOCH_TIME asc;".format(tablename,date1,date2)) #SQL query
    data = mycursor.fetchall() #fetch the data
    Vent_Anem_Temp = pd.DataFrame(list(data)) #convert imported data to a dataframe
    Vent_Anem_Temp.columns = ['Local_DT','EPOCH_TIME','Rotations','Velocity','Temp_1','Temp_2'] #name columns
    cols = Vent_Anem_Temp.columns.drop('Local_DT') #get all column names but date
    Vent_Anem_Temp[cols]=Vent_Anem_Temp[cols].apply(pd.to_numeric,errors='coerce') #change all but date to floats
    
    return Vent_Anem_Temp
#=========================================================================================================#
    
def get_wbb_weather(date1,date2):
    ######################################################################
    # Function to get WBB weather data                                   #
    ######################################################################
    import pandas as pd
   
    #Connect to SQL
    mycursor = sql_connect()
    mycursor.execute("SELECT Date_Time,air_temp_set_1,wind_speed_set_1,wind_direction_set_1,wind_cardinal_direction_set_1d\
                        FROM Aug2019_WBB_Weather\
                        WHERE Date_Time >= '{} 00:00:00' AND Date_Time <= '{} 23:59:59.99'\
                        ORDER BY Date_Time asc;".format(date1,date2))
    x = mycursor.fetchall()
    WBB_weather = pd.DataFrame(x)
    WBB_weather.columns = ['Corrected_DT','Temp','ws','wd','wcd']
    cols = WBB_weather.columns.drop(['Corrected_DT','wcd'])
    WBB_weather[cols]=WBB_weather[cols].apply(pd.to_numeric,errors='coerce')
    
    return WBB_weather
#=========================================================================================================#
    
def get_wbb_co2(date1,date2):
    ######################################################################
    # Function to get WBB weather data                                   #
    ######################################################################
    import pandas as pd
    mycursor = sql_connect()
    mycursor.execute("SELECT EPOCH_TIME, Local_DT, CO2d_ppm_cal, CH4d_ppm_cal\
                        FROM Aug2019_WBB_CO2\
                        WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                        ORDER BY EPOCH_TIME ASC;".format(date1,date2))
    x = mycursor.fetchall()
    WBB_CO2 = pd.DataFrame(x)
    WBB_CO2.columns = ['EPOCH_TIME','Corrected_DT','WBB_CO2','WBB_CH4']
    cols = WBB_CO2.columns.drop(['Corrected_DT'])
    WBB_CO2[cols]=WBB_CO2[cols].apply(pd.to_numeric,errors='coerce')
    
    return WBB_CO2
#=========================================================================================================#

def get_picarro_data(tablename,date1,date2,spikes_or_all,split_or_concat,i):
    import pandas as pd
    ###################################################################
    # Function to get exclusively "Spike Necessary" Picarro Data from #
    # SQL. Input the SQL Tablename, and date range between which data #
    # will be fetched. For one day's worth of data, enter the same    #
    # Date.                                                           #
    ###################################################################
    
    #Connect to SQL
    mycursor = sql_connect()
    if (spikes_or_all == 'spikes') & (split_or_concat == 'split'):
        raise KeyError("Cannot have spikes and split, as the spikes df only gets some wind information")
        
        
    if spikes_or_all == 'spikes':
        mycursor.execute("SELECT Local_DT, EPOCH_TIME, CO2_dry, ANEMOMETER_UY\
                        FROM {}\
                        WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                        order by EPOCH_TIME asc;".format(tablename,date1,date2)) #SQL statement
        data = mycursor.fetchall() #fetch the data
        Picarro = pd.DataFrame(list(data)) #convert data to a dataframe
        Picarro.columns = ['Local_DT','EPOCH_TIME','Pic_CO2','ANEM_Y'] #name columns
    elif spikes_or_all == 'all':
        if split_or_concat =='concat':
            mycursor.execute("SELECT Local_DT, EPOCH_TIME, CO2_dry, CH4_dry, ANEMOMETER_UY, ANEMOMETER_UX, ANEMOMETER_UZ, Location_Picarro\
                            FROM {}\
                            WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                            order by EPOCH_TIME asc;".format(tablename,date1,date2)) #SQL statement
            data = mycursor.fetchall() #fetch the data
            Picarro = pd.DataFrame(list(data)) #convert data to a dataframe
            Picarro.columns = ['Local_DT','EPOCH_TIME','Pic_CO2','Pic_CH4','ANEM_Y','ANEM_X','ANEM_Z','Pic_Loc'] #name columns
        elif split_or_concat == 'split':
            if i == 0:
                mycursor.execute("SELECT Local_DT, EPOCH_TIME, CO2_dry, CH4_dry, Location_Picarro\
                                FROM {}\
                                WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                                order by EPOCH_TIME asc;".format(tablename,date1,date2)) #SQL statement
                data = mycursor.fetchall() #fetch the data
                Picarro = pd.DataFrame(list(data)) #convert data to a dataframe
                Picarro.columns = ['Local_DT','EPOCH_TIME','Pic_CO2','Pic_CH4','Pic_Loc'] #name columns  
            else:
                mycursor.execute("SELECT Local_DT, EPOCH_TIME, ANEMOMETER_UY, ANEMOMETER_UX, ANEMOMETER_UZ, Location_Picarro\
                                FROM {}\
                                WHERE Local_DT >= '{} 00:00:00' AND Local_DT <= '{} 23:59:59.99'\
                                order by EPOCH_TIME asc;".format(tablename,date1,date2)) #SQL statement
                data = mycursor.fetchall() #fetch the data
                Picarro = pd.DataFrame(list(data)) #convert data to a dataframe
                Picarro.columns = ['Local_DT','EPOCH_TIME','ANEM_Y','ANEM_X','ANEM_Z','Pic_Loc'] #name columns
        else:
            raise KeyError('Input "split" or "concat"')
    else:
        raise ValueError('Input spikes or all as the last argument')
    cols = Picarro.columns.drop('Local_DT') #get all column names but date
    Picarro[cols]=Picarro[cols].apply(pd.to_numeric,errors='coerce') #change all but date to floats
    
    return Picarro

#=========================================================================================================#
    
def get_sql_data(LI_vent_sql_tablename,Multiplexer_sql_tablename,\
                 Vent_Anem_Temp_sql_tablename,Picarro_sql_tablename,date1,date2,spikes_or_all,split_or_concat):
    
    ######################################################################
    #Script pulls in all of the necessary data in the date range input into the function
    #Inputs: The names of each SQL table
    #        date1 - start date for range of data to pull
    #        date2 - end date for range of data to pull                    #
    ######################################################################

    import pandas as pd

    dict_of_dfs = {}
    #Import source (LI_8100_Vent) data
    #If there is a value error (no data in table for date range), set up an empty dataframe and pass the error
    print('Retrieving LI_vent data')
    try:
        dict_of_dfs['LI_Vent']=get_LI_data(LI_vent_sql_tablename,date1,date2)
    except ValueError:
        dict_of_dfs['LI_Vent'] = pd.DataFrame() #set empty dataframe
        pass
    
    
    #Import Multiplexer data
    #If there is a value error (no data in table for date range), set up an empty dataframe and pass the error
    print('Retrieving Multiplexer data')
    try:
        if split_or_concat == 'concat':
            dict_of_dfs['Multiplexer'] = get_multiplexer_data(Multiplexer_sql_tablename,date1,date2,split_or_concat,0)
        elif split_or_concat == 'split':
            for i in range(1,4):
                #if i < 4:
                dict_of_dfs['Multiplexer_CO2_{}'.format(i)] = get_multiplexer_data(Multiplexer_sql_tablename,date1,date2,split_or_concat,i)
                #else:
                 #   dict_of_dfs['Multiplexer_Weather'] = get_multiplexer_data(Multiplexer_sql_tablename,date1,date2,split_or_concat,i)
                
        else: 
            raise KeyError('Input "split" or "concat" as the last argument')
    except ValueError:
        dict_of_dfs['Multiplexer'] = pd.DataFrame() #make empty dataframe
        pass
    
    #Import Vent_Anem_Temp data
    #If there is a value error (no data in table for date range), set up an empty dataframe and pass the error
    print('Retrieving Vent_Anem_Temp data')
    try:
        dict_of_dfs['Vent_Anem_Temp'] = get_vent_anem_temp_data(Vent_Anem_Temp_sql_tablename,date1,date2)
    except ValueError:
        dict_of_dfs['Vent_Anem_Temp'] = pd.DataFrame() #make empty dataframe
        pass

    
    #Import Picarro data
    #If there is a value error (no data in table for date range), set up an empty dataframe and pass the error
    #print('Retrieving Picarro data')
    #try:
    #    if split_or_concat == 'concat':
    #        dict_of_dfs['Picarro'] = get_picarro_data(Picarro_sql_tablename,date1,date2,spikes_or_all,split_or_concat,0)
    #    if split_or_concat == 'split':
    #        for i in range(0,2):
    #            if i == 0 :
    #                dict_of_dfs['Picarro_CO2'] = get_picarro_data(Picarro_sql_tablename,date1,date2,spikes_or_all,split_or_concat,i)
    #            else:
    #                dict_of_dfs['Picarro_ANEM'] = get_picarro_data(Picarro_sql_tablename,date1,date2,spikes_or_all,split_or_concat,i)
    #except ValueError:
    #    dict_of_dfs['Picarro'] = pd.DataFrame() #make empty dataframe
    #    pass
    
    #Import WBB_weather data
    #If there is a value error (no data in table for date range), set up an empty dataframe and pass the error
    print('Retrieving WBB Weather data')
    try:
        dict_of_dfs['WBB_Weather'] = get_wbb_weather(date1,date2)
    except ValueError:
        dict_of_dfs['WBB_Weather'] = pd.DataFrame() #make empty dataframe
        pass
    
    #Import WBB_CO2 data
    #If there is a value error (no data in table for date range), set up an empty dataframe and pass the error
    print('Retrieving WBB CO2 data')
    try:
        dict_of_dfs['WBB_CO2'] = get_wbb_co2(date1,date2)
    except ValueError:
        dict_of_dfs['WBB_CO2'] = pd.DataFrame() #make empty dataframe
        pass
    
    return dict_of_dfs#return all of the fetched dataframes

#=========================================================================================================#
    
#Plot a simple graph with local_dt on the x axis, and an input (y_ax) on the y axis
def simple_plot(df,x_ax,y_ax):
    import matplotlib.pyplot as plt
    
    fig,ax = plt.subplots() #make the fig,ax
    ax.yaxis.grid(which="major") #plot horizontal gridlines
    ax.plot(df[x_ax],df[y_ax]) #plot
    plt.gcf().autofmt_xdate() #get a nice date format for the x axis
    fig.tight_layout()
    plt.show()
#=========================================================================================================#
def plot_stacked_same(args):
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    fig,ax = plt.subplots(figsize=(20,10))
    
    
    for arg in args:
        ax.plot(arg[0][arg[1]],arg[0][arg[2]])
    plt.gcf().autofmt_xdate()
    fig.tight_layout()
    plt.show()
#=========================================================================================================#
 #Make a figure of multiple plots plotted above one another. X axes have the same values
def plot_vertical_stack(args):
    ######################################################################################################################
    # Function to refine dataframes to a user input time frame.                                                          #
    # Input a list of lists, as many as desired. The format should be:                                                   #
    # [[dataframe1,'x_axis_column_label','y_axis_column_label],[dataframe2,'x_axis_column_label','y_axis_column_label]]  #
    
    ######################################################################################################################
    
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as grd
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    
    num_plots = len(args)
    fig = plt.figure(figsize = (20,num_plots*10))
    gs = grd.GridSpec(num_plots,1)
    i=0
    for arg in args:
        
        if i == 0:
            ax = fig.add_subplot(gs[i])
        else:
            ax = fig.add_subplot(gs[i],sharex=ax)
        if arg[2] == 'wd':
            ax.scatter(arg[0][arg[1]],arg[0][arg[2]],s=2)
            ax.set_title("{}".format(arg[2]),size=20)
        else: 
            ax.plot(arg[0][arg[1]],arg[0][arg[2]])
            ax.set_title("{}".format(arg[2]),size=20)

        if ('ANEM_X' in arg[0].columns) & (arg[2] == 'CO2'):
            ax.set_ylim([390,650]) 
        i+=1
    plt.gcf().autofmt_xdate()
    fig.tight_layout()
    plt.show() 
    
#============================================================================================================#
    
def plot_refinement_all(args,stack_or_separate):
    
    if stack_or_separate =='stack':
        plotter = plot_stacked_same
    elif stack_or_separate =='separate':
        plotter = plot_vertical_stack
    else:
        ValueError('enter stack or separate')
    
    for arg in args:
        if arg[1] in arg[0]:
            continue
        else:
            arg[0] = arg[0].reset_index().copy()
    
    
    
    plotter(args) #plot the data
    cont_ref = True #set "Continue refining" to true, wont be changed until user says so
    ask = input("Is this an acceptable range? ") #ask the user if this is a good range, or if they want to continue refining
    
    if ask == 'y':
        clip_df_list = []
        cont_ref = False    
        for i in range(0,len(args)):
            clip_df_list.append(args[i][0])
    while cont_ref: #continue asking user to refine until command is given
        clip_df_list = []
        DT1 = input("Input Start DateTime as YYYY-mm-DD HH:MM:SS - ") #get start of range
        DT2 = input("Input End DateTime as YYYY-mm-DD HH:MM:SS - ") #get end of range

        for i in range(0,len(args)):
            df = args[i][0]
            df1 = df.loc[(df[args[i][1]]>=DT1)&(df[args[i][1]]<=DT2)] #clip the data to the range given
            clip_df_list.append(df1)
    
        plot_list = []
        for i in range(0,len(args)):
            l = []
            l.append(clip_df_list[i])
            l.append(args[i][1])
            l.append(args[i][2])
            plot_list.append(l)
            
        plotter(plot_list) #plot over that range
        
        ask = input("Is this an acceptable range? ") #ask the user if this is a good range, or if they want to continue refining
        if ask == 'y':
            cont_ref = False
    
    return clip_df_list

#==============================================================================================================#
    
def plot_spike_with_pts(df_list,ET,y_ax_var,y_ax_str):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(15,3))
    ax.plot(df_list['EPOCH_TIME'],df_list[y_ax_str],color='blue')
    ax.scatter(df_list['EPOCH_TIME'],df_list[y_ax_str],color='red')        
    ax.scatter(ET,y_ax_var,color='black')    
    fig.subplots_adjust()
    plt.show()
    
#==============================================================================================================#

# This function gets a single "spike" from picarro anemometer data
# Input: Picarro style dataframe
# Output: List of the selected EPOCH time associated with the spike
def get_single_pic_anem_spikes(df):
    import numpy as np
    import pandas as pd
    
    df = df.reset_index(drop=True) # reset the index
    y_ax = 'ANEM_Y'  # set the y_ax to ANEM_Y - this is where spikes are generally input    
    x_ax = 'Local_DT'

    simple_plot(df,x_ax,y_ax)  # plot the data

    #This is one method for "spiking" the anem data. We unplug the serial cable so the value
    #pauses during that time. Therefore we check for where the value remains the same over a 
    #given time
    df['shift'] = df['ANEM_Y'].shift(-15)   #create a column shifted 10 seconds later 
    df['shift2'] = df['ANEM_Y'].shift(-10)   #create a column shifted 5 seconds later
    df['shift3'] = df['ANEM_Y'].shift(-5)   #create a column shifted 5 seconds later

    spike_ixs = np.where((df['ANEM_Y']==df['shift'])&(df['ANEM_Y']==df['shift2'])&(df['ANEM_Y']==df['shift3']))[0]  #check to see where ANEM_Y values are  
                                                                                       #equal to the ANEM_Y values 5 and 10
                                                                                       #seconds later. Store in a list. 

    #initialize some variables
    num_spikes = 0
    spike_start = []
    spike_end = []
    spike_start.append(spike_ixs[0])

    #Count number of spikes (sustained same value without large gap)
    #Append the start of each spike and end of each spike to created arrays
    for i in range(0,len(spike_ixs)-2):
        if (spike_ixs[i+1]-spike_ixs[i]>20):
            spike_end.append(spike_ixs[i])
            num_spikes += 1
            spike_start.append(spike_ixs[i+1])
    spike_end.append(spike_ixs[-1])  
    spikes = pd.DataFrame({'Start':spike_start, 'End':spike_end}, columns =['Start','End'])

    #Create a list of dataframes. Each dataframe is the range around the spike. It is "r" 
    #datapoints (in this case 0.1s) before the spike start and "r" datapoints after the 
    #spike end
    r = 30 #set r
    df_list = {}
    for i in range(0,len(spikes)):
        df_list[i] = df[spikes['Start'][i]-r:spikes['End'][i]+r] #get the range

    #create a list of spike starts so we don't have to keep referencing into the spikes dataframe
    st_spike_idx = []
    for i in range(0,len(df_list)):
        st_spike_idx.append(spikes['Start'][i] - 1)  #append the start of each spike

    #Here we will ask the user if the spike is in the correct spot
    for j in range(0,len(spikes)):
        refine = True
        while refine:       
            ANEM = []
            ET = []

            for i in range(0,len(spikes)):
                ET.append(df['EPOCH_TIME'][st_spike_idx[i]])
                ANEM.append(df['ANEM_Y'][st_spike_idx[i]])
            plot_spike_with_pts(df_list[j],ET[j],ANEM[j],'ANEM_Y')

            ask = input("Is the spike start in the correct spot?")
            if ask == 'y':
                refine = False
            else:
                spike_refine = int(input("Spike Index Move: "))
                st_spike_idx[j] = st_spike_idx[j] + spike_refine
    return ET #return the list of spike start EPOCH times

#==============================================================================================================#
   
def get_single_pic_co2_spikes(df):
    import numpy as np
    import pandas as pd
    
    df = df.reset_index(drop=True)
    y_ax = 'Pic_CO2'

    simple_plot(df,'Local_DT',y_ax)


    threshold = float(input("Enter the threshold value which all spikes go above: "))
    spike_ixs = np.where(df[y_ax]>=threshold)[0]

    num_spikes = 0
    spike_start = []
    spike_end = []
    spike_start.append(spike_ixs[0])

        #Count number of spikes (sustained -99.99 values without large gap)
        #Append the start of each spike and end of each spike to created arrays
    for i in range(0,len(spike_ixs)-2):
        if (spike_ixs[i+1]-spike_ixs[i]>20):
            spike_end.append(spike_ixs[i])
            num_spikes += 1
            spike_start.append(spike_ixs[i+1])
    spike_end.append(spike_ixs[-1])  
    spikes = pd.DataFrame({'Start':spike_start, 'End':spike_end}, columns =['Start','End'])


    df_list = {}
    for i in range(0,len(spikes)):
        df_list[i] = df[spikes['Start'][i]-30:spikes['End'][i]+30]

    st_spike_idx = []
    for i in range(0,len(df_list)):
        co2_diff = df_list[i][y_ax][spikes['Start'][i]] - df_list[i][y_ax][spikes['Start'][i]-1]
        st_spike_idx.append(spikes['Start'][i] - 1)
        while co2_diff > 1000:
            co2_diff = df_list[i][y_ax][st_spike_idx[i]] - df_list[i][y_ax][st_spike_idx[i]-1]
            st_spike_idx[i] = st_spike_idx[i]-1
        st_spike_idx[i] += 1

    for j in range(0,len(spikes)):
        refine = True
        while refine:       
            CO2 = []
            ET = []

            for i in range(0,len(spikes)):
                ET.append(df['EPOCH_TIME'][st_spike_idx[i]])
                CO2.append(df[y_ax][st_spike_idx[i]])
            plot_spike_with_pts(df_list[j],ET[j],CO2[j],y_ax)

            ask = input("Is the spike start in the correct spot?")
            if ask == 'y':
                refine = False
            else:
                spike_refine = int(input("Spike Index Move: "))
                st_spike_idx[j] = st_spike_idx[j] + spike_refine

    return ET
#==============================================================================================================#
# This function gets a single "spike" from picarro CO2 data
# Input: Picarro style dataframe
# Output: List of the selected EPOCH time associated with the spike 
def get_single_LI_spike(df):
    import numpy as np
    import pandas as pd
    
    df = df.reset_index(drop=True) #reset index
    y_ax = 'LI_CO2' #looking at CO2 data

    simple_plot(df,'Local_DT',y_ax) #plot the data for reference
    
    threshold = float(input("Enter the threshold value which all spikes go above: ")) #have the user input a threshold above which the spike goes
    
    spike_ixs = np.where((df[y_ax]>=threshold) | (df[y_ax]<0))[0] #find the indidcies where CO2 value is above input threshold

    num_spikes = 0
    spike_start = []
    spike_end = []
    spike_start.append(spike_ixs[0])

    #Count number of spikes (sustained values above threshold without large gap)
    #Append the start of each spike and end of each spike to created arrays
    for i in range(0,len(spike_ixs)-2):
        if (spike_ixs[i+1]-spike_ixs[i]>3):
            spike_end.append(spike_ixs[i])
            num_spikes += 1
            spike_start.append(spike_ixs[i+1])
    spike_end.append(spike_ixs[-1])  
    
    #create a spikes dataframe with start, center, and end indicies
    spikes = pd.DataFrame({'Start':spike_start, 'End':spike_end}, columns =['Start','End'])
    spikes['Center'] = spikes.apply(lambda row: int((row.Start + row.End)/2),axis=1)

    #Create a list of dataframes. Each dataframe is the range around the spike. It is "r" 
    #datapoints (in this case 0.1s) before the spike start and "r" datapoints after the 
    #spike end
    r = 10
    df_list = {}
    for i in range(0,len(spikes)):
        df_list[i] = df[spikes['Start'][i]-r:spikes['End'][i]+r]   
    
    #create a list of spike starts so we don't have to keep referencing into the spikes dataframe
    st_spike_idx = []
    for i in range(0,len(df_list)):
        co2_diff = abs(df_list[i][y_ax][spikes['Start'][i]] - df_list[i][y_ax][spikes['Start'][i]-1]) #get the difference between point i and i+1
        st_spike_idx.append(spikes['Start'][i] - 1) #append the index value for the start of the spike to the list
        while co2_diff > 1000: #we want to find where the change is large - so where the change between points is greater than 1000, keep moving backward
            co2_diff = abs(df_list[i][y_ax][st_spike_idx[i]] - df_list[i][y_ax][st_spike_idx[i]-1]) #find the difference between the current value and the point before. 
            st_spike_idx[i] = st_spike_idx[i]-1 #increment the index backward
        st_spike_idx[i] += 1 #once the difference is less than 1000, store the next value as the start of the spike

    #Here we will ask the user if the spike is in the correct spot
    for j in range(0,len(spikes)):
        refine = True
        while refine:       
            CO2 = []
            ET = []

            for i in range(0,len(spikes)):
                ET.append(df['EPOCH_TIME'][st_spike_idx[i]])
                CO2.append(df[y_ax][st_spike_idx[i]])
            plot_spike_with_pts(df_list[j],ET[j],CO2[j],y_ax)

            ask = input("Is the spike start in the correct spot?")
            if ask == 'y':
                refine = False
            else:
                spike_refine = int(input("Spike Index Move: "))
                st_spike_idx[j] = st_spike_idx[j] + spike_refine
    
    return ET

#==============================================================================================================#
    
def get_single_multiplexer_spike(df,y_ax):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as grd
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    
    df = df.reset_index(drop=True)

    simple_plot(df,'Local_DT',y_ax)

    threshold = float(input("Enter the threshold value which all spikes go above: "))
    spike_ixs = np.where((df[y_ax]>=threshold))[0]

    num_spikes = 0
    spike_start = []
    spike_end = []
    spike_start.append(spike_ixs[0])

    for i in range(0,len(spike_ixs)-2):
        if (spike_ixs[i+1]-spike_ixs[i]>2):
            spike_end.append(spike_ixs[i])
            num_spikes += 1
            spike_start.append(spike_ixs[i+1])
    spike_end.append(spike_ixs[-1])  
    spikes = pd.DataFrame({'Start':spike_start, 'End':spike_end}, columns =['Start','End'])
    spikes['Center'] = spikes.apply(lambda row: int((row.Start + row.End)/2),axis=1)

    print(spikes)
    
    df_list = {}
    for i in range(0,len(spikes)):
        df_list[i] = df[spikes['Start'][i]-5:spikes['End'][i]+5]

        
    st_spike_idx = []
    for i in range(0,len(df_list)):
        co2_diff = abs(df_list[i][y_ax][spikes['Start'][i]] - df_list[i][y_ax][spikes['Start'][i]-1])
        st_spike_idx.append(spikes['Start'][i] - 1)
        while co2_diff > 200:
            co2_diff = abs(df_list[i][y_ax][st_spike_idx[i]] - df_list[i][y_ax][st_spike_idx[i]-1])
            st_spike_idx[i] = st_spike_idx[i]-1
        st_spike_idx[i] += 1

    for j in range(0,len(spikes)):
        refine = True
        while refine:       
            CO2 = []
            ET = []
            for i in range(0,len(spikes)):
                ET.append(df['EPOCH_TIME'][st_spike_idx[i]])
                CO2.append(df[y_ax][st_spike_idx[i]])

            fig = plt.figure(figsize=(15,len(spikes)*3))
            gs = grd.GridSpec(len(spikes),1)
            #for i in range(0,len(df_list)):
            ax = fig.add_subplot(gs[j])
            ax.plot(df_list[j]['EPOCH_TIME'],df_list[j][y_ax],color='blue')
            ax.scatter(df_list[j]['EPOCH_TIME'],df_list[j][y_ax],color='red')
            ax.scatter(ET[j],CO2[j],color='black')    
            fig.subplots_adjust()
            plt.show()

            ask = input("Is the spike start in the correct spot?")
            if ask == 'y':
                refine = False
            else:
                spike_refine = int(input("Spike Index Move: "))
                st_spike_idx[j] = st_spike_idx[j] + spike_refine          

    return ET


#==============================================================================================================#
    
def get_all_multiplexer_spikes(df):
    multi_spike_ET = {}
    for i in range(1,4):
        multi_spike_ET[i] = get_single_multiplexer_spike(df,"CO2_{}".format(i))

    return multi_spike_ET

#==============================================================================================================#
    
def get_single_vent_anem_temp_spike(df,y_ax):
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    from datetime import datetime
    
    
    df = df.reset_index(drop=True)
    
    #Plot for visualization
    fig,ax = plt.subplots() #make the fig,ax
    ax.yaxis.grid(which="major") #plot horizontal gridlines
    ax.plot(df['Local_DT'],df[y_ax]) #plot
    ax.scatter(df['Local_DT'],df[y_ax],color='Black')
    plt.gcf().autofmt_xdate() #get a nice date format for the x axis
    fig.tight_layout()
    plt.show()
    
    #confirm spike or dip
    redo = True
    while redo:
        up_or_down = input("Is this a spike or a dip? ")
        if up_or_down == 'spike':
            threshold = float(input("Enter the threshold value which all spikes go above: "))
            spike_ixs = np.where(df[y_ax]>=threshold)[0] #If it's a spike, get the indecies of all values above the input threshold
            first_rot = df[y_ax].loc[spike_ixs[0]] #Store the number of rotations recorded for the first "spike" index
            rot_tot = df.loc[spike_ixs[0]:spike_ixs[-1]].sum()[y_ax] #Get the sum of all rotations recorded during the spike
            rot_p_sec = rot_tot/10 #The spike is a 10 second run, divide by 10 to get a "rotations per second" value
            sec_before = int(round(first_rot/rot_p_sec)) #Find the amount of time before the initial spike epoch time when the spike started (number of rotations in first 
                                                        # spike recording divided by spikes per second)
            redo = False
        elif up_or_down == 'dip':
            threshold = float(input("Enter the threshold value which all spikes go below: "))
            spike_ixs = np.where(df[y_ax]<=threshold)[0] #If it's a dip, get the indecies of all values below the input threshold
            if len(spike_ixs) == 1:
                sec_before = df['EPOCH_TIME'].loc[spike_ixs[0]]-df['EPOCH_TIME'].loc[spike_ixs[0]-1]
            else: 
                before_ix = spike_ixs[0]-1
                after_ix =spike_ixs[-1]+1
                before_rot = df[y_ax].loc[before_ix] #Get the numer of rotations before the spike began to use in average
                after_rot = df[y_ax].loc[after_ix] #Get the number of rotations after the spike ended to use in average
                #average_rot = (before_rot+after_rot)/2 #get the average number of rotations surrounding the dip

                first_rot_diff = before_rot - df[y_ax].loc[spike_ixs[0]] #Store the difference in number of rotations recorded for the first "spike" index
                last_rot_diff = after_rot - df[y_ax].loc[spike_ixs[-1]]

                rot_tot = first_rot_diff+last_rot_diff #Get the sum of all rotations recorded during the spike subtracted from twice the average
                rot_p_sec = rot_tot/10 #The spike is a 10 second run, divide by 10 to get a "rotations per second" value
                sec_before = int(round(first_rot_diff/rot_p_sec)) #Find the amount of time before the initial spike epoch time when the spike started (number of rotations in first 
                                                            # spike recording divided by spikes per second)
            redo = False
        else:
            print("Didn't enter spike or dip correctly.")
            redo = True

    
    
    ET = []
    ET.append(df['EPOCH_TIME'].loc[spike_ixs[0]]-sec_before) #Append the spike time. This is the spike start epoch time minus the seconds before as calculated above. 
                                                            # the reason this is subtracted is that the time on the arduino is recorded at the END of the averaging scheme
                                                            # meaning that the count recorded is the count between the previous time and the time recorded for that count
    x_line = datetime.fromtimestamp(ET[0])    
        
    fig,ax = plt.subplots() #make the fig,ax
    ax.yaxis.grid(which="major") #plot horizontal gridlines
    ax.plot(df['Local_DT'],df[y_ax]) #plot
    ax.scatter(df['Local_DT'],df[y_ax],color='Black')
    plt.axvline(x=x_line,color='red')
    plt.xlabel('Time')
    plt.ylabel('Anemometer Rotations')
    plt.gcf().autofmt_xdate() #get a nice date format for the x axis
    fig.tight_layout()
    plt.show()
    
    
    return ET

#==============================================================================================================#
def spike_ET_df_creation(**kwargs):
    import pandas as pd
    
    spike_ET_df = pd.DataFrame()
    for key, value in kwargs.items():
        if 'ANEM' in key :
            spike_ET_df['Picarro_ANEM'] = get_single_pic_anem_spikes(value)
            print("1",spike_ET_df)
        elif 'CO2' in key:
            spike_ET_df['Picarro_CO2']= get_single_pic_co2_spikes(value)   
        elif 'vent' in key:
            spike_ET_df['LI_8100_Vent']= get_single_LI_spike(value)
        elif 'remote' in key:
            spike_ET_df['LI_8100_Remote']= get_single_LI_spike(value)
        elif 'Multiplexer' in key:
            #spike_ET_df['Multiplexer_CO2_1'] = get_single_multiplexer_spike(value,'CO2_1')
            multi_spikes = get_all_multiplexer_spikes(value)
            for i in range(1,4):
                spike_ET_df['Multiplexer_CO2_{}'.format(i)] = multi_spikes[i]
        elif 'Temp' in key:
            spike_ET_df['Vent_Anem_Temp'] = get_single_vent_anem_temp_spike(value,'Rotations')
        
    return spike_ET_df

#==============================================================================================================#
    
def fill_multiplexer_gaps(df):
    import pandas as pd
    
    df.EPOCH_TIME = df.EPOCH_TIME.astype(int)
    st = df['EPOCH_TIME'].min()
    end =  df['EPOCH_TIME'].max()
    x = df.set_index('EPOCH_TIME').reindex(range(st,end,1)).interpolate().rename_axis('EPOCH_TIME').reset_index()

    x['Local_DT'] = pd.to_datetime(x['EPOCH_TIME'],unit='s') - pd.Timedelta('06:00:00')

    return x

#==============================================================================================================#
    
def append_real_DT(spike_ET_df,actual_spike_df):
    import pandas as pd
    from datetime import datetime
    from time import mktime
    
    actual_DT = []
    for i in range(0,len(spike_ET_df)):
        actual_DT.append(input('Actual datetime for this ({}) spike? '.format(i)))

    actual_DT_df = pd.DataFrame({'Actual_DT':actual_DT})
    df_to_append = pd.concat([actual_DT_df,spike_ET_df],axis=1)

    actual_spike_df = actual_spike_df.append(df_to_append,sort=True).reset_index(drop=True)
    
    actual_spike_df['Actual_ET'] = actual_spike_df.apply(lambda row: mktime(datetime.strptime(row['Actual_DT'],"%Y-%m-%d %H:%M:%S").timetuple()),axis=1)    
    
    return actual_spike_df

#==============================================================================================================#
    
def create_actual_spike(actual_spike_df,**kwargs):    
    for key, value in kwargs.items():
        if 'CO2' in key :
            Picarro_clip = plot_refinement_all([[value,'Local_DT','Pic_CO2']],'separate')[0]
            spike_ET_df = spike_ET_df_creation(Picarro_CO2_df = Picarro_clip)       
        elif 'ANEM' in key:
            Picarro_clip = plot_refinement_all([[value,'Local_DT','ANEM_Y']],'separate')[0]
            spike_ET_df = spike_ET_df_creation(Picarro_ANEM_df = Picarro_clip)    
            print("2",spike_ET_df)
        elif 'vent' in key:
            LI_vent_clip = plot_refinement_all([[value,'Local_DT','LI_CO2']],'separate')[0]
            spike_ET_df = spike_ET_df_creation(LI_vent_df = LI_vent_clip)
        elif 'Multiplexer' in key:
            Multiplexer_clip = plot_refinement_all([[value,'Local_DT','CO2_1'],[value,'Local_DT','CO2_3'],[value,'Local_DT','CO2_3']],'separate')[0]
            Multiplexer_clip = fill_multiplexer_gaps(Multiplexer_clip)
            spike_ET_df = spike_ET_df_creation(Multiplexer_df = Multiplexer_clip)
        elif 'Temp' in key:
            Vent_Anem_Temp_clip = plot_refinement_all([[value,'Local_DT','Velocity']],'separate')[0]
            spike_ET_df = spike_ET_df_creation(Vent_Anem_Temp_df = Vent_Anem_Temp_clip)
        arg1 = append_real_DT(spike_ET_df,actual_spike_df)
    return arg1

#==============================================================================================================#

def create_lag_df(actual_spike_df):
    import pandas as pd
    
    cols = actual_spike_df.columns.drop(['Actual_DT','Actual_ET'])
    lags = pd.DataFrame()
    lags['Actual_DT'] = actual_spike_df['Actual_DT']
    lags['Actual_ET'] = actual_spike_df['Actual_ET']
    for col in cols:
        lags[col] = actual_spike_df['Actual_ET'] - actual_spike_df[col]


    return lags

#==============================================================================================================#

def get_lag_groups(actual_spike_df,column):
    import pandas as pd
    import numpy as np
    
    spike_df = actual_spike_df[['Actual_DT','Actual_ET',column]].dropna()
    spike_df['lags'] = spike_df.apply(lambda row: row['Actual_ET']-row[column],axis=1)
    spike_df['diff'] = spike_df['Actual_ET'] - spike_df['Actual_ET'].shift(1)
    spike_df.reset_index(drop=True,inplace=True)
    
    grp = int(0)
    df_list = {}
    st_ix = 0 
    end_ix = 0
    for i in range(1,len(spike_df)):
        if spike_df.loc[i,'diff'] < 1000:
            end_ix += 1
        else:
            df_list[grp] = pd.DataFrame(spike_df.loc[st_ix:end_ix])
            grp+=1
            end_ix += 1
            st_ix = end_ix
    df_list[grp] = pd.DataFrame(spike_df.loc[st_ix:end_ix])
    

    ETs = []
    ave_lags = []
    for i in range(0,len(df_list)):
        ETs.append((df_list[i][column].iloc[0]+df_list[i][column].iloc[-1])/2)
        ave_lags.append(np.mean(df_list[i]['lags']))

    final_lags = pd.DataFrame({'mid_ET':ETs,'ave_lag':ave_lags})

    for i in range(0,len(final_lags)-1):
        final_lags.loc[i,'slope']= (final_lags.loc[i,'ave_lag']-final_lags.loc[i+1,'ave_lag'])/(final_lags.loc[i,'mid_ET']-final_lags.loc[i+1,'mid_ET'])
        
    return final_lags

#==============================================================================================================#
    
def df_correction_lag_slope(final_lags,df):
    ######################################################################
    # Correct the time drift of a dataframe using the grouped lag df     #
    ######################################################################
    
    import pandas as pd
    from datetime import datetime

    def row_correction(row,final_lags_df,grp):
        return row+final_lags_df['ave_lag'][grp]+(row-final_lags_df['mid_ET'][grp])*final_lags_df['slope'][grp]

    df_to_correct = df.copy()
    df_corr_list = {}
    df_corr_list[0] = df_to_correct.where((df_to_correct['EPOCH_TIME'] < final_lags.loc[1,'mid_ET'])).dropna()
    df_corr_list[0]['Corrected_ET'] = df_corr_list[0]['EPOCH_TIME'].apply(row_correction,args=(final_lags,0))
    df_corr_list[0]['Corrected_DT'] = df_corr_list[0]['Corrected_ET'].apply(lambda x: datetime.fromtimestamp(x))

    for i in range(1,len(final_lags)-1):
        df_corr_list[i] = df_to_correct.where((df_to_correct['EPOCH_TIME'] >= final_lags.loc[i,'mid_ET']) & (df_to_correct['EPOCH_TIME'] <= final_lags.loc[i+1,'mid_ET'])).dropna(how='all')
        df_corr_list[i]['Corrected_ET'] = df_corr_list[i]['EPOCH_TIME'].apply(row_correction,args=(final_lags,i))
        df_corr_list[i]['Corrected_DT'] = df_corr_list[i]['Corrected_ET'].apply(lambda x: datetime.fromtimestamp(x))

    corrected_df = pd.concat(df_corr_list)
    corrected_df.drop_duplicates(['EPOCH_TIME'],inplace=True)

    return corrected_df

#==============================================================================================================#
def drift_correct(dict_of_dfs):
    import pandas as pd
    from datetime import datetime
    
    print("Initializing Drift Correct")
    
    data = dict_of_dfs.copy()
    spikes = pd.read_pickle('Spike_ETs.pkl')
    for key in data:
        if data[key].empty:
            print("{} is empty - no correction needed".format(key))
            continue
        print('Correcting data for {}'.format(key))

        if (key == 'WBB_CO2')|(key=='WBB_Weather'):
            continue
        lags = get_lag_groups(spikes,key)
        data[key] = df_correction_lag_slope(lags,data[key])
    
    if 'Multiplexer_CO2_3' in data.keys():
        data['Multiplexer_Weather'] = data['Multiplexer_CO2_3'][['EPOCH_TIME','Local_DT','Rotations','Wind_Velocity','Wind_Direction','Corrected_ET']]
        data['Multiplexer_Weather']['Corrected_ET'] = data['Multiplexer_Weather'].apply(lambda row: row['Corrected_ET']-2,axis=1)
        data['Multiplexer_Weather']['Corrected_DT'] = data['Multiplexer_Weather']['Corrected_ET'].apply(lambda row: datetime.fromtimestamp(row))
        
        data['Multiplexer_CO2_3'].drop(['Rotations','Wind_Velocity','Wind_Direction'],axis=1,inplace=True)
    
    return data
#==============================================================================================================#
def delete_WBB_cal(df):
    import pandas as pd
    df_to_corr = df.copy()
    z = df_to_corr.where(df_to_corr['WBB_CO2']<10).dropna()

    z['diff'] = z['EPOCH_TIME']-z['EPOCH_TIME'].shift(1)
    z.reset_index(drop=True,inplace=True)

    grp = int(0)
    df_list = {}
    st_ix = 0
    end_ix = 0

    for i in range(1,len(z)):
        if z.loc[i,'diff'] < 1000:
            end_ix += 1
        else:
            df_list[grp] = pd.DataFrame(z.loc[st_ix:end_ix])
            grp+=1
            end_ix += 1
            st_ix = end_ix
    df_list[grp] = pd.DataFrame(z.loc[st_ix:end_ix])
    
    starts = []
    ends = [] 
    for i in range(0,len(df_list)):
        starts.append(df_list[i]['EPOCH_TIME'].iloc[0])
        ends.append(df_list[i]['EPOCH_TIME'].iloc[-1])

    df = pd.DataFrame({'Starts':starts,'Ends':ends})
    beg = df_to_corr['EPOCH_TIME'].iloc[0]
    end = df_to_corr['EPOCH_TIME'].iloc[-1]

    df = df.loc[(df['Starts']>beg)&(df['Ends']<end)]

    new_df = df_to_corr
    for i in range(0,len(df)):
        new_df = new_df.loc[(new_df['EPOCH_TIME']<(df['Starts'].iloc[i]-120))|(new_df['EPOCH_TIME']>(df['Ends'].iloc[i]+120))]

    return new_df

#==============================================================================================================#
def get_grouped_spike_list(spike_df,key):
    
    
    import pandas as pd
    
    spike_df = spike_df[['Actual_DT','Actual_ET',key]].dropna()
    spike_df['lags'] = spike_df.apply(lambda row: row['Actual_ET']-row[key],axis=1)
    spike_df['diff'] = spike_df['Actual_ET'] - spike_df['Actual_ET'].shift(1)
    spike_df.reset_index(drop=True,inplace=True)
    
    grp = int(0)
    df_list = {}
    st_ix = 0 
    end_ix = 0
    for i in range(1,len(spike_df)):
        if spike_df.loc[i,'diff'] < 1000:
            end_ix += 1
        else:
            df_list[grp] = pd.DataFrame(spike_df.loc[st_ix:end_ix])
            grp+=1
            end_ix += 1
            st_ix = end_ix
    df_list[grp] = pd.DataFrame(spike_df.loc[st_ix:end_ix])
    
        
    return df_list

#==============================================================================================================#
    
def wind_dir(x,y,pos):
    import numpy as np
    
    #determine when we need to add 360 such that the angle falls between 0 and 360 (arctan gives vals -180 to 180)
    def add_360(angle):
        if angle < 0:
            result = angle+360
        else:
            result = angle
        return result
    
    #Anemometer was facing west for positions 1,2,3,4,6. Anemometer was facing west for postion 5
    if pos == 5:
        result = add_360(90 - (np.arctan2(y,x)/np.pi*180)) #convert x,y data to a direction in degrees from 0 to 360
    else:
        result = add_360(-90 - (np.arctan2(y,x)/np.pi*180)) #convert x,y data to a direction in degrees from 0 to 360
    return result 

#==============================================================================================================#

def wind_add(df,x_lab,y_lab):
    ######################################################################
    # Use R to plot a bivariate polar plot. Uses R package "openair"     #
    # Converts a pandas dataframe to r dataframe and plots.              #
    # Inputs:   df = pandas dataframe with x and y wind data to be       #
    #                converted to direction and speed. Must also have a  #
    #                column labeled 'Pic_Loc' representing the location  #
    #                of the picarro.                                     #
    #           x_lab,y_lab = columns to be used for direction and speed #
    #                         calculation                                #
    ######################################################################
    import numpy as np
    print("Adding Wind Direction as 'wd'")
    wd_vec = np.vectorize(wind_dir) #vectorize the wind direction function
    if 'Pic_Loc' in df.columns:
        df['wd'] = wd_vec(df[x_lab],df[y_lab],df['Pic_Loc']) #add the 2d wind direction
    else:
        df['wd'] = wd_vec(df[x_lab],df[y_lab],1)
    print("Adding Wind Speed as 'ws'")
    df['ws'] = np.sqrt(df[x_lab]**2+df[x_lab]**2) #add the 2d wind speed
    return df
#==============================================================================================================#
def multi_direction_correction(df):
    import numpy as np
    data = df.copy()
    def add_360(angle):
        if angle < 0:
            result = angle+360
        else:
            result = angle
        return result
    
    add = np.vectorize(add_360)
    data['Wind_Direction'] = data['Wind_Direction'] - 90
    data['Wind_Direction'] = add(data['Wind_Direction'])
    return data
#==============================================================================================================#

def polar_plot(df,pollutant):
    ######################################################################
    # Use R to plot a bivariate polar plot. Uses R package "openair"     #
    # Converts a pandas dataframe to r dataframe and plots.              #
    # Inputs:   df = pandas dataframe with pollutant,                    #
    #               wind direction (labeled 'wd') and wind speed         #
    #               (labeled 'ws').                                      #
    #           pollutant = concentration column, likely 'Pic_CO2 or CH4 #
    ######################################################################
    
    #Import packages
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r, pandas2ri
    from rpy2.robjects.lib import grdevices
    base = importr('base')
    utils = importr('utils')
    import IPython
    from IPython.display import Image, display
    graphics = importr('graphics')
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    if not ro.packages.isinstalled('openair'):
        utils.install_packages('openair')
        
    r=ro.r   #set r object to "r"
    r.library('openair') #load openair
    
    pandas2ri.activate() #activate dataframe converter
    r_dataframe = pandas2ri.py2ri(df) #convert pandas to r

    with rpy2.robjects.lib.grdevices.render_to_bytesio(grdevices.png, width=900, height=700, res=150) as img: #set image settings
        r.polarPlot(r_dataframe,pollutant) #prepare plot

    IPython.display.display(IPython.display.Image(data=img.getvalue(), format='png', embed=True)) #display plot
    
#==============================================================================================================#
    
def wind_rose(df):
    ######################################################################
    # Use R to plot a wind rose diagram. Uses R package "openair"        #
    # Converts a pandas dataframe to r dataframe and plots.              #
    # Inputs:   df = pandas dataframe with pollutant,                    #
    #               wind direction (labeled 'wd') and wind speed         #
    #               (labeled 'ws').                                      #
    ######################################################################
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r, pandas2ri
    from rpy2.robjects.lib import grdevices
    base = importr('base')
    utils = importr('utils')
    import IPython
    from IPython.display import Image, display
    graphics = importr('graphics')
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    if not ro.packages.isinstalled('openair'):
        utils.install_packages('openair')
    r=ro.r
    r.library('openair')
    
    pandas2ri.activate()
    r_dataframe = pandas2ri.py2ri(df)

    with rpy2.robjects.lib.grdevices.render_to_bytesio(grdevices.png, width=900, height=700, res=150) as img:
        r.windRose(r_dataframe)

    IPython.display.display(IPython.display.Image(data=img.getvalue(), format='png', embed=True))

#==============================================================================================================#
    
def pollution_rose(df,pollutant):
    ######################################################################
    # Use R to plot a pollution rose diagram. Uses R package "openair"   #
    # Converts a pandas dataframe to r dataframe and plots.              #
    # Inputs:   df = pandas dataframe with pollutant,                    #
    #               wind direction (labeled 'wd') and wind speed         #
    #               (labeled 'ws').                                      #
    #           pollutant = concentration column, likely 'Pic_CO2 or CH4 #
    ######################################################################
    
    #Import everything to view graphics
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r, pandas2ri
    from rpy2.robjects.lib import grdevices
    base = importr('base')
    utils = importr('utils')
    import IPython
    from IPython.display import Image, display
    graphics = importr('graphics')
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    if not ro.packages.isinstalled('openair'):
        utils.install_packages('openair')
        
        
    r=ro.r #set r object to "r"
    r.library('openair') #load library openair
    
    pandas2ri.activate() #activate the pandas to r dataframe function
    r_dataframe = pandas2ri.py2ri(df) #convert pandas df to r df

    with rpy2.robjects.lib.grdevices.render_to_bytesio(grdevices.png, width=900, height=700, res=150) as img: #graphical settings
        r.pollutionRose(r_dataframe,pollutant)  # setup plot through r

    IPython.display.display(IPython.display.Image(data=img.getvalue(), format='png', embed=True)) #display plot
    
#==============================================================================================================#

#==============================================================================================================#
def full_download_process():
    ######################################################################
    # Function to download and store all necessary data for a user input #
    # date range.                                                        #
    # Returns a dataframe with all variables concatenated, cleaned, and  #
    # drift corrected.                                                   #
    ######################################################################
    import pandas as pd
    date1,date2 = get_date_range() #ask user for date range
    data = get_sql_data("Aug2019_LI_8100_Vent",\
                  "Aug2019_Multiplexer","Aug2019_Vent_Anem_Temp",\
                  "Aug2019_Picarro",date1,date2,'all','split') #fetch data from four instruments between dates
    
    data = drift_correct(data) #correct drifts
    for key in data:
        data[key].reset_index(drop=True,inplace=True) #reset indecies (get added for some reason)
        
        
    #data = remove_spikes(pd.read_pickle('Spike_ETs.pkl'),data) #remove the spikes so they don't skew the data
    #for key in data:
    #    data[key]['DOW'] = data[key]['Corrected_DT'].dt.dayofweek

    return data

#==============================================================================================================#


def daterange(start_date, end_date):
    from datetime import timedelta, datetime
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    for n in range(int ((end_date - start_date).days)+1):
        yield (start_date + timedelta(n)).strftime("%Y-%m-%d")


#==============================================================================================================#
def remove_spikes(spike_df,data_dict):
    import pandas as pd
    data = data_dict.copy()

    for key in data:
        if data[key].empty:
            continue
        if key == 'WBB_Weather':
            continue
        elif key == 'WBB_CO2':
            data[key] = delete_WBB_cal(data[key])
            continue
        elif key == 'Multiplexer_Weather':
            key = 'Multiplexer_CO2_1'
        df = get_grouped_spike_list(spike_df,key)
        starts = []
        ends = [] 
        for i in range(0,len(df)):
            starts.append(df[i]['Actual_ET'].iloc[0])
            ends.append(df[i]['Actual_ET'].iloc[-1])

        df = pd.DataFrame({'Starts':starts,'Ends':ends})
        beg = data[key]['Corrected_ET'].iloc[0]
        end = data[key]['Corrected_ET'].iloc[-1]

        df = df.loc[(df['Starts']>beg)&(df['Ends']<end)]

        new_df = data[key]
        for i in range(0,len(df)):
            new_df = new_df.loc[(new_df['Corrected_ET']<(df['Starts'].iloc[i]-180))|(new_df['Corrected_ET']>(df['Ends'].iloc[i]+180))]

        data[key]=new_df

    return data
    
    #==============================================================================================================#

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