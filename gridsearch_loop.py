from CO2_Dataset_Preparation import *
from ML_Model_Master import *
import os
import logging 
from datetime import datetime

#Data parameters
position_number = 1
excess_rolls = [60,6000,36000] #these specify the rolling window on which a minimum is applied for excess 
#feature_columns = ['Pic_CO2','Pic_CH4','ANEM_X','ANEM_Y','ANEM_Z','wd','ws']
downsample_sec = 10
periods_to_lag = 6
tower = 'Picarro'
train_percent = 0.8

#setup logfile
logfile_name = f'Gridsearch_Logs/{tower}_CH4in_PN{position_number}_DS{downsample_sec}_Lag{periods_to_lag}_Train{train_percent}.log'
logging.basicConfig(filename=logfile_name,level=logging.DEBUG)

try:
    #Load Dataset from Folder
    data = Processed_Set(tower,position_number,excess_rolls,True)
    data._retrieve_data('../CO2_Data_Processed/')
    data._apply_excess()

    #Build ML dataset
    ml_data = ML_Data(downsample_sec,periods_to_lag,tower,train_percent)
    ml_data._ML_Process(data)

    #Dump to pickle file
    ml_datafile_name = f'{tower}_PN{position_number}_DS{downsample_sec}_Lag{periods_to_lag}_Train{train_percent}.pkl'
    with open(ml_datafile_name, 'wb') as file:
        pickle.dump(ml_data, file)

    #Setup output file
    output_file_name = f'Gridsearch_Output/{tower}_CH4in_PN{position_number}_DS{downsample_sec}_Lag{periods_to_lag}_Train{train_percent}.out'

    #Model Parameter Loops
    activation = 'relu'
    neurons = [256]#,256]
    dropout_rate = [0.2,0.3]
    learn_rate = [0.001]#,1e-5]
    decay = [1e-5,1e-6]
    batch_size = [10,20,50]#,100]
    epochs = [100]#,50,100]
    error_metric = 'rmse'

    #Get total number of models to be trained
    tot_train = len(neurons)*len(dropout_rate)*len(learn_rate)*len(decay)*len(batch_size)*len(epochs)-1
    i=0

    #Header for output file
    f = open(output_file_name,'a')
    f.write(f"-----OUTPUT OF GRIDSEARCH: TRAINING {tot_train} MODELS------\n")
    f.write("-----DATA PARAMETERS-----\n")
    f.write("position_number,tower,downsample_sec,periods_to_lag,train_percent,error_metric,excess_rolls\n")
    f.write(f"{position_number},{tower},{downsample_sec},{periods_to_lag},{train_percent},{error_metric},{excess_rolls}\n")
    f.write("-----TRAINING PARAMETER LABELS------\n")
    f.write("activation,neurons,dropout_rate,learn_rate,decay,batch_size,epochs\n")
    f.write("-----TRAINING OUTPUT FORMAT-----\n")
    f.write(f"loss,{error_metric},val_loss,val_{error_metric}\n")
    f.write("-----BEGIN------\n")
    f.close()

    #Loop through all permutations
    for neur in neurons:
        for dr in dropout_rate:
            for lr in learn_rate:
                for dec in decay:
                    for bs in batch_size:
                        for ep in epochs:
                            #Log, Print, and Run build_train.py
                            logging.info(f"------MODEL {i} of {tot_train}-----")
                            print(f"------MODEL {i} of {tot_train}-----",flush = True)
                            os.popen(f"python build_train.py {logfile_name} {output_file_name} {ml_datafile_name} {position_number} \
                            {downsample_sec} {periods_to_lag} {tower} {train_percent} {activation} {neur} {dr} \
                            {lr} {dec} {bs} {ep} {error_metric}").read()
                            i+=1
    #Delete ML data file 
    os.remove(ml_datafile_name)
except Exception as e:
    logging.exception('Exception Raised')
    raise
else:
    logging.info('gridsearch_loop executed successfully')