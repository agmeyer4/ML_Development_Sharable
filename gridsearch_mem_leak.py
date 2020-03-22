from CO2_Dataset_Preparation import *
from ML_Model_Master import *
from datetime import datetime
import pickle

position_number = 4
feature_columns = ['Pic_CO2','ANEM_X','ANEM_Y','ANEM_Z','wd','ws']
downsample_sec = 60
periods_to_lag = [3]
tower = 'Picarro'
train_percent = 0.7

activation = 'relu'
neurons = [128]
dropout_rate = [0.2,0.3]
learn_rate = [0.001]#,1e-4,1e-5]
decay = [1e-5]#,1e-6]
batch_size = [10,20]#,50,100]
epochs = [10]#,50,100]

logfile = None

with open('ML_dataset_60DS.pkl', 'rb') as file:
    ml_data = pickle.load(file)

try:
    print_log_flush("-------------------------BUILD AND TRAIN MODELS-------------------------",logfile)

    tot_train = len(periods_to_lag)*len(neurons)*len(dropout_rate)*len(learn_rate)*len(decay)*len(batch_size)*len(epochs)-1

    models = []
    i = 0

    for neur in neurons:
        for dr in dropout_rate:
            for lr in learn_rate:
                for dec in decay:
                    for bs in batch_size:
                        for ep in epochs:
                            print_log_flush(f"---Training Model: {i} of {tot_train}---",logfile)
                            ml_model = ML_Model_Builder(activation,neur,dr,lr,dec,bs,ep)
                            ml_model._train_model(ml_data)
                            delattr(ml_model,'logfile')
                            models.append(ml_model)
                            print(models)
                            i+=1

    error_name  = 'rmse'   

    error_vals = []
    for m in models:
        error_vals.append(m.history.history[error_name][-1])
    best_idx = error_vals.index(min(error_vals))

    print_log_flush("-------------------------RESULTS-------------------------",logfile)

    print_log_flush(f"Best score for '{error_name}' was {min(error_vals)} in model {best_idx}",logfile)

    print_log_flush(f"Downsampling Seconds: {models[best_idx].downsample_sec}\n\
    Lag Periods: {models[best_idx].periods_to_lag}\n\
    Activation: {models[best_idx].activation}\n\
    Neurons: {models[best_idx].neurons}\n\
    Learning Rate: {models[best_idx].learn_rate}\n\
    Decay: {models[best_idx].decay}\n\
    Epochs: {models[best_idx].epochs}",logfile)

    print_log_flush("-------------------------SAVE FILE-------------------------",logfile)

    with open('{}.pkl'.format(file_name), 'wb') as models_file:
         pickle.dump(models, models_file)

    print_log_flush(f"Saved list of models to {file_name}",logfile)
    print_log_flush(f"Models built with optimizer: {models[best_idx].opt_string}",logfile)

    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    print_log_flush('*******************************END TIME = {}************************************'.format(dt_string),logfile)
except Exception as e:
    print_log_flush('Error occurred ' + str(e),logfile)
if logfile is not None:
    logfile.close()


