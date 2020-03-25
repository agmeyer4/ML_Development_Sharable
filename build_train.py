from CO2_Dataset_Preparation import *
from ML_Model_Master import *
from pympler import muppy,summary,tracker
import logging
import sys

#Setup Logging to File
logging.basicConfig(filename=sys.argv[1],level=logging.DEBUG)

try:
    #tr = tracker.SummaryTracker()
    output_file_name = sys.argv[2]
    ml_datafile_name = sys.argv[3]

    position_number = int(sys.argv[4])
    downsample_sec = int(sys.argv[5])
    periods_to_lag = int(sys.argv[6])
    tower = str(sys.argv[7])
    train_percent = float(sys.argv[8])

    activation = str(sys.argv[9])
    neurons = int(sys.argv[10])
    dropout_rate = float(sys.argv[11])
    learn_rate = float(sys.argv[12])
    decay = float(sys.argv[13])
    batch_size = int(sys.argv[14])
    epochs = int(sys.argv[15])
    error_metric = str(sys.argv[16])

    if tower == 'Picarro':
        feature_columns = ['Pic_CO2','ANEM_X','ANEM_Y','ANEM_Z','wd','ws']

    #Load Data
    with open(ml_datafile_name, 'rb') as file:
        ml_data = pickle.load(file)

    # all_objects = muppy.get_objects()
    # sums = (summary.summarize(all_objects))
    # summary.print_(sums)
    # del sums 
    # gc.collect()

    ml_model = ML_Model_Builder(activation,neurons,dropout_rate,learn_rate,decay,batch_size,epochs)
    ml_model._train_model(ml_data)
    mod_hist = ml_model.history.history

    f = open(output_file_name, 'a')
    f.write(f"{activation},{neurons},{dropout_rate},{learn_rate},{decay},{batch_size},{epochs}\n")
    for i in range(0,len(mod_hist['loss'])):
        f.write(f"{mod_hist['loss'][i]},{mod_hist[error_metric][i]},{mod_hist['val_loss'][i]},{mod_hist[f'val_{error_metric}'][i]}\n")
    f.close()

    del ml_model
    gc.collect()
except Exception as e:
    logging.exception('Exception Raised')
    raise
else:
    logging.info('build_train executed successfuly')