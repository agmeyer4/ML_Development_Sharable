from CO2_Dataset_Preparation import *
from ML_Model_Master import *
import pickle
import re

class Gridsearch_Result_Retrieve:
    def __init__(self,logfile):
        self.logfile = logfile
    def _read_data_vars(self):
        print("-----Generating Data Variables-----")
        f = open(self.logfile,'r')
        line_num = 0
        data_line = -1
        for line in f.readlines():
            if 'position' in line:
                data_line = line_num+1
            elif data_line == line_num:
                data_vars = line.replace('\n','').split(",",6)
                self.position_number = int(data_vars[0])
                self.tower = data_vars[1]
                self.downsample_sec = int(data_vars[2])
                self.periods_to_lag = int(data_vars[3])
                self.train_percent = float(data_vars[4])
                self.error_metric = data_vars[5]
                
                self.excess_rolls = []
                er_lst =re.split(',|\]|\[',data_vars[6])
                for i in range(1,len(er_lst)-1):
                    self.excess_rolls.append(int(er_lst[i]))
                
                if self.tower == 'Picarro':
                    self.feature_columns = ['Pic_CO2','ANEM_X','ANEM_Y','ANEM_Z','wd','ws']
                f.close()
                break
            line_num+=1
        f.close()
        
    class Model:
        def __init__(self,line,error_metric):
            model_vars = line.replace('\n','').split(",")
            self.activation = model_vars[0]
            self.neurons = int(model_vars[1])
            self.dropout_rate = float(model_vars[2])
            self.learn_rate = float(model_vars[3])
            self.decay = float(model_vars[4])
            self.batch_size = int(model_vars[5])
            self.epochs = int(model_vars[6])
            self.error_metric = error_metric
    
    def _read_models(self):
        print("-----Reading Model Outputs-----")
        f = open(self.logfile,'r')
        line_num = 0
        self.models = []
        for line in f.readlines():
            if 'relu' in line:
                mod_attr_line = line_num
                self.models.append(self.Model(line,self.error_metric))
                self.models[-1].mod_hist = {'loss':[],f'{self.error_metric}':[],'val_loss':[],f'val_{self.error_metric}':[]}
            elif len(self.models) == 0:
                continue
            else:
                hist_vals = line.replace('\n','').split(",")
                self.models[-1].mod_hist['loss'].append(float(hist_vals[0]))
                self.models[-1].mod_hist[self.error_metric].append(float(hist_vals[1]))
                self.models[-1].mod_hist['val_loss'].append(float(hist_vals[2]))
                self.models[-1].mod_hist[f'val_{self.error_metric}'].append(float(hist_vals[3]))

            line_num+=1
        f.close()
    def _find_best_model(self):
        print("-----Finding Best Model-----")
        self.last_error = []
        for mod in self.models:
            self.last_error.append(mod.mod_hist[mod.error_metric][-1])
        self.best_model_idx = self.last_error.index(min(self.last_error))
        self.best_model_attr = self.models[self.best_model_idx]
        print(f"Best Model Index = {self.best_model_idx}")

    
    def _get_best_data(self,data_path):
        print("-----Setup Data for Best Model Retrain-----")
        #Load Dataset from Folder
        data = Processed_Set(self.tower,self.position_number,self.excess_rolls,True)
        data._retrieve_data(data_path)
        data._apply_excess()

        #Build ML dataset
        self.best_ml_data = ML_Data(self.downsample_sec,self.periods_to_lag,self.tower,self.train_percent)
        self.best_ml_data._ML_Process(data)
        
        
    def _retrain_fit_best(self):
        print("-----Regenerating Best Model-----")
        self.best_model = ML_Model_Builder(self.best_model_attr.activation,self.best_model_attr.neurons,\
                                           self.best_model_attr.dropout_rate,\
                                           self.best_model_attr.learn_rate,self.best_model_attr.decay,\
                                           self.best_model_attr.batch_size,self.best_model_attr.epochs)
        print("-----Retraining Best Model-----")
        self.best_model._train_model(self.best_ml_data)
        print("-----Fitting Best Model-----")
        self.best_model._fit_data(self.best_ml_data)
    def _plot_best_comparison(self,roll):
        print("-----Plotting Best Model Comparison-----")
        #Return to original data shape and scale
        data = self.best_ml_data
        X_test_original_shape = data.X_test.reshape(data.orig_X_test_shape) #reshape from 3d time
        y_test_original_shape = data.y_test.reshape(data.orig_y_test_shape)#reshape from 3d time 

        merged_tests = np.concatenate((X_test_original_shape,y_test_original_shape[:,None]),axis=1) #concat X and y
        unscaled_test = pd.DataFrame(data.min_max_scalar.inverse_transform(merged_tests)).iloc[:,-1] #unscale using declared scalar

        y_fit_original_shape = data.y_fit.reshape(data.orig_y_test_shape)
        merged_tests = np.concatenate((X_test_original_shape,y_fit_original_shape[:,None]),axis=1)
        unscaled_fit = pd.DataFrame(data.min_max_scalar.inverse_transform(merged_tests)).iloc[:,-1]

        #Put into pandas df
        comparison = pd.concat([unscaled_test,unscaled_fit],axis=1)
        comparison.columns = ['test','fit']

        #PLOT PREDICTED VS OBSERVED
        fig, ax = plt.subplots(figsize = (20,10))

        ax.plot(comparison['fit'].rolling(roll).mean(),label='ML_Predicted')
        ax.plot(comparison['test'].rolling(roll).mean(),label='Actual_Test_Data')

        ax.set_xlabel('Time (s)',fontsize=16)
        ax.set_ylabel('Mass Flow from Vent (g/s)',fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)


        ax.legend(fontsize=16)
        plt.show()