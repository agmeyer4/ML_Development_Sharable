from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
def create_model(learn_rate=0.01,activation='relu',dropout_rate=0.0,neurons=128):
#    from keras.models import Sequential
#    from keras.layers import Dense, LSTM, Dropout	

#   def rmse(y_true, y_pred):
#       from keras import backend
#       return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

#    def r_square(y_true, y_pred):
#        from keras import backend as K
#        SS_res =  K.sum(K.square(y_true - y_pred)) 
#        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
#        return (1 - SS_res/(SS_tot + K.epsilon()))

    model = Sequential()
    model.add(LSTM(neurons,activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation = 'sigmoid'))
    opt = tf.keras.optimizers.Adam(lr=learn_rate,decay=1e-5)

    model.compile(loss='mse',optimizer=opt,metrics=['mean_squared_error'])
    
    return model
