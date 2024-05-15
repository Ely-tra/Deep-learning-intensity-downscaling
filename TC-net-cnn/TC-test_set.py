#This script is used to generate a testing result for any model created from the fifth step.

import tensorflow as tf
import numpy as np
import os
import libtcg_utils as tcg_utils
import matplotlib.pyplot as plt

#=================================================================================================
# Defining target model and data
#=================================================================================================

number_of_channel = 13
windowsize = [18,18]
model_name = ''

#=================================================================================================
# Defining preprocessing functions
#=================================================================================================

def resize_preprocess(image, HEIGHT, WIDTH, method):
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image

def mae_for_output(index):
    def mae(y_true, y_pred):
        return tf.keras.metrics.mean_absolute_error(y_true[:, index], y_pred[:, index])
    mae.__name__ = f'mae_{index+1}'  # Naming for clarity in logs
    return mae

def rmse_for_output(index):
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:, index], y_pred[:, index]))
    rmse.__name__ = f'rmse_{index+1}'  # Naming for clarity in logs
    return rmse
custom_objects = {
    'mae_1': mae_for_output(0),
    'rmse_1': rmse_for_output(0)
}

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        m = tf.keras.metrics.RootMeanSquaredError()
        m.update_state(y_true, y_pred)
        return m.result().numpy()

def MAE(y_true, y_pred):
    m = tf.keras.metrics.MeanAbsoluteError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def normalize_channels(X,y):
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            #print('Normalization factor for sample and channel',i,var,', is: ',abs(maxvalue))
            X[i,:,:,var] = X[i,:,:,var]/abs(maxvalue)
            maxnew = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            #print('-->After normalization of sample and channel',i,var,', is: ',abs(maxnew))
            #input('Enter to continue...')
    print("Finish normalization...")
    return X,y


#=================================================================================================
# Produce result
#=================================================================================================

data_directory = '/N/slate/kmluong/Training_data/'
if number_of_channel == 5:
  number_of_channel = ''
else:
  number_of_channel = str(number_of_channel)
if windowsize==[18,18]:
  windowsize = ''
else:
  windowsize = '.' +  str(windowsize[0]) + 'x' + str(windowsize[1])

prefix = 'Split/data/'

fea_path = data_directory + 'test' + number_of_channel + 'x' + windowsize + '.npy'
lab_path = data_directory + 'test' + number_of_channel + 'y' + windowsize + '.npy'

if mode=='VMAX':
  b=0
  t = 'Maximum wind speed'
  u = 'Knots'
if mode=='PMIN':
  b=1
  t = 'Minimum Pressure'
  u = 'milibar
if mode=='RMW':
  b=2
x = np.load(fea_path)

y = np.load(lab_path)[:,b]

x=np.transpose(x, (0,2,3,1))
x,y = normalize_channels(x,y)

model = tf.keras.models.load_model(data_directory + 'model/' + model_name, custom_objects=custom_objects)
name = model_name
predict = model.predict(x)
datadict[name+'rmse'] = root_mean_squared_error(predict,y)
datadict[name+'MAE'] = MAE(predict,y)
datadict[name] = predict
fig, axs = plt.subplots(1, 2, figsize=(14, 6),gridspec_kw={'width_ratios': [1.2, 1]})
plt.suptitle(name+' RMSE '+ str("{:.2f}".format(datadict[name+'rmse'])) + ' MAE ' +str("{:.2f}".format(datadict[name+'MAE'])))
axs[0].boxplot([datadict[name].reshape(-1), y], labels=['Predicted', 'Truth'])

