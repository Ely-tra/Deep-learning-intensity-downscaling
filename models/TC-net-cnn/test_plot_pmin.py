"""
This script loads test data for a specific model, performs preprocessing and normalization,
then uses the model to predict outcomes. The results are then analyzed by computing RMSE
and MAE metrics, and visualized through boxplots and scatter plots to compare predicted
values against true values.
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras import backend as K
from matplotlib.lines import Line2D
#
# Define parameters and data path. Note that x_size is the input data size. By default
# is (64x64) after resized for windowsize < 26x26. For a larger windown size, set it
# to 128.
#
workdir = '/N/slate/kmluong/TC-net-cnn_workdir/Domain_data/'
var_num = 13
windowsize = [25,25]
mode = 'PMIN'
x_size = 64
exp_name = "exp_13features_" + str(windowsize[0])+'x'+str(windowsize[1])
model_name = "model_"+mode+"13_" + str(windowsize[0])+'x'+str(windowsize[1])
directory = workdir + exp_name
all_files = os.listdir(directory)

# Filter and pair feature and label files
for file in all_files:
    if "test" in file and 'x_' in file:
        fea_path = directory + '/' + file  
        label_file = file.replace("x_", "y_")
        if label_file in all_files:
            lab_path = directory + '/' + label_file
        else:
            print(f"There is no test label data...exit")
            quit()
#==============================================================================================
# Metric and preprocessing function definitions
#==============================================================================================
def root_mean_squared_error(y_true, y_pred):
    """Calculate root mean squared error."""
    m = tf.keras.metrics.RootMeanSquaredError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def MAE(y_true, y_pred):
    """Calculate mean absolute error."""
    m = tf.keras.metrics.MeanAbsoluteError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def mae_for_output(index):
    """Metric function to return MAE for a specific output index."""
    def mae(y_true, y_pred):
        return tf.keras.metrics.mean_absolute_error(y_true[:, index], y_pred[:, index])
    mae.__name__ = f'mae_{index+1}'
    return mae

def rmse_for_output(index):
    """Metric function to return RMSE for a specific output index."""
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:, index], y_pred[:, index]))
    rmse.__name__ = f'rmse_{index+1}'
    return rmse

def resize_preprocess(image, HEIGHT, WIDTH, method):
    """Resize and preprocess images using the specified method."""
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image

def normalize_channels(X, y):
    """Normalize the channel data for all samples in the dataset."""
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i, :, :, var].flat[np.abs(X[i, :, :, var]).argmax()]
            X[i, :, :, var] = X[i, :, :, var] / abs(maxvalue)
    print("Finish normalization...")
    return X, y
#
# MAIN CALL: Initialize dictionary to store results
#
datadict = {}
custom_objects = {
    'mae_1': mae_for_output(0),
    'rmse_1': rmse_for_output(0)
}

# Determine labels and units based on mode
if mode == 'VMAX':
    b = 0
    t = 'Maximum wind speed'
    u = 'Knots'
elif mode == 'PMIN':
    b = 1
    t = 'Minimum pressure'
    u = 'Milibar'
elif mode == 'RMW':
    b = 2
    t = 'Radius of Maximum Wind'
    u = 'Kilometers'

# Load and preprocess data
y = np.load(lab_path)[:, b]
x = np.load(fea_path)
x = np.transpose(x, (0, 2, 3, 1))
x, y = normalize_channels(x,y)
x = resize_preprocess(x, x_size, x_size, 'lanczos5')

# Load model and perform predictions
model = tf.keras.models.load_model(directory + '/' + model_name, custom_objects=custom_objects)
name = model_name
predict = model.predict(x)

# Calculate metrics and store results
datadict[name + 'rmse'] = root_mean_squared_error(predict, y)
datadict[name + 'MAE'] = MAE(predict, y)
datadict[name] = predict


fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
axs[0].boxplot([datadict[name].reshape(-1), y])
axs[0].grid(True)
axs[0].set_ylabel('Milibars', fontsize=20)
axs[0].text(0.95, 0.05, '(a)', transform=axs[0].transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[0].set_xticklabels(['Predicted', 'Truth'], fontsize=20)


# Second subplot
axs[1].scatter(y, datadict[name].reshape(-1))
axs[1].grid()
axs[1].set_xlabel('Truth', fontsize=20)
axs[1].set_ylabel('Prediction', fontsize=20)
axs[1].text(0.95, 0.05, '(b)', transform=axs[1].transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
axs[1].plot(np.arange(min(y), max(y)), np.arange(min(y), max(y)), 'r-', alpha=0.8)
mae = datadict[name+'MAE']
rmse = datadict[name+'rmse']
axs[1].fill_between(np.arange(min(y), max(y)), np.arange(min(y), max(y)) + mae, np.arange(min(y), max(y)) - mae, color='red', alpha=0.3)
axs[1].tick_params(axis='both', which='major', labelsize=14)

# Legends with RMSE and MAE without markers
custom_lines = [
                Line2D([0], [0], color='red', lw=4, alpha=0.3),
                Line2D([0], [0], color='none', marker='', label=f'RMSE: {rmse:.2f}'),
                Line2D([0], [0], color='none', marker='', label=f'MAE: {mae:.2f}')]

axs[1].legend(custom_lines, [ 'MAE Area', f'RMSE: {rmse:.2f}', f'MAE: {mae:.2f}'], fontsize=12)

plt.savefig(directory + '/fig_' + str(name) + '.png')
print(f'Saved the result as Model:{name}.png')
print('RMSE = ' + str("{:.2f}".format(datadict[name + 'rmse'])) + ' and MAE = ' + str("{:.2f}".format(datadict[name + 'MAE'])))
print('Completed!')
                                                                             

