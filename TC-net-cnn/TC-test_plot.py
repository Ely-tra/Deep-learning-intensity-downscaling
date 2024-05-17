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

#==============================================================================================
# Constants and configuration settings
#==============================================================================================

# Define the directory containing the files
directory = "/N/slate/kmluong/Training_data/Split/"
# List all files in the directory
files = os.listdir(directory+'model/')
files.sort()  # Optional: sort the files alphabetically

# Print all files with an index
for index, file in enumerate(files):
    print(f"{index} {file}")

# Ask the user to choose a file number
file_number = int(input("Enter the number to select a model: "))

# Ensure the input number is within the valid range
if 0 <= file_number < len(files):
    model_name = files[file_number]
    print(f"You selected: {model_name}")
else:
    print("Invalid file number.")

directory = "/N/slate/kmluong/Training_data/Split/"

all_files = os.listdir(directory+'data/')

# Filter and pair feature and label files
paired_files = []
for file in all_files:
    if "test" in file and 'x.' in file:
        label_file = file.replace("x.", "y.")
        if label_file in all_files:
            paired_files.append((file, label_file))

# Sort pairs alphabetically by the feature file name
paired_files.sort()

# Print paired files with an index
for index, (feature, label) in enumerate(paired_files):
    print(f"{index} - Features: {feature}, Labels: {label}")

# Ask the user to choose a pair number
pair_number = int(input("Enter the number to select a training set: "))

# Ensure the input number is within the valid range
if 0 <= pair_number < len(paired_files):
    fea_path, lab_path = paired_files[pair_number]
    fea_path = directory + 'data/' + fea_path
    lab_path = directory + 'data/' + lab_path
    print(f"You selected: Features: {fea_path}, Labels: {lab_path}")
else:
    print("Invalid pair number.")
mode = str(input('Pick a mode, VMAX, PMIN, RMW: '))


x_size = int(input('Set the square input image size, if the dataset is of image larger than 26 by 26 degree, this number should be 128, else 64: '))

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

# Initialize dictionary to store results
datadict = {}
custom_objects = {
    'mae_1': mae_for_output(0),
    'rmse_1': rmse_for_output(0)
}
#==============================================================================================
# File paths and mode configuration
#==============================================================================================

root = '/N/slate/kmluong/Training_data/'
prefix = 'Split/data/'

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
x = resize_preprocess(x, x_size, x_size, 'lanczos5')

# Load model and perform predictions
model = tf.keras.models.load_model(root + 'model/' + model_name, custom_objects=custom_objects)
name = model_name
predict = model.predict(x)

# Calculate metrics and store results
datadict[name + 'rmse'] = root_mean_squared_error(predict, y)
datadict[name + 'MAE'] = MAE(predict, y)
datadict[name] = predict

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
plt.suptitle(name + ' RMSE ' + str("{:.2f}".format(datadict[name + 'rmse'])) + ' MAE ' + str("{:.2f}".format(datadict[name + 'MAE'])))
axs[0].boxplot([datadict[name].reshape(-1), y], labels=['Predicted', 'Truth'])
axs[0].set_title(t)
axs[0].set_ylabel(u)
axs[0].grid(True)
axs[1].scatter(y, datadict[name].reshape(-1))
axs[1].grid()
axs[1].set_xlabel('Truth')
axs[1].set_ylabel('Prediction')
axs[1].plot(np.arange(y.min(), y.max()), np.arange(y.min(), y.max()), 'r-', alpha=0.8)
axs[1].fill_between(np.arange(y.min(), y.max()), np.arange(y.min(), y.max()) + datadict[name + 'MAE'], np.arange(y.min(), y.max()) - datadict[name + 'MAE'], color='red', alpha=0.3, label='MAE')
axs[1].legend()
plt.savefig('Model:' + str(name) + '.png')
print(f'Saved the result as Model:{name}.png')
print('RMSE = ' + str("{:.2f}".format(datadict[name + 'rmse'])) + ' and MAE = ' + str("{:.2f}".format(datadict[name + 'MAE'])))
print('Completed!')
