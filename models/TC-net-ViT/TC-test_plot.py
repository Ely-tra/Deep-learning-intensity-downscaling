#
# DESCRIPTION:
#	This script plots test data for a trained model, performs preprocessing/norm,
#	then uses the model to predict outcomes. The results are then computed RMSE
#	and MAE metrics, and visualized through box/scatter plots to compare predicted
#	values against true values.
#
# HIST: - Jan 26, 2024: created by Khanh Luong for CNN
#       - Oct 02, 2024: adapted for VIT by Tri Nguyen
#       - Oct 19, 2024: cross-checked and cleaned up by CK
#       - Oct 30, 2024: added arguments input by TN
#====================================================================================
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras import backend as K
from matplotlib.lines import Line2D
from tensorflow.keras import layers
import argparse
#
# Define parameters and data path. Note that x_size is the input data size. By default
# is (64x64) after resized for windowsize < 26x26. For a larger windown size, set it
# to 128.
#

def parse_args():
    parser = argparse.ArgumentParser(description="Test and Plot Model Predictions for TC Intensity")
    parser.add_argument("--mode", default="VMAX", type=str, help="Mode of operation (e.g., VMAX, PMIN, RMW)")
    parser.add_argument("--workdir", default="/N/project/Typhoon-deep-learning/output/", type=str, help="Directory to save output data")
    parser.add_argument("--windowsize", default=[19, 19], type=int, nargs=2, help="Window size as two integers (e.g., 19 19)")
    parser.add_argument("--var_num", type=int, default = 13, help="Number of variables (not used directly here but might be needed for file paths)")
    parser.add_argument("--x_size", type=int, default = 72, help="X dimension size for the input")
    parser.add_argument("--y_size", type=int, default = 72, help="Y dimension size for the input")
    parser.add_argument('--st_embed', action='store_true', help='Including space-time embedded')
    parser.add_argument("--model_name", default='ViTmodel1', type=str, help="Base of the model name")
    parser.add_argument('--validation_year', nargs='+', type=int, default=[2014], help='Year(s) taken for validation')
    parser.add_argument('--test_year', nargs='+', type=int, default=[2017], help='Year(s) taken for test')

    return parser.parse_args()
args = parse_args()
# Set parameters based on parsed arguments
validation_year = args.validation_year
test_year = args.test_year
mode = args.mode
workdir = args.workdir
windowsize = list(args.windowsize)
var_num = args.var_num
x_size = args.x_size
y_size = args.y_size
st_embed = args.st_embed
model_name = args.model_name
#model_name = f'{model_name}_val{validation_year}_test{test_year}{mode}{('_st' if st_embed else '')}'
model_name = f'{model_name}_val{validation_year}_test{test_year}{mode}{"_st" if st_embed else ""}'
exp_name = f"exp_{var_num}features_{windowsize[0]}x{windowsize[1]}/"
directory = workdir + exp_name
data_dir = directory + '/data/'
model_dir = directory + '/model/' + model_name
windows = f'{windowsize[0]}x{windowsize[1]}'

######################################################################################
# All fucntions below
######################################################################################
def mode_switch(mode):
    switcher = {
        'VMAX': 0,
        'PMIN': 1,
        'RMW': 2
    }
    # Return the corresponding value if mode is found, otherwise return None as default
    return switcher.get(mode, None)
def get_year_directories(data_directory):
    """
    List all directory names within a given directory that are formatted as four-digit years.

    Parameters:
    - data_directory (str): Path to the directory containing potential year-named subdirectories.

    Returns:
    - list: A list of directory names that match the four-digit year format.
    """
    all_entries = os.listdir(data_directory)
    year_directories = [
        entry for entry in all_entries
        if os.path.isdir(os.path.join(data_directory, entry)) and re.match(r'^\d{4}$', entry)
    ]
    return year_directories

def load_data_for_test_year(data_directory, mode, test_year, var_num, windows):
    """
    Loads data from specified directory for specified test years and organizes it into test sets.

    Args:
        data_directory (str): The root directory where data files are stored.
        mode (str): Mode of operation which defines how labels should be manipulated or filtered.
        test_year (list): List of years to be used for testing.
        var_num (int): Variable number identifier used in file naming.
        windows (str): Window size identifier used in file naming.

    Returns:
        tuple: Tuple containing three elements:
               - test_features (np.ndarray): Array of test features from the test years.
               - test_labels (np.ndarray): Array of test labels corresponding to test_features.
               - test_space_times (np.ndarray): Array of test spatial and temporal data corresponding to test_features.
    """
    years = get_year_directories(data_directory)
    months = range(1, 13)  # Months labeled 1 to 12
    b = mode_switch(mode)  # Adjust this function to handle different modes appropriately
    test_features, test_labels, test_space_times = [], [], []

    # Loop over each year
    for year in years:
        if year not in test_year:
            continue  # Focus only on the test year

        # Loop over each month
        for month in months:
            feature_filename = f'features{var_num}_{windows}{month:02d}fixed.npy'
            label_filename = f'labels{var_num}_{windows}{month:02d}.npy'
            space_time_filename = f'spacetime{var_num}_{windows}{month:02d}.npy'

            # Construct full paths
            feature_path = os.path.join(data_directory, year, feature_filename)
            label_path = os.path.join(data_directory, year, label_filename)
            space_time_path = os.path.join(data_directory, year, space_time_filename)

            # Check if files exist before loading
            if os.path.exists(feature_path) and os.path.exists(label_path) and os.path.exists(space_time_path):
                features = np.load(feature_path)
                labels = np.load(label_path)[:, b]
                space_time = np.load(space_time_path)

                # Append to test lists
                test_features.append(features)
                test_labels.append(labels)
                test_space_times.append(space_time)
            else:
                print(f"Warning: Files not found for year {year} and month {month}")
                print(label_path, feature_path)

    # Concatenate all loaded data into single arrays
    test_features = np.concatenate(test_features, axis=0) if test_features else np.array([])
    test_labels = np.concatenate(test_labels, axis=0) if test_labels else np.array([])
    test_space_times = np.concatenate(test_space_times, axis=0) if test_space_times else np.array([])

    return test_features, test_labels, test_space_times

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
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        patches = tf.image.extract_patches(images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')

        patches = tf.reshape(
            patches,
            (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels))

        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim  # Initialize projection_dim attribute
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(
            np.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config

def normalize_Z(Z):
    Z[:,2] = (Z[:,2]+90) / 180
    Z[:,3] = (Z[:,3]+180) / 360
    return Z



#==============================================================================================
# Main call
#==============================================================================================


X, Y, Z = load_data_for_test_year(data_dir, mode, test_year, var_num, windows)
X=np.transpose(X, (0, 2, 3, 1))

# Normalize the data before encoding
x, y = normalize_channels(X, Y)
z = normalize_Z(Z)
number_channels=x.shape[3]
# Load model and perform predictions
model = tf.keras.models.load_model(model_dir, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
name = model_name
if st_embed:
    predict = model.predict([x, z])
else:
    predict = model.predict(x)

# Calculate metrics and store results
datadict[name + 'rmse'] = root_mean_squared_error(predict, y)
datadict[name + 'MAE'] = MAE(predict, y)
datadict[name] = predict

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
axs[0].boxplot([datadict[name].reshape(-1), y])
axs[0].grid(True)
axs[0].set_ylabel('Knots', fontsize=20)
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
