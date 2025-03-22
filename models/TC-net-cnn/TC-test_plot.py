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
import re
#
# Define parameters and data path. Note that x_size is the input data size. By default
# is (64x64) after resized for windowsize < 26x26. For a larger wind:wown size, set it
# to 128.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Test and Plot Model Predictions for TC Intensity")
    parser.add_argument("--mode", default="VMAX", type=str, help="Mode of operation (e.g., VMAX, PMIN, RMW)")
    parser.add_argument('-r', "--root", default="/N/project/Typhoon-deep-learning/output/", type=str, 
                        help="Directory to save output data")
    parser.add_argument('-imsize', '--image_size', type=int, default=64, help='Size to resize the image to')
    parser.add_argument('--st_embed', type=int, default=0, help='Including space-time embedded')
    parser.add_argument("--model_name", default='CNNmodel', type=str, help="Base of the model name")
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', 
                        help='Temporary working folder')
    parser.add_argument("--text_report_name", default= 'report.txt', type=str, 
                        help="Filename to write text report to, will be inside text_report dir")
    parser.add_argument('-ss', '--data_source', type=str, default='MERRA2', help='Data source')
    parser.add_argument('-tid', '--temp_id', type=str)
    parser.add_argument('-u', '--unit', type=str, default='Knots', help = 'Displayed unit')

    return parser.parse_args()

args = parse_args()
mode = args.mode
workdir = args.root
image_size = args.image_size
st_embed = args.st_embed
model_name = args.model_name
text_report_name=args.text_report_name
data_source=args.data_source
work_folder=args.work_folder
temp_id=args.temp_id
unit=args.unit
model_name = f'{model_name}_{data_source}_{mode}{"_st" if st_embed else ""}'
report_directory = os.path.join(workdir, 'text_report')
os.makedirs(report_directory, exist_ok=True)
text_report_path=os.path.join(report_directory, text_report_name)
model_dir = workdir + '/model/' + model_name
temp_dir = os.path.join(work_folder, 'temp')

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
        int(entry) for entry in all_entries
        if os.path.isdir(os.path.join(data_directory, entry)) and re.match(r'^\d{4}$', entry)
    ]
    return year_directories

def load_data(temp_dir, temp_id=temp_id):
    global test_x, test_y, test_z

    # Load mandatory test data files
    test_x = np.load(os.path.join(temp_dir, f'test_x_{temp_id}.npy'))
    test_y = np.load(os.path.join(temp_dir, f'test_y_{temp_id}.npy'))

    # Optionally load test_z if it exists
    if f'test_z_{temp_id}.npy' in os.listdir(temp_dir):
        test_z = np.load(os.path.join(temp_dir, f'test_z_{temp_id}.npy'))
    else:
        test_z = None  # Ensure test_z is defined even if it does not exist

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

        patches = tf.image.extract_patches(images, sizes=[1, self.patch_size, self.patch_size, 1], 
                  strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')

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

custom_objects = {
    'mae_1': mae_for_output(0),
    'rmse_1': rmse_for_output(0)
}

def plotPrediction(datadict,predict,truth,pc,mode,name,unit,report_directory):
    if mode == "ALL":
        test_y = truth[:,pc]
        if pc == 0:
            myUnit = "knot"
            myMode = "VMAX"
        elif pc == 1:
            myUnit = "hPa"
            myMode = "PMIN"
        elif pc == 2:
            myUnit = "nm"
            myMode = "RMW"
    else:
        test_y = truth
        myMode = mode
        myUnit = unit

    # Calculate metrics and store results
    datadict[name + 'rmse'] = root_mean_squared_error(predict[:,pc], test_y)
    datadict[name + 'MAE'] = MAE(predict[:,pc], test_y)
    datadict[name] = predict[:,pc]

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    axs[0].boxplot([datadict[name].reshape(-1), test_y])
    axs[0].grid(True)
    axs[0].set_ylabel(myUnit, fontsize=20)
    axs[0].text(0.95, 0.05, '(a)', transform=axs[0].transAxes, fontsize=20, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].set_xticklabels(['Predicted', 'Truth'], fontsize=20)

    # Second subplot
    axs[1].scatter(test_y, datadict[name].reshape(-1))
    axs[1].grid()
    axs[1].set_xlabel('Truth', fontsize=20)
    axs[1].set_ylabel('Prediction', fontsize=20)
    axs[1].text(0.95, 0.05, '(b)', transform=axs[1].transAxes, fontsize=20, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    axs[1].plot(np.arange(min(test_y), max(test_y)), np.arange(min(test_y), max(test_y)), 'r-', alpha=0.8)
    mae = datadict[name+'MAE']
    rmse = datadict[name+'rmse']
    axs[1].fill_between(np.arange(min(test_y), max(test_y)), 
                        np.arange(min(test_y), max(test_y)) + mae, 
                        np.arange(min(test_y), max(test_y)) - mae, 
                        color='red', alpha=0.3)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    
    # Legends with RMSE and MAE without markers
    custom_lines = [
                    Line2D([0], [0], color='red', lw=4, alpha=0.3),
                    Line2D([0], [0], color='none', marker='', label=f'RMSE: {rmse:.2f}'),
                    Line2D([0], [0], color='none', marker='', label=f'MAE: {mae:.2f}')]

    axs[1].legend(custom_lines, [ 'MAE Area', f'RMSE: {rmse:.2f}', f'MAE: {mae:.2f}'], fontsize=12)

    figPath = f"{report_directory}/fig_{myMode}{name}.png" 
    textPath = f"{report_directory}/{myMode}{name}.txt" 
    plt.savefig(figPath)
    print(f"Saving result to: {figPath}")
    print('RMSE = ' + str("{:.2f}".format(datadict[name + 'rmse'])) + ' and MAE = ' + str("{:.2f}".format(datadict[name + 'MAE'])))
    output_str = 'RMSE = ' + str("{:.2f}".format(datadict[name + 'rmse'])) + ' and MAE = ' + str("{:.2f}".format(datadict[name + 'MAE']))
    if not os.path.exists(report_directory):
        os.makedirs(report_directory)
    with open(textPath, 'w') as file:
        file.write(f"Saving result to: {figPath}\n")
        file.write(output_str + '\n')
        file.write('Predictions vs Actual Values:\n')
        for i in range(len(predict)):
            file.write(f"{predict[i][pc]}, {test_y[i]} \n")

#==============================================================================================
# Main call
#==============================================================================================

b=mode_switch(mode)
load_data(temp_dir)

# Normalize the data before encoding
test_x=np.transpose(test_x, (0, 2, 3, 1))
if mode == "ALL":
    test_x, test_y = normalize_channels(test_x, test_y[:,0:3])
else:
    test_x, test_y = normalize_channels(test_x, test_y[:,b])
test_x = resize_preprocess(test_x, image_size, image_size, 'lanczos5')
if st_embed:
    test_z = normalize_Z(test_z)
number_channels=test_x.shape[3]

# Load model and perform predictions
model = tf.keras.models.load_model(model_dir, custom_objects=custom_objects)
name = model_name
if st_embed:
    predict = model.predict([test_x, test_z])
else:
    predict = model.predict(test_x)
print(f"Prediction output shape is {predict.shape}")
if mode == "ALL":
    for pc in range(3):
        plotPrediction(datadict,predict,test_y,pc,mode,name,unit,report_directory)
else:
    plotPrediction(datadict,predict,test_y,0,mode,name,unit,report_directory)


print('Completed!')
