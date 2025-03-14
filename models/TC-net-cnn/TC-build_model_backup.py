# DESCRIPTION: This script utilizes TensorFlow to implement a (CNN) designed for correcting 
#       TC intensity/structure from grided climate data, using the workflow inherited from the
#       previous TC formation project (https://github.com/kieucq/tcg_deep_learning). The model 
#       consists of several layers with varying functionalities including convolutional layers 
#       for TC feature extraction and dense layers for regression. Special attention is given 
#       to preprocessing steps like normalization and resizing, and the model is tuned to adapt 
#       its learning rate over epochs.
#
#       Note that one can re-design the model by looking at the model's layer configurations 
#       and loss functions (see line ???), set the dataset paths (see line ???), and run the 
#       script. The model architecture (see line ???) can be adjusted by modifying the parameters 
#       for convolutional and dense layers.   
#
# MODEL LAYERS:
#       - Layer 1 (Conv2D): 128 filters, 15x15 kernel, 'relu' activation, input shape=input data
#       - Layer 2 (MaxPooling2D): Pool size of 2, reduces spatial dimensions by half.
#       - Layer 3 (Conv2D): 64 filters, 15x15 kernel, uses 'relu' activation.
#       - Layer 4 (MaxPooling2D): Pool size of 2, further reduces spatial dimensions.
#       - Layer 5 (Conv2D): 256 filters, 9x9 kernel, uses 'relu' activation.
#       - Layer 6 (MaxPooling2D): Pool size of 2, reduces spatial dimensions.
#       - Layer 7 (Conv2D): Configurable number of filters, 5x5 kernel, uses 'relu' activation 
#       - Layer 8 (Conv2D): Same as previous, but with 'valid' padding to adjust output size.
#       - Flatten and Dense layers: Transform convolutional output to 1D .
#
# FUNCTIONS:
#       - mae_for_output: Custom mean absolute error function for specific outputs. Interchangable 
#         with TF MAE metric
#       - rmse_for_output: Custom root mean squared error function for specific outputs. 
#         Interchangable with TF RMSE metric.
#       - main: Orchestrates model construction, compilation, and training using specified 
#         parameters and datasets.
#       - resize_preprocess: Resizes images to a specified height and width.
#       - normalize_channels: Normalizes data channels within the input array.
#
# USAGE: Users need to modify the main call with proper paths and parameters before running 
#
# HIST: - May 14, 2024: created by Khanh Luong
#       - May 18, 2024: cross-checked and cleaned up by CK
#
# AUTH: Minh Khanh Luong
#==============================================================================================
import tensorflow as tf
import numpy as np
import time
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import argparse
import os
import re
import json
#
# Edit the parameters properly before running this script
#
def parse_args():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer model for TC intensity correction.')
    parser.add_argument('--mode', type=str, default = 'VMAX', help='Mode of operation (e.g., VMAX, PMIN, RMW)')
    parser.add_argument('--model_name', type=str, default = 'CNNmodel', help='Core name of the model')
    parser.add_argument('--root', type=str, default = '/N/project/Typhoon-deep-learning/output/', help='Working directory path')
    parser.add_argument('--windowsize', type=int, nargs=2, default = [19,19], help='Window size as two integers (e.g., 19 19)')
    parser.add_argument('--var_num', type=int, default = 13, help='Number of variables')
    parser.add_argument('--st_embed', action='store_true', help='Including space-time embedded')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--image_size', type=int, default=64, help='Size to resize the image to')
    parser.add_argument('--validation_year', nargs='+', type=int, default=[2014], help='Year(s) taken for validation')
    parser.add_argument('--test_year', nargs='+', type=int, default=[2017], help='Year(s) taken for test')
    parser.add_argument('--config', type=str, default = 'model_core/test.json')
    return parser.parse_args()
args = parse_args()
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs        	
image_size = args.image_size  				
validation_year = args.validation_year
test_year = args.test_year
mode = args.mode
root = args.root
windowsize = list(args.windowsize)
var_num = args.var_num
st_embed = args.st_embed
config_path = args.config
    
windows = f'{windowsize[0]}x{windowsize[1]}'
work_dir = root +'/exp_'+str(var_num)+'features_'+windows+'/'
data_dir = work_dir + 'data/'
model_dir = work_dir + 'model/'
model_name = args.model_name
model_name = f'{model_name}_{mode}{"_st" if st_embed else ""}'

#####################################################################################
# DO NOT EDIT BELOW UNLESS YOU WANT TO MODIFY THE SCRIPT
#####################################################################################
def load_json_config(path):
    with open(path, 'r') as file:
        return json.load(file)

def apply_operation(x, op, inputs, flows, st_embed = False):
    # Handle 'slice' operation
    if op['type'] == 'slice':
        if 'slice_range' not in op or not isinstance(op['slice_range'], list) or len(op['slice_range']) != 2:
            raise ValueError("Invalid 'slice' operation configuration. 'slice_range' must be a list of two integers.")
        return layers.Lambda(lambda x: x[:, :, :, op['slice_range'][0]:op['slice_range'][1]])(x)
    
    # Handle Conv2D
    elif op['type'] == 'Conv2D':
        return layers.Conv2D(**{k: v for k, v in op.items() if k != 'type'})(x)
    
    # Handle Concatenate
    elif op['type'] == 'concatenate':
        if 'inputs' not in op or not isinstance(op['inputs'], list):
            raise ValueError("Invalid 'concatenate' operation configuration. 'inputs' must be a list of input names.")
        # UPDATED: Look up tensors in both inputs and flows
        concat_inputs = []
        for item in op['inputs']:
            if item in inputs:
                concat_inputs.append(inputs[item])
            elif item in flows:
                concat_inputs.append(flows[item])
            else:
                raise ValueError(f"Cannot find '{item}' in either 'inputs' or 'flows'.")
        print(concat_inputs)
        return layers.concatenate(concat_inputs, axis=op.get('axis', -1))
    
    # Handle Flatten
    elif op['type'] == 'Flatten':
        return layers.Flatten()(x)
    
    # Handle Dense
    elif op['type'] == 'Dense':
        return layers.Dense(**{k: v for k, v in op.items() if k != 'type'})(x)
    
    # Handle Data Augmentation
    elif op['type'] == 'RandomRotation':
        if 'factor' not in op:
            raise ValueError("RandomRotation requires a 'factor' parameter.")
        return layers.RandomRotation(factor=op['factor'])(x)
    elif op['type'] == 'RandomZoom':
        if 'factor' not in op:
            raise ValueError("RandomZoom requires a 'factor' parameter.")
        return layers.RandomZoom(height_factor=op['factor'], width_factor=op['factor'])(x)

    # Handle MaxPooling2D
    elif op['type'] == 'MaxPooling2D':
        return layers.MaxPooling2D(**{k: v for k, v in op.items() if k != 'type'})(x)

    elif op['type'] == 'RandomFlip':
        if 'mode' not in op:
            raise ValueError("RandomFlip requires a 'mode' parameter ('horizontal', 'vertical', or 'horizontal_and_vertical').")
        return layers.RandomFlip(mode=op['mode'])(x)

    elif op['type'] == 'BatchNormalization':
        # Typically no parameters are needed, but you can pass parameters like momentum and epsilon if specified
        bn_params = {k: v for k, v in op.items() if k != 'type'}
        return layers.BatchNormalization(**bn_params)(x)

    elif op['type'] == 'Dropout':
        if 'rate' not in op:
            raise ValueError("Dropout requires a 'rate' parameter.")
        return layers.Dropout(rate=op['rate'])(x)

    # Handle Conditional Operation
    elif op['type'] == 'conditional':
        if op['condition'] == 'st_embed':
            # Instead of looking in inputs, just check the st_embed flag
            branch = op['true_branch'] if st_embed else op['false_branch']
            for sub_op in branch:
                x = apply_operation(x, sub_op, inputs, flows)
        return x

    elif op['type'] == 'Input':
        # Register a new input based on configuration specified in the operation
        if 'name' in op and 'shape' in op:
            inputs[op['name']] = keras.Input(shape=op['shape'], name=op['name'])
            return inputs[op['name']]
        else:
            raise ValueError("Input operation must specify 'name' and 'shape'.")

    else:
        raise ValueError(f"Unsupported operation type: {op['type']}")

def build_model_from_json(config, st_embed=False):
    # Collect the initial inputs
    inputs = {
        inp['name']: keras.Input(shape=inp['shape'], name=inp['name'])
        for inp in config['inputs']
        if not inp.get('optional', False) or (inp.get('optional') and inp.get('use_if') == 'st_embed' and st_embed)
    }

    # Dictionary to store intermediate flow outputs
    flows = {}

    for flow in config['process_flows']:
        print("Processing flow: ", flow['name'])  # Debug print
        if 'condition' in flow and flow['condition'] == 'st_embed' and not st_embed:
            print("Skipping flow due to condition: ", flow['name'])  # Debug skip message
            continue
        
        # Check if this flow has a single 'input' or multiple 'inputs'
        if 'input' in flow:
            if flow['input'] in flows:
                x = flows[flow['input']]
            elif flow['input'] in inputs:
                x = inputs[flow['input']]
            else:
                raise ValueError(f"Input {flow['input']} not found in inputs or flows.")
        else:
            # If multiple inputs, gather them from flows
            x = [flows[inp] for inp in flow['inputs'] if inp in flows]
            if len(x) != len(flow['inputs']):
                missing_inputs = set(flow['inputs']) - set(flows.keys())
                raise ValueError(f"Missing required inputs: {missing_inputs}")

        print("Input to operations: ", x)  # Debug print to check inputs before operations
        # Apply each operation
        for op in flow['operations']:
            x = apply_operation(x, op, inputs, flows, st_embed = st_embed)  # Pass flows here

        # Store the resulting tensor in flows dictionary
        flows[flow['name']] = x
        print("Output of flow stored: ", flow['name'])  # Confirm output is stored

    # Build the final model using the last flow as output
    model = keras.Model(inputs=list(inputs.values()), 
                        outputs=flows[config['process_flows'][-1]['name']])
    print("Model built successfully.")  # Confirm model build completion
    return model

def mode_switch(mode):
    switcher = {
        'VMAX': 0,
        'PMIN': 1,
        'RMW': 2
    }
    # Return the corresponding value if mode is found, otherwise return None or a default value
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

def load_data_excluding_year(data_directory, mode, validation_year = validation_year, test_year = test_year):
    """
    Loads data from specified directory excluding specified years and organizes it into training and validation sets.

    Args:
        data_directory (str): The root directory where data files are stored.
        mode (str): Mode of operation which defines how labels should be manipulated or filtered.
        validation_year (list): List of years to be used for validation.
        test_year (list): List of years to be excluded from the loading process.
        var_num (int): Variable number identifier used in file naming.
        windows (str): Window size identifier used in file naming.

    Returns:
        tuple: Tuple containing six elements:
               - all_features (np.ndarray): Array of all features excluding validation and test years.
               - all_labels (np.ndarray): Array of all labels corresponding to all_features.
               - all_space_times (np.ndarray): Array of all spatial and temporal data corresponding to all_features.
               - val_features (np.ndarray): Array of validation features from the validation years.
               - val_labels (np.ndarray): Array of validation labels corresponding to val_features.
               - val_space_times (np.ndarray): Array of validation spatial and temporal data corresponding to val_features.
    """
    years = get_year_directories(data_directory)
    months = range(1, 13)  # Months labeled 1 to 12
    b = mode_switch(mode)  # Make sure this function is defined elsewhere
    all_features, all_labels, all_space_times = [], [], []
    val_features, val_labels, val_space_times = [], [], []

    # Loop over each year
    for year in years:
        #print(year)
        if year in validation_year:
            print("validation year", year)
        if year in test_year:
            print("test", year)
            continue  # Skip the excluded year
        
        # Loop over each month
        for month in months:
            feature_filename = f'features{var_num}_{windows}{month:02d}fixed.npy'
            label_filename = f'labels{var_num}_{windows}{month:02d}.npy'
            space_time_filename = f'space_time_info{var_num}_{windows}{month:02d}.npy'
            
            # Construct full paths
            feature_path = os.path.join(data_directory, str(year), feature_filename)
            label_path = os.path.join(data_directory, str(year), label_filename)
            space_time_path = os.path.join(data_directory, str(year), space_time_filename)

            # Check if files exist before loading
            if os.path.exists(feature_path) and os.path.exists(label_path) and os.path.exists(space_time_path):
                features = np.load(feature_path)
                labels = np.load(label_path)[:, b]
                space_time = np.load(space_time_path)
                # Append to lists
                if year in validation_year:
                    #
                    val_features.append(features)
                    val_labels.append(labels)
                    val_space_times.append(space_time)
                else:
                    all_features.append(features)
                    all_labels.append(labels)
                    all_space_times.append(space_time)
            else:
                print(f"Warning: Files not found for year {year} and month {month}")
                #print(label_path, feature_path)

    # Concatenate all loaded data into single arrays
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_space_times = np.concatenate(all_space_times, axis=0)
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_space_times = np.concatenate(val_space_times, axis=0)

    return all_features, all_labels, all_space_times, val_features, val_labels, val_space_times



# Defining metrics
def mae_for_output(index):
    # Mean absolute error, Interchangable with Tensorflow's MAE metrics but can work with multiple outputs.
    def mae(y_true, y_pred):
        return tf.keras.metrics.mean_absolute_error(y_true[:, index], y_pred[:, index])
    mae.__name__ = f'mae_{index+1}'  # Naming for clarity in logs
    return mae

def rmse_for_output(index):
    # Root mean squared error, same as MAE.
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:, index], y_pred[:, index]))
    rmse.__name__ = f'rmse_{index+1}'  # Naming for clarity in logs
    return rmse

#==============================================================================================
# Resize data into desired height and width. Input should be in form [height, width, channel].
#==============================================================================================

def resize_preprocess(image, HEIGHT, WIDTH, method):
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image

#==============================================================================================
# Normalize bands value.
# NOTE: normalize only features, not labels (althought requires labels as input)
# NOTE: normalize by sample, not batch normalization.
#==============================================================================================
def normalize_channels(X,y):
    """
    Normalizes each channel in each sample individually.

    Parameters:
    - X: Input array of shape (nsample, height, width, number_channels).
    - y: Corresponding labels.

    Returns:
    - Normalized X and y arrays.
    """
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            X[i,:,:,var] = X[i,:,:,var] / abs(maxvalue)
    print("Finish normalization...")
    return X,y

#==============================================================================================
# Defining custom learning rate
#==============================================================================================
def lr_scheduler(epoch, lr):
    """
    Adjusts the learning rate based on the current training epoch.

    This function defines a custom learning rate schedule that adjusts the 
    learning rate during training. The learning rate decreases according 
    to a predefined schedule to help the model converge more effectively.

    Parameters:
    - epoch (int): The current epoch number during training.
    - lr (float): The current learning rate.

    Returns:
    - float: The adjusted learning rate for the current epoch.
    """
    
    # Base learning rate
    lr0 = 0.001
    
    # Calculate the new learning rate based on the epoch number
    lr = -0.0497 + (1.0 - (-0.0497)) / (1 + (epoch / 107.0) ** 1.35)
    
    # If the epoch number is greater than 940, set a minimum learning rate
    if epoch > 940:
        lr = 0.0001
    
    # Multiply the new learning rate by the base learning rate
    return lr * lr0

def normalize_Z(Z):
    Z[:,2] = (Z[:,2]+90) / 180
    Z[:,3] = (Z[:,3]+180) / 360
    return Z
#==============================================================================================
# Model
#==============================================================================================
def main(X, Y, X_val, Y_val, loss='huber', activ='relu', NAME='best_model', st_embed=False, Z=None, Z_val=None, batch_size = batch_size, epoch = num_epochs):
    config = load_json_config(config_path)
    model = build_model_from_json(config, st_embed=st_embed)
    
    # Include `z_input` in the inputs
    model.compile(
        optimizer='adam',
        loss='huber',
        metrics=[mae_for_output(i) for i in range(1)] + [rmse_for_output(i) for i in range(1)]
    )
    
    # Redefine the model with updated inputs and outputs
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(NAME, save_best_only=True),
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    ]

    if st_embed:
        history = model.fit([X, Z], Y, batch_size=batch_size, epochs=epoch, validation_data=([X_val, Z_val], Y_val), verbose=2, callbacks=callbacks)
    else:
        history = model.fit(X, Y, batch_size=batch_size, epochs=epoch, validation_data=(X_val, Y_val), verbose=2, callbacks=callbacks)

    return history

#==============================================================================================
# MAIN CALL:
#==============================================================================================
X, Y, Z, X_val, Y_val, Z_val = load_data_excluding_year(data_dir, mode) 
X=np.transpose(X, (0, 2, 3, 1))
    
# Normalize the data before encoding
X,Y = normalize_channels(X, Y)
Z = normalize_Z(Z)
X_val,Y_val = normalize_channels(X_val, Y_val)
X_val = np.transpose(X_val, (0, 2, 3, 1))
Z_val = normalize_Z(Z_val)
X=resize_preprocess(X, image_size,image_size, 'lanczos5')
X_val=resize_preprocess(X_val, image_size,image_size, 'lanczos5')
number_channels=X.shape[3]

print('Input shape of the X features data: ',X.shape)
print('Input shape of the y label data: ',Y.shape)
print('Number of input channel extracted from X is: ',number_channels)

history = main(X=X, Y=Y, X_val=X_val, Y_val=Y_val, NAME = os.path.join(model_dir, model_name), st_embed=st_embed, Z=Z, Z_val=Z_val)
