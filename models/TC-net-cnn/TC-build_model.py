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
    parser.add_argument('-m', '--mode', type=str, default = 'VMAX', help='Mode of operation (e.g., VMAX, PMIN, RMW)')
    parser.add_argument('-mname', '--model_name', type=str, default = 'CNNmodel', help='Core name of the model')
    parser.add_argument('-r', '--root', type=str, default = '/N/project/Typhoon-deep-learning/output/', help='Working directory path')
    parser.add_argument('-ws', '--windowsize', type=int, nargs=2, default = [19,19], help='Window size as two integers (e.g., 19 19)')
    parser.add_argument('-vno', '--var_num', type=int, default = 13, help='Number of variables')
    parser.add_argument('-st', '--st_embed', type=bool, default = False, help='Including space-time embedded')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('-bsize', '--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('-eno', '--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('-imsize', '--image_size', type=int, default=64, help='Size to resize the image to')
    parser.add_argument('-cfg', '--config', type=str, default = 'model_core/test.json')
    parser.add_argument('-ss', '--data_source', type=str, default = 'MERRA2')
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Temporary working folder')
    return parser.parse_args()
args = parse_args()
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs        	
image_size = args.image_size  				
mode = args.mode
root = args.root
var_num = args.var_num
st_embed = args.st_embed
config_path = args.config
data_source=args.data_source
work_folder=args.work_folder


model_dir = os.path.join(root, 'model')
model_name = args.model_name
model_name = f'{model_name}_{data_source}_{mode}{"_st" if st_embed else ""}'
temp_dir = os.path.join(work_folder, 'temp')
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
def load_data(temp_dir):
    global train_x, train_y, train_z, val_x, val_y, val_z

    # Check for training data files and load them if they exist
    if 'train_x.npy' in os.listdir(temp_dir):
        train_x = np.load(os.path.join(temp_dir, 'train_x.npy'))
    if 'train_y.npy' in os.listdir(temp_dir):
        train_y = np.load(os.path.join(temp_dir, 'train_y.npy'))
    if 'train_z.npy' in os.listdir(temp_dir):
        train_z = np.load(os.path.join(temp_dir, 'train_z.npy'))

    # Check for validation data files and load them if they exist
    if 'val_x.npy' in os.listdir(temp_dir):
        val_x = np.load(os.path.join(temp_dir, 'val_x.npy'))
    if 'val_y.npy' in os.listdir(temp_dir):
        val_y = np.load(os.path.join(temp_dir, 'val_y.npy'))
    if 'val_z.npy' in os.listdir(temp_dir):
        val_z = np.load(os.path.join(temp_dir, 'val_z.npy'))


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
def main(X, Y, loss='huber', NAME='best_model', st_embed=False, batch_size=32, epoch=100):
    # Load model configuration and build the model
    config = load_json_config(config_path)
    model = build_model_from_json(config, st_embed=st_embed)
    
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=[mae_for_output(i) for i in range(1)] + [rmse_for_output(i) for i in range(1)]
    )
    
    model.summary()
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(NAME, save_best_only=True),
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    ]
    
    # Set up training inputs and labels
    train_inputs = [X]
    train_Y = Y  # Default training labels
    if st_embed and 'train_z' in globals() and train_z is not None:
        train_inputs.append(train_z)
    
    # Determine validation data and labels
    if 'val_x' in globals() and val_x is not None and 'val_y' in globals() and val_y is not None:
        if st_embed and 'val_z' in globals() and val_z is not None:
            val_inputs = [val_x, val_z]
        else:
            val_inputs = [val_x]
        val_Y = val_y
        val_data = (val_inputs, val_Y)
    else:
        val_data = None

    # Fit the model using the (possibly split) training labels
    history = model.fit(train_inputs, train_Y, batch_size=batch_size, epochs=epoch,
                        validation_data=val_data, verbose=2, callbacks=callbacks, shuffle=True)

    return history


#==============================================================================================
# MAIN CALL:
#==============================================================================================

b=mode_switch(mode)
load_data(temp_dir)

# Transpose train_x as it is always present
train_x = np.transpose(train_x, (0, 2, 3, 1))

# Normalize train data, which is always present
train_x, train_y = normalize_channels(train_x, train_y[:,b])

# Check if train_z exists and should be normalized
if 'train_z' in globals() and train_z is not None and st_embed:
    train_z = normalize_Z(train_z)

# Check if validation data exists before normalization and transposition
if 'val_x' in globals() and 'val_y' in globals():
    val_x, val_y = normalize_channels(val_x, val_y[:,b])
    val_x = np.transpose(val_x, (0, 2, 3, 1))

# Normalize val_z if it exists and st_embed is true
if 'val_z' in globals() and val_z is not None and st_embed:
    Z_val = normalize_Z(val_z)

# Resize train_x since it is always present
train_x = resize_preprocess(train_x, image_size, image_size, 'lanczos5')

# Resize and preprocess val_x if it exists
if 'val_x' in globals():
    val_x = resize_preprocess(val_x, image_size, image_size, 'lanczos5')

# Assuming train_x is defined and checking the number of channels
number_channels = train_x.shape[3]

print('Input shape of the X features data: ',train_x.shape)
print('Input shape of the y label data: ',train_y.shape)
print('Number of input channel extracted from X is: ',number_channels)

history = main(X=train_x, Y=train_y, NAME = os.path.join(model_dir, model_name), st_embed=st_embed, val_pc=val_pc)
