# DESCRIPTION: This script utilizes TensorFlow to implement a (ViT) designed for correcting 
#       TC intensity/structure from grided climate data, using the workflow inherited from the
#       previous TC formation project (https://github.com/kieucq/tcg_deep_learning). The model 
#       consists of several layers with varying functionalities including convolutional layers 
#       for TC feature extraction and dense layers for regression. Special attention is given 
#       to preprocessing steps like normalization and resizing, and the model is tuned to adapt 
#       its learning rate over epochs.
#
# MODEL CONFIGURATION:
#
# FUNCTIONS:
#       - mae_for_output: Custom mean absolute error function for specific outputs. Interchangable 
#         with TF MAE metric
#       - rmse_for_output: Custom root mean squared error function for specific outputs. 
#         Interchangable with TF RMSE metric.
#       - main: Orchestrates model construction, compilation, and training using specified 
#         parameters and datasets.
#       - normalize_channels: Normalizes data channels within the input array.
#
# USAGE: Users need to modify the main call with proper paths and parameters before running 
#
# HIST: - Sep, 09, 2024: created by Tri Nguyen from the previous TCG's projection, using the 
#                        available VIT model from Keras Image classification with Vision 
#                        Transformer.
#       - Oct, 18, 2024: cleaned up and noted by CK for better flows
#       - Oct, 30, 2024: workflow re-designed by Tri Nguyen
#       - Nov, 04, 2024: can now parse model's parameter directly. By Minh Khanh
#       - Nov, 27, 2024: cleaned up by CK and fixed the st_embed option
#
# AUTH: Tri Huu Minh Nguyen
#      
#==============================================================================================
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, Flatten
from tensorflow.keras.models import Model 
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import re

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Vision Transformer model for TC intensity correction.')
    parser.add_argument('--mode', type=str, default = 'VMAX',
                        help='Mode of operation (e.g., VMAX, PMIN, RMW)')
    parser.add_argument('--model_name', type=str, default = 'ViTmodel1',
                        help='Core name of the model')
    parser.add_argument('--root', type=str, default = '/N/project/Typhoon-deep-learning/output/',
                        help='Working directory path')
    parser.add_argument('--windowsize', type=int, nargs=2, default = [19,19],
                        help='Window size as two integers (e.g., 19 19)')
    parser.add_argument('--var_num', type=int, default = 13, help='Number of variables')
    parser.add_argument('--x_size', type=int, default = 72, help='X dimension size for the input')
    parser.add_argument('--y_size', type=int, default = 72, help='Y dimension size for the input')
    parser.add_argument('--st_embed', type=int, default = 0,
                        help='Including extra space-time information')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--patch_size', type=int, default=12,
                        help='Size of patches to be extracted from input images')
    parser.add_argument('--image_size', type=int, default=72,
                        help='Size to resize the image to')
    parser.add_argument('--projection_dim', type=int, default=64,
                        help='Dimension of the projection space')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of heads in multi-head attention')
    parser.add_argument('--transformer_layers', type=int, default=8,
                        help='Number of transformer layers')
    parser.add_argument('--mlp_head_units', nargs='+', type=int, default=[2048, 1024],
                        help='Number of units in MLP head layers')
    parser.add_argument('--validation_year', nargs='+', type=int, default=[2014],
                        help='Year(s) taken for validation')
    parser.add_argument('--test_year', nargs='+', type=int, default=[2017],
                        help='Year(s) taken for test')
    parser.add_argument('-ss', '--data_source', type=str, default = 'MERRA2')
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', 
                        help='Temporary working folder')
    parser.add_argument('-tid', '--temp_id', type=str)
    return parser.parse_args()

# Configurable VIT parameters
args = parse_args()
learning_rate = args.learning_rate
weight_decay = args.weight_decay
batch_size = args.batch_size
num_epochs = args.num_epochs        	
image_size = args.image_size  		
patch_size = args.patch_size		
projection_dim = args.projection_dim             
num_heads = args.num_heads			
validation_year = args.validation_year
test_year = args.test_year
transformer_units = [projection_dim*2,projection_dim]  		
transformer_layers = args.transformer_layers
mlp_head_units = args.mlp_head_units
num_patches = (image_size // patch_size) ** 2
mode = args.mode
root = args.root
windowsize = list(args.windowsize)
var_num = args.var_num
x_size = args.x_size
y_size = args.y_size
st_embed = args.st_embed
data_source = args.data_source
work_folder = args.work_folder
temp_id=args.temp_id
windows = f'{windowsize[0]}x{windowsize[1]}'
work_dir = root +'/exp_'+str(var_num)+'features_'+windows+'/'
data_dir = work_dir + 'data/'
model_dir = work_dir + 'model/'
temp_dir = os.path.join(work_folder, 'temp')
model_name = args.model_name
model_name = f'{model_name}_{data_source}_{mode}{"_st" if st_embed == 1 else ""}'
if mode == "ALL":
    num_classes = 3
else:
    num_classes = 1

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
        print(year)
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
                print(label_path, feature_path)

    # Concatenate all loaded data into single arrays
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_space_times = np.concatenate(all_space_times, axis=0)
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_space_times = np.concatenate(val_space_times, axis=0)

    return all_features, all_labels, all_space_times, val_features, val_labels, val_space_times


def load_data(temp_dir, temp_id=temp_id):
    global train_x, train_y, train_z, val_x, val_y, val_z
    print("temp_dir", temp_dir)
    print("temp_id", temp_id)
    # Check for training data files and load them if they exist
    if f'train_x_{temp_id}.npy' in os.listdir(temp_dir):
        train_x = np.load(os.path.join(temp_dir, f'train_x_{temp_id}.npy'))
    if f'train_y_{temp_id}.npy' in os.listdir(temp_dir):
        train_y = np.load(os.path.join(temp_dir, f'train_y_{temp_id}.npy'))
    if f'train_z_{temp_id}.npy' in os.listdir(temp_dir):
        train_z = np.load(os.path.join(temp_dir, f'train_z_{temp_id}.npy'))

    # Check for validation data files and load them if they exist
    if f'val_x_{temp_id}.npy' in os.listdir(temp_dir):
        val_x = np.load(os.path.join(temp_dir, f'val_x_{temp_id}.npy'))
    if f'val_y_{temp_id}.npy' in os.listdir(temp_dir):
        val_y = np.load(os.path.join(temp_dir, f'val_y_{temp_id}.npy'))
    if f'val_z_{temp_id}.npy' in os.listdir(temp_dir):
        val_z = np.load(os.path.join(temp_dir, f'val_z_{temp_id}.npy'))


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

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
                                           strides=[1, self.patch_size, self.patch_size, 1], 
                                           rates=[1, 1, 1, 1], padding='VALID')

        patches = tf.reshape(
            patches,
            (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels))

        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

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

def create_vit_classifier(st_embed, input_shape = (30,30,12)):
    inputs = keras.Input(shape=input_shape)
    # Augment data.
    additional_input = Input(shape=(4,), name='additional_input')
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    if st_embed == 1:
        representation = keras.layers.Concatenate()([representation, additional_input])
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    if st_embed == 1:
        model = keras.Model(inputs=[inputs, additional_input], outputs=logits)
    else:
        model = keras.Model(inputs=inputs, outputs=logits)
    return model

"""
def main(X=[],Y=[],Z=[], X_val=[], Y_val = [], Z_val = [], size=[18,18], st_embed = st_embed):
    histories = []
    print("St_embed", st_embed)
    model = create_vit_classifier(st_embed, input_shape= (X.shape[1], X.shape[2], X.shape[3]))
    print(model.summary())
    model.compile(optimizer='adam',
                      loss=tf.keras.losses.LogCosh(name="log_cosh"),
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name="RMSE")])
    model_checkpoint_path = os.path.join(model_dir, model_name)
    callbacks=[
	keras.callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=True), 
	keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
]
    early_stopping = EarlyStopping(monitor='val_RMSE', patience=5, restore_best_weights=True)    
    
    if st_embed == 1:
        hist = model.fit([X, Z], Y, epochs=num_epochs, batch_size=128, 
                         callbacks=callbacks, validation_data=([X_val, Z_val], Y_val), verbose=2)
    else:
        hist = model.fit(X, Y, epochs=num_epochs, batch_size=128, 
                         callbacks=callbacks, validation_data=(X_val, Y_val), verbose=2)
"""

def main(X, Y, st_embed = st_embed):
    histories = []
    print("St_embed", st_embed)
    model = create_vit_classifier(st_embed, input_shape= (X.shape[1], X.shape[2], X.shape[3]))
    print(model.summary())
    model.compile(optimizer='adam',
                      loss=tf.keras.losses.LogCosh(name="log_cosh"),
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name="RMSE")])
    model_checkpoint_path = os.path.join(model_dir, model_name)
    callbacks=[
        keras.callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=True), 
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
]
    early_stopping = EarlyStopping(monitor='val_RMSE', patience=5, restore_best_weights=True)    
    
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
    history = model.fit(train_inputs, train_Y, batch_size=batch_size, 
              epochs= num_epochs,validation_data=val_data, verbose=2, 
              callbacks=callbacks, shuffle=True)
    return history    
    
"""
def normalize_channels(X,y):
    
    Normalizes each channel in each sample individually.

    Parameters:
    - X: Input array of shape (nsample, height, width, number_channels).
    - y: Corresponding labels.

    Returns:
    - Normalized X and y arrays.
   
   
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            X[i,:,:,var] = X[i,:,:,var] / abs(maxvalue)
    print("Finish normalization...")
    return X,y
"""

def normalize_channels(X, y):
    """
    Normalizes each channel in each sample individually by converting to NumPy, 
    then rewriting back into X.

    X: tf.Tensor or NumPy array of shape (nsample, height, width, number_channels)
    y: Corresponding labels.

    Returns:
    - X, y with normalization applied (X will be tf.Tensor if originally tf.Tensor).
    """
    # If X is a Tensor, convert to NumPy first (entire array).
    # For large data, consider slice-by-slice .numpy() instead.
    is_tf_tensor = isinstance(X, tf.Tensor)
    if is_tf_tensor:
        X_np = X.numpy()
    else:
        X_np = X  # Already NumPy

    nsample = X_np.shape[0]
    number_channels = X_np.shape[3]

    for i in range(nsample):
        for var in range(number_channels):
            slice_ = X_np[i, :, :, var]
            maxvalue = slice_.flat[np.abs(slice_).argmax()]
            # Avoid divide-by-zero if maxvalue == 0
            if maxvalue == 0:
                continue
            X_np[i, :, :, var] = slice_ / abs(maxvalue)

    print("Finish normalization (NumPy method)...")

    # Convert back to tf.Tensor if original was tf.Tensor
    if is_tf_tensor:
        X = tf.convert_to_tensor(X_np, dtype=X.dtype)

    return X, y

def normalize_Z(Z):
    Z[:,2] = (Z[:,2]+90) / 180
    Z[:,3] = (Z[:,3]+180) / 360
    return Z

def resize_preprocess(image, HEIGHT, WIDTH, method):
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image

# Main call
if __name__ == "__main__":
    b = mode_switch(mode)
    load_data(temp_dir)
    train_x = np.transpose(train_x, (0, 2, 3, 1))
    train_x = resize_preprocess(train_x, image_size, image_size, 'lanczos5')

    # Normalize train data, which is always present
    if mode == "ALL":
        train_x, train_y = normalize_channels(train_x, train_y[:,0:3])
    else:
       train_x, train_y = normalize_channels(train_x, train_y[:,b])
    if 'train_z' in globals() and train_z is not None and st_embed:
       train_z = normalize_Z(train_z)

    # Check if validation data exists before normalization and transposition
    if 'val_x' in globals() and 'val_y' in globals():
       if mode == "ALL":
           val_x, val_y = normalize_channels(val_x, val_y[:,0:3])
       else:
           val_x, val_y = normalize_channels(val_x, val_y[:,b])
       val_x = np.transpose(val_x, (0, 2, 3, 1))
       val_x = resize_preprocess(val_x, image_size, image_size, 'lanczos5')

    # Normalize val_z if it exists and st_embed is true
    if 'val_z' in globals() and val_z is not None and st_embed:
       val_z = normalize_Z(val_z)
    
    number_channels = train_x.shape[3]

    print('Input shape of the X features data: ',train_x.shape)
    print('Input shape of the y label data: ',train_y.shape)
    print('Number of input channel extracted from X is: ',number_channels)
    main(train_x, train_y , st_embed)    
    
