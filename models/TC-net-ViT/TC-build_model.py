# DESCRIPTION: This script utilizes TensorFlow to implement a (ViT) designed for correcting 
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
# MODEL CONFIGURATION:
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
# HIST: - Sep, 09, 2024: created by Tri Nguyen from the previous TCG's projection, using the 
#                        available VIT model from Keras Image classification with Vision 
#                        Transformer.
#       - Oct, 18, 2024: cleaned up and noted by CK for better flows
#
# AUTH: Tri Huu Minh Nguyen
#      
#==============================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
import sys
import libtcg_utils as tcg_utils
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
#
# Edit data path and model parameters below
#
root = '/N/project/Typhoon-deep-learning/output/'
var_num = 13
windowsize = [19,19]
mode = 'VMAX'
windows = f'{windowsize[0]}x{windowsize[1]}'
work_dir = root +'/exp_'+str(var_num)+'features_'+windows+'/'
data_dir = work_dir + 'data/'
model_dir = work_dir + 'model/'
model_name = 'ViT_model1'
xfold = 7 
st_embed = True
model_name = model_name + '_fold' + str(xfold) + '_' + mode  + ('_st' if st_embed else '')
#
# Configurable VIT parameters
#
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100        	# For real training, use num_epochs=100. 10 is a test value
image_size = 72  		# We'll resize input images to this size
patch_size = 12  		# Size of the patches to be extract from the input images
projection_dim = 64             # embedding dim
num_heads = 4			# number of heads
num_classes = 1			# number of class
transformer_units = [		# Size of the transformer layers
    projection_dim*2,
    projection_dim]  		
transformer_layers = 8
mlp_head_units = [2048,1024]
num_patches = (image_size // patch_size) ** 2

#==============================================================================================
# All functions are below
#==============================================================================================
def mode_switch(mode):
    switcher = {
        'VMAX': 0,
        'PMIN': 1,
        'RMW': 2
    }
    # Return the corresponding value if mode is found, otherwise return None or a default value
    return switcher.get(mode, None)

def load_data_excluding_fold(data_directory, xfold = xfold, mode = mode):
    months = range(1, 13)  # Months labeled 1 to 12
    k = 10  # Total number of folds
    b = mode_switch(mode)
    all_features = []
    all_labels = []
    all_space_times = []
    # Loop over each fold
    for fold in range(1, k+1):
        if fold == xfold:
            continue  # Skip the excluded fold

        # Loop over each month
        for month in months:
            feature_filename = f'test_features_fold{fold}_{windows}{month:02d}fixed.npy'
            label_filename = f'test_labels_fold{fold}_{windows}{month:02d}fixed.npy'
            space_time_filename = f'test_spacetime_fold{fold}_{windows}{month:02d}fixed.npy'
            #print(f'Loading {feature_filename}')
            # Construct full paths
            feature_path = os.path.join(data_directory, feature_filename)
            label_path = os.path.join(data_directory, label_filename)
            space_time_path = os.path.join(data_directory, space_time_filename)
            # Check if files exist before loading
            if os.path.exists(feature_path) and os.path.exists(label_path):
                # Load the data
                features = np.load(feature_path)
                labels = np.load(label_path)[:,b]
                space_time = np.load(space_time_path)
                # Append to lists
                all_features.append(features)
                all_labels.append(labels)
                all_space_times.append(space_time)
            else:
                print(f"Warning: Files not found for fold {fold} and month {month}")
                print(label_path,feature_path)

    # Concatenate all loaded data into single arrays
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_space_times = np.concatenate(all_space_times, axis=0)
    return all_features, all_labels, all_space_times

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

        patches = tf.image.extract_patches(images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')

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

def create_vit_classifier(input_shape = (30,30,12), st_embed = st_embed):
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
    if st_embed:
        representation = keras.layers.Concatenate()([representation, additional_input])
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=[inputs, additional_input], outputs=logits)
    return model

def main(X=[],y=[],Z=[], size=[18,18], st_embed = st_embed):
    histories = []

    model = create_vit_classifier(input_shape= (X.shape[1], X.shape[2], X.shape[3]), st_embed = st_embed)
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
    hist = model.fit([X,Z], Y, epochs = 1000, batch_size = 128, callbacks=callbacks, validation_split=0.2, verbose=2)

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

def normalize_Z(Z):
    Z[:,2] = (Z[:,2]+90) / 180
    Z[:,3] = (Z[:,3]+180) / 360
    return Z

#==============================================================================================
# Main call
#==============================================================================================

X, Y, Z = load_data_excluding_fold(data_dir, xfold) 
X=np.transpose(X, (0, 2, 3, 1))
    
# Normalize the data before encoding
X,Y = normalize_channels(X, Y)
Z = normalize_Z(Z)
number_channels=X.shape[3]
print('Input shape of the X features data: ',X.shape)
print('Input shape of the y label data: ',Y.shape)
print('Number of input channel extracted from X is: ',number_channels)

print ("number of input examples = " + str(X.shape[0]))
print ("X shape: " + str(X.shape))
print ("Y shape: " + str(Y.shape))
main(X=X,y=Y,Z = Z, size=windowsize)
