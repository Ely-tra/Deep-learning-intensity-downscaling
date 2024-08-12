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
#
# Edit the parameters properly before running this script
#
workdir = '/N/slate/kmluong/TC-net-cnn_workdir/Domain_data/'
var_num = 13
windowsize = [25,25]
mode = 'VMAX'

#####################################################################################
# DO NOT EDIT BELOW UNLESS YOU WANT TO MODIFY THE SCRIPT
#####################################################################################
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

#==============================================================================================
# Model
#==============================================================================================
def main(X, y, loss='huber', activ='relu', NAME='best_model'):
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ])
    print('--> Running configuration: ', NAME)

    inputs = keras.Input(shape=X.shape[1:])
    x = data_augmentation(inputs)
    x = layers.Conv2D(filters=128, kernel_size=15, padding='same', activation=activ, name="my_conv2d_11")(x)
    x = layers.MaxPooling2D(pool_size=2, name="my_pooling_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=15, padding='same', activation=activ, name="my_conv2d_2")(x)
    x = layers.MaxPooling2D(pool_size=2, name="my_pooling_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=256, kernel_size=9, padding='same', activation=activ, name="my_conv2d_3")(x)
    x = layers.MaxPooling2D(pool_size=2, name="my_pooling_3")(x)
    x = layers.Conv2D(filters=512, kernel_size=5, padding='same', activation=activ, name="my_conv2d_4")(x)
    x = layers.Conv2D(filters=512, kernel_size=5, padding='valid', activation=activ, name="my_conv2d_5")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten(name="my_flatten")(x)
    x = layers.Dropout(0.4)(x)

    for _ in range(2):
        x = layers.Dense(512 - _ * 200, activation=activ)(x)

    outputs = layers.Dense(1, activation=activ, name="my_dense")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(NAME, save_best_only=True),
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    ]

    model.compile(
        loss=loss,
        optimizer="adam",
        metrics=[mae_for_output(i) for i in range(1)] + [rmse_for_output(i) for i in range(1)]
    )

    history = model.fit(X, y, batch_size=128, epochs=1000, validation_split=2/9, verbose=2, callbacks=callbacks)
    return history

#==============================================================================================
# MAIN CALL:
#==============================================================================================
windows = str(windowsize[0])+'x'+str(windowsize[1])
root = workdir+'/exp_'+str(var_num)+'features_'+windows+'/'
best_model_name = root + '/model_VMAX'+str(var_num)+'_'+windows
X = np.load(root+'/train'+str(var_num)+'x_'+windows+'.npy')
X=np.transpose(X, (0, 2, 3, 1))

if mode=='VMAX':
  b=0
if mode=='PMIN':
  b=1
if mode=='RMW':
  b=2
y = np.load(root+'/train'+str(var_num)+'y_'+windows+'.npy')[:,b]
x_train,y_train = normalize_channels(X,y)
x_train=resize_preprocess(x_train, 64,64, 'lanczos5')
number_channels=X.shape[3]

print('Input shape of the X features data: ',X.shape)
print('Input shape of the y label data: ',y.shape)
print('Number of input channel extracted from X is: ',number_channels)

history = main(X=x_train, y=y_train, NAME=best_model_name)
