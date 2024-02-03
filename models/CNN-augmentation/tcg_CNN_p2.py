#
# NOTE: This machine learning program is for predicting TC formation, using
#       input dataset in the NETCDF format. The program treats different
#       2D input fields as different channels of an image. This specific
#       program requires a set of 12 2D-variables (12-channel image) and
#       consists of three stages
#       - Stage 1: reading NETCDF input and generating (X,y) data with a
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,y) pair and build a CNN model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_CNN.model.
#       - Stage 3: import the trained model from Stage 2, and make a list
#                  of prediction from normalized test data.
#
# INPUT: This Stage 2 script requires two specific input datasets that are
#        generated from Step 1, including
#        1. tcg_X.pickle: data contains all images of yes/no TCG events, each
#           of these images must have 12 channels
#        2. tcg_y.pickle: data contains all labels of each image (i.e., yes
#           or no) of TCG corresponding to each data in X.
#
#        Remarks: Note that each channel must be normalized separealy. Also
#        the script requires a large memory allocation. So users need to have
#        GPU version to run this.
#
# OUTPUT: A CNN model built from Keras saved under tcg_CNN.model
#
# HIST: - 27, Oct 22: Created by CK
#       - 01, Nov 22: Modified to include more channels
#       - 17, Nov 23: cusomize it for jupiter notebook
#       - 21, Feb 23: use functional model instead of sequential model  
#       - 05, Jun 23: Re-check for consistency with Stage 1 script and added more hyperparamter loops
#       - 20, Jun 23: Updated for augmentation/dropout layers
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
import tensorflow as tf
import numpy as np
import pickle
import time
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import libtcg_utils as tcg_utils
#
# build a range of CNN models with different number of dense layer, layer sizes, and
# convolution layers to optimize the performance
#
def main(dense_layers=[1],layer_sizes=[32],conv_layers=[3],X=[],y=[],lead_time="00"):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)])
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-layer-{}-dense.model_{}h".format(conv_layer,layer_size,dense_layer,lead_time)
                print('--> Running configuration: ',NAME)

                inputs = keras.Input(shape=X.shape[1:])          
                x = data_augmentation(inputs)            
                x = layers.Conv2D(filters=layer_size,kernel_size=conv_layer,activation="relu",name="my_conv2d_1")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_1")(x)
                x = layers.Conv2D(filters=layer_size*2,kernel_size=conv_layer,activation="relu",name="my_conv2d_2")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_2")(x)
                if conv_layer == 3:
                    x = layers.Conv2D(filters=layer_size*4,kernel_size=conv_layer,activation="relu",name="my_conv2d_3")(x)
                    x = layers.MaxPooling2D(pool_size=2,name="my_pooling_3")(x)
    
                if X.shape[1] > 128:
                    x = layers.Conv2D(filters=256,kernel_size=conv_layer,activation="relu",name="my_conv2d_4")(x)
                    x = layers.MaxPooling2D(pool_size=2,name="my_pooling_4")(x)
                    x = layers.Conv2D(filters=256,kernel_size=conv_layer,activation="relu",name="my_conv2d_5")(x)
                x = layers.Flatten(name="my_flatten")(x)
                x = layers.Dropout(0.2)(x)
            
                for _ in range(dense_layer):
                    x = layers.Dense(layer_size,activation="relu")(x)                
                
                outputs = layers.Dense(1,activation="sigmoid",name="my_dense")(x)
                model = keras.Model(inputs=inputs,outputs=outputs,name="my_functional_model")
                model.summary()
                keras.utils.plot_model(model)
            
                callbacks=[keras.callbacks.ModelCheckpoint(NAME,save_best_only=True)]
                model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
                history = model.fit(X, y, batch_size=128, epochs=30, validation_split=0.1, callbacks=callbacks)
    return history
#
# Visualize the output of the training model (work for jupyter notebook only)
#
def view_history(history):
    import matplotlib.pyplot as plt
    #print(history.__dict__)
    #print(history.history)
    val_accuracy = history.history['val_accuracy']
    accuracy = history.history['accuracy']
    epochs = history.epoch 
    plt.plot(epochs,val_accuracy,'r',label="val accuracy")
    plt.plot(epochs,accuracy,'b',label="train accuracy")
    plt.legend()

    plt.figure()
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    plt.plot(epochs,val_loss,'r',label="val loss")
    plt.plot(epochs,loss,'b',label="train loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    n = len(sys.argv)
    print("Total arguments input are:", n)
    print("Name of Python script:", sys.argv[0])
    if n < 2:
       print("Need a forecast lead time to process...Stop")
       print("+ Example: tcg_CNN_p2.py 00")
       exit()
    leadtime = str(sys.argv[1])
    print("Forecast lead time to run is: ",leadtime)
    #sys.exit()
    #
    # read in data output from Part 1 and normalize it
    #
    pickle_in = open("tcg_CNNaugment_X.pickle","rb")
    X = pickle.load(pickle_in)
    pickle_in = open("tcg_CNNaugment_y.pickle","rb")
    y = pickle.load(pickle_in)
    y = np.array(y)
    number_channels=X.shape[3]
    print('Input shape of the X features data: ',X.shape)
    print('Input shape of the y label data: ',y.shape)
    print('Number of input channel extracted from X is: ',number_channels)
    
    x_train,y_train = tcg_utils.normalize_channels(X,y) 
    #
    # define the model architechture
    #   
    DENSE_LAYER = [0, 1, 2]
    LAYER_SIZES = [32]
    CONV_LAYERS = [3, 5]
    history = main(dense_layers=DENSE_LAYER,layer_sizes=LAYER_SIZES,conv_layers=CONV_LAYERS,
                   X=x_train,y=y_train,lead_time=leadtime)
  
    check_visualization = "no"
    if check_visualization== "yes": 
        view_history(history)
