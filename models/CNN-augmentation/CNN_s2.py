import tensorflow as tf
import numpy as np
import pickle
import time
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import libtcg_utils as tcg_utils
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