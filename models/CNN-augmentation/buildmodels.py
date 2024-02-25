import tensorflow as tf
import numpy as np
import time
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
def main(dense_layers=[1],layer_sizes=[32],conv_layers=[3],X=[],y=[], target='VMAX', loss='', firstlayer=3,firstfilter=64):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.2)])
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME =root+'model/'+ loss+'-'+str(firstlayer)+"{}-conv-{}-layer-{}-dense.model".format(conv_layer,layer_size,dense_layer)
                print('--> Running configuration: ',NAME)
                inputs = keras.Input(shape=X.shape[1:])
                x = data_augmentation(inputs)
                #x = layers.Conv2D(filters=256,kernel_size=7, padding='same' ,activation="relu",name="my_conv2d_1")(x)
                x = layers.Conv2D(filters=firstfilter,kernel_size=7, padding='same',activation=activ,name="my_conv2d_11")(x)
                #x = layers.Conv2D(filters=layer_size*2,kernel_size=3, activation="relu",name="my_conv2d_12")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_1")(x)
                x=tf.keras.layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=layer_size*2,kernel_size=conv_layer, activation=activ,name="my_conv2d_2")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_2")(x)
                x=tf.keras.layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=layer_size*2,kernel_size=conv_layer,activation=activ,name="my_conv2d_3")(x)
                x = layers.Conv2D(filters=layer_size*2,kernel_size=conv_layer, activation=activ,name="my_conv2d_31")(x)
                #x = layers.Conv2D(filters=layer_size*4,kernel_size=5,padding='same',activation="relu",name="my_conv2d_32")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_3")(x)

                x=tf.keras.layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=layer_size*2,kernel_size=3, padding='same', activation=activ,name="my_conv2d_4")(x)
                
                x = layers.Conv2D(filters=layer_size*2,kernel_size=3, activation=activ,name="my_conv2d_5")(x)
                #x = layers.MaxPooling2D(pool_size=2,name="my_pooling_4")(x)
                x=tf.keras.layers.BatchNormalization()(x)
                x = layers.Flatten(name="my_flatten")(x)
                x = layers.Dropout(0.3)(x)
                for _ in range(dense_layer):
                    x = layers.Dense(layer_size,activation=activ)(x)                
                
                outputs = layers.Dense(1,activation=activ ,name="my_dense")(x)
                model = keras.Model(inputs=inputs,outputs=outputs,name="my_functional_model")
                model.summary()
            
                callbacks=[keras.callbacks.ModelCheckpoint(NAME,save_best_only=True)]
                model.compile(loss=loss,optimizer="adam",metrics=['MAE',tf.keras.metrics.RootMeanSquaredError()])
                history = model.fit(X, y, batch_size=5000, epochs=1000, validation_split=0.1, callbacks=callbacks)
    return history
def resize_preprocess(image, HEIGHT, WIDTH, method):
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image
def normalize_channels(X,y):
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            #print('Normalization factor for sample and channel',i,var,', is: ',abs(maxvalue))
            X[i,:,:,var] = X[i,:,:,var]/abs(maxvalue)
            maxnew = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            #print('-->After normalization of sample and channel',i,var,', is: ',abs(maxnew))
            #input('Enter to continue...')
    print("Finish normalization...")
    return X,y
root='/N/slate/kmluong/Training_data/Split/'
X = np.load(root+'data/trainx.npy')
X=np.transpose(X, (0, 2, 3, 1))
y = np.load(root+'data/trainy.npy')[:,0]
number_channels=X.shape[3]



activ='linear'
x_train,y_train = normalize_channels(X,y)
'''
for i in range(y.shape[1]):
        print(np.max(y[:,i]))
        y_train[:,i]=y[:,i]/np.max(y[:,i])
        '''
x_train=resize_preprocess(x_train, 64,64, 'lanczos5')
DENSE_LAYER = [2]
LAYER_SIZES = [32]
CONV_LAYERS = [3]
losses=["mean_absolute_percentage_error", "huber_loss", "log_cosh"]
print('Input shape of the X features data: ',X.shape)
print('Input shape of the y label data: ',y.shape)
print('Number of input channel extracted from X is: ',number_channels)
firstfilter=[32]
for f in firstfilter:
    history = main(dense_layers=DENSE_LAYER,layer_sizes=LAYER_SIZES,conv_layers=CONV_LAYERS,
               X=x_train,y=y_train, target='VMAX', loss='huber', firstlayer=7,
               firstfilter=f)
