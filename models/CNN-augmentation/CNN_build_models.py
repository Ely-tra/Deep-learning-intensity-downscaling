print('Initializing', flush=True)
import tensorflow as tf
import numpy as np
import time
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import libtcg_utils as tcg_utils
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

##################################################################
#Split data
##################################################################

def split_data(datax, datay, percentage):
    datax, datay = shuffle(datax, datay, random_state=0)
    testx , testy  = datax[:int(len(datax)/(100/percentage))], datay[:int(len(datax)/(100/percentage))]
    trainx, trainy = datax[int(len(datax)/(100/percentage)):], datay[int(len(datax)/(100/percentage)):]
    return trainx, trainy, testx, testy

##################################################################
#Resize
##################################################################

def resize_preprocess(image, HEIGHT, WIDTH, method):
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image


##################################################################
#Model
##################################################################

def main(dense_layers=[1],layer_sizes=[32],conv_layers=[3],X=[],y=[], target='VMAX'):
    if target=='VMAX':
        mtrc='huber'
    if target=='PMIN':
        mtrc='MSE'
    if target=='RMW':
        mtrc='mean_absolute_percentage_error'
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)])
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = root+'Split/model/'+target+"-{}-conv-{}-layer-{}-dense.model".format(conv_layer,layer_size,dense_layer)
                print('--> Running configuration: ',NAME)
                inputs = keras.Input(shape=X.shape[1:])
                x = data_augmentation(inputs)
                x = layers.Conv2D(filters=layer_size,kernel_size=conv_layer,activation="relu",name="my_conv2d_1")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_1")(x)
                x = layers.Conv2D(filters=layer_size*2,kernel_size=conv_layer, activation="relu",name="my_conv2d_2")(x)
                x = layers.MaxPooling2D(pool_size=2,name="my_pooling_2")(x)
                if conv_layer == 3:
                    x = layers.Conv2D(filters=layer_size*4,kernel_size=conv_layer,activation="relu",name="my_conv2d_3")(x)
                    x = layers.MaxPooling2D(pool_size=2,name="my_pooling_3")(x)
    
                if X.shape[1] > 128:
                    x = layers.Conv2D(filters=256,kernel_size=conv_layer,padding='same',activation="relu",name="my_conv2d_4")(x)
                    x = layers.MaxPooling2D(pool_size=2,name="my_pooling_4")(x)
                    x = layers.Conv2D(filters=256,kernel_size=conv_layer,padding='same',activation="relu",name="my_conv2d_5")(x)
                x = layers.Flatten(name="my_flatten")(x)
                x = layers.Dropout(0.2)(x)
            
                for _ in range(dense_layer):
                    x = layers.Dense(layer_size,activation="relu")(x)                
                
                outputs = layers.Dense(1,activation='linear',name="my_dense")(x)
                model = keras.Model(inputs=inputs,outputs=outputs,name="my_functional_model")
                model.summary()
            
                callbacks=[keras.callbacks.ModelCheckpoint(NAME,save_best_only=True)]
                model.compile(loss=mtrc,optimizer="adam",metrics=keras.metrics.R2Score())
                history = model.fit(X, y, batch_size=512, epochs=30, validation_split=0.1, callbacks=callbacks)
    return history

##################################################################
#Run function
##################################################################


def build_models(dense_layers=[1],layer_sizes=[32],conv_layers=[3], root='', basin='AL' ,targets=[0], imsize=[64,64]):
    count=0
    print('Splitting data', flush=True)
    datax=np.load(root+'CNNfeatures'+basin+'fixed.npy')
    datay=np.load(root+'CNNlabels'+basin+'.npy')

    trainx, trainy, testx, testy = split_data(datax, datay, 5)
    np.save(root+'Split/data/trainx.npy',trainx)
    np.save(root+'Split/data/trainy.npy',trainy)
    np.save(root+'Split/data/testx.npy',testx)
    np.save(root+'Split/data/testy.npy',testy)
    print('Data splitted, saved at ' +root+ 'Split/data', flush=True)

    X = np.load(root+'Split/data/trainx.npy')
    X=np.transpose(X, (0, 2, 3, 1))
    y = np.load(root+'Split/data/trainy.npy')
    number_channels=X.shape[3]
    print('Input shape of the X features data: ',X.shape, flush= True)
    print('Input shape of the y label data: ',y.shape, flush= True)
    print('Number of input channel extracted from X is: ',number_channels, flush= True) 
    
    x_train,y_train = tcg_utils.normalize_channels(X,y)
    x_train=resize_preprocess(x_train, imsize[0], imsize[1], 'lanczos5')
    for target in targets:
        y_train_tg=y_train[:,target]
        if target==0:
            name='VMAX'
        if target==1:
            name='PMIN'
        if target==2:
            name='RMW'
        main(dense_layers=dense_layers,layer_sizes=layer_sizes,conv_layers=conv_layers,
               X=x_train,y=y_train_tg, target=name)
        count+=1
    print('Completed', flush=True)
    print(str(count)+' models built.', flush=True)
    
##################################################################
#Run
##################################################################


print('Initialization complete', flush=True)

DENSE_LAYER = [0,1,2]
LAYER_SIZES = [32]
CONV_LAYERS = [3]
root='/N/slate/kmluong/Training_data/'

build_models(dense_layers=DENSE_LAYER,layer_sizes=LAYER_SIZES,conv_layers=CONV_LAYERS, root=root, targets=[0,1,2])
