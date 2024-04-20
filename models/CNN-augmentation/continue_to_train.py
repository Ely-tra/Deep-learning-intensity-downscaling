import tensorflow as tf
import numpy as np
import time
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
root='/N/slate/kmluong/Training_data/Split/'
X = np.load(root+'data/train13x.npy')
X=np.transpose(X, (0, 2, 3, 1))
y = np.load(root+'data/train13y.npy')[:,2]
number_channels=X.shape[3]
modir = root+'model/'+'Rparam13testmodel2'
init_epoch=641
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
activ='relu'
x_train,y_train = normalize_channels(X,y)
'''
for i in range(y.shape[1]):
        print(np.max(y[:,i]))
        y_train[:,i]=y[:,i]/np.max(y[:,i])
        '''
x_train=resize_preprocess(x_train, 64,64, 'lanczos5')
def lr_scheduler(epoch, lr):
                  lr0=0.001
                  lr = -0.0497 + (1.0 - (-0.0497)) / (1 + (epoch / 107.0) ** 1.35)
                  if epoch>940:
                    lr=0.0001
                  return lr*lr0
def mae_for_output(index):
    def mae(y_true, y_pred):
        return tf.keras.metrics.mean_absolute_error(y_true[:, index], y_pred[:, index])
    mae.__name__ = f'mae_{index+1}'  # Naming for clarity in logs
    return mae

def rmse_for_output(index):
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:, index], y_pred[:, index]))
    rmse.__name__ = f'rmse_{index+1}'  # Naming for clarity in logs
    return rmse
custom_objects = {
    'mae_1': mae_for_output(0),
    'rmse_1': rmse_for_output(0)
}
callbacks=[keras.callbacks.ModelCheckpoint(modir,save_best_only=True), keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]
model=tf.keras.models.load_model(modir, custom_objects=custom_objects)
model.fit(x_train, y_train, batch_size=128, epochs=3000, validation_split=2/9,verbose=2, callbacks=callbacks, initial_epoch=init_epoch)
