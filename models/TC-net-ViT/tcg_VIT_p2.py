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
import pickle
import sys
import libtcg_utils as tcg_utils
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
num_classes = 1
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
]

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    # def call(self, images):
    #     input_shape = tf.shape(images)
    #     batch_size = input_shape[0]
    #     height = input_shape[1]
    #     width = input_shape[2]
    #     channels = input_shape[3]
    #     num_patches_h = height // self.patch_size
    #     num_patches_w = width // self.patch_size
    #     patches = tf.image.extract_patches(images, sizes=self.patch_size)
    #     patches = tf.reshape(
    #         patches,
    #         (
    #             batch_size,
    #             num_patches_h * num_patches_w,
    #             self.patch_size * self.patch_size * channels,
    #         ),
    #     )
    #     return patches

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

def create_vit_classifier(input_shape = (30,30,12)):
    inputs = keras.Input(shape=input_shape)
    # Augment data.
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
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def main(X=[],y=[],lead_time='00'):
    histories = []

    #optimizer = keras.optimizers.AdamW(
    #    learning_rate=learning_rate, weight_decay=weight_decay
    #)

    # model.compile(
    #     optimizer=optimizer,
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[
    #         keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    #         keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    #     ],
    # )

    model = create_vit_classifier(input_shape= (X.shape[1], X.shape[2], X.shape[3]))
    print(model.summary())
    model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.3)])

    # checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    # checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     checkpoint_filepath,
    #     monitor="val_accuracy",
    #     save_best_only=True,
    #     save_weights_only=True,
    # )

    # history = model.fit(
    #     x=x_train,
    #     y=y_train,
    #     batch_size=batch_size,
    #     epochs=num_epochs,
    #     validation_split=0.1,
    #     callbacks=[checkpoint_callback],
    # )

#    callbacks=[keras.callbacks.ModelCheckpoint("tcg_VIT"  + ".model_" + str(lead_time),monitor = "val_accuracy", save_weights_only=True, save_best_only=True)]
    callbacks=[keras.callbacks.ModelCheckpoint("tcg_VIT"  + ".model_" + str(lead_time) + ".keras", save_best_only=True)]
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)    
    hist = model.fit(X, Y, epochs = 100, batch_size = 128, validation_split=0.2, callbacks=[callbacks, early_stopping])
    print("History", hist)
    print("call back", callbacks)
    histories.append(hist.history)
    return histories

def view_history(history):

    val_accuracy = history.history['val_binary_accuracy']
    accuracy = history.history['binary_accuracy']
    epochs = history.epoch
    plt.plot(epochs,val_accuracy,'r',label="val binary_accuracy")
    plt.plot(epochs,accuracy,'b',label="train binary_accuracy")
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
       print("+ Example: tcg_ResNet_p2.py 00")
       exit()
    leadtime = str(sys.argv[1])


    print("+ Example: tcg_ResNet_p2.py 00")

    pickle_in = open("tcg_VIT_X.pickle","rb")
    X = pickle.load(pickle_in)
    pickle_in = open("tcg_VIT_y.pickle","rb")
    y = pickle.load(pickle_in)
    Y = np.array(y)
    number_channels=X.shape[3]
    print('Input shape of the X features data: ',X.shape)
    print('Input shape of the y label data: ',Y.shape)
    print('Number of input channel extracted from X is: ',number_channels)

    x_train,y_train = tcg_utils.normalize_channels(X,Y)
    print ("number of input examples = " + str(X.shape[0]))
    print ("X shape: " + str(X.shape))
    print ("Y shape: " + str(Y.shape))
    #
    # define the model architecture
    #
    histories = main(X=x_train,y=y_train,lead_time=leadtime)
    with open('./tcg_histories_vit.pickle', 'wb') as out:
        pickle.dump(histories,out)

    check_visualization = "no"
    if check_visualization== "yes":
        view_history(histories)
