"""
This script loads test data for a specific model, performs preprocessing and normalization,
then uses the model to predict outcomes. The results are then analyzed by computing RMSE
and MAE metrics, and visualized through boxplots and scatter plots to compare predicted
values against true values.
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras import backend as K
from matplotlib.lines import Line2D
from tensorflow.keras import layers
#
# Define parameters and data path. Note that x_size is the input data size. By default
# is (64x64) after resized for windowsize < 26x26. For a larger windown size, set it
# to 128.
#

workdir = "/N/slate/kmluong/TC-net-ViT_workdir/Domain_data/"
windowsize = [18,18]
mode = "VMAX"
st_embed = True
xfold = 7 #vi co ta sinh ngay 17-10
model_name = 'ViT_model1_fold' + str(xfold) + '_' + mode + ('_st' if st_embed else '')
exp_name = "exp_13features_" + str(windowsize[0])+'x'+str(windowsize[1])
directory = workdir + exp_name
data_dir = directory + '/data/'
model_dir = directory + '/model/' + model_name





def mode_switch(mode):
    switcher = {
        'VMAX': 0,
        'PMIN': 1,
        'RMW': 2
    }
    # Return the corresponding value if mode is found, otherwise return None or a default value
    return switcher.get(mode, None)
def load_data_fold(data_directory, xfold = xfold, mode = mode):
    months = range(1, 13)  # Months labeled 1 to 12
    b = mode_switch(mode)
    all_features = []
    all_labels = []
    all_space_times = []

    # Process only the specified fold
    for month in months:
        feature_filename = f'test_features_fold{xfold}_18x18{month:02d}fixed.npy'
        label_filename = f'test_labels_fold{xfold}_18x18{month:02d}fixed.npy'
        space_time_filename = f'test_spacetime_fold{xfold}_18x18{month:02d}fixed.npy'

        # Construct full paths
        feature_path = os.path.join(data_directory, feature_filename)
        label_path = os.path.join(data_directory, label_filename)
        space_time_path = os.path.join(data_directory, space_time_filename)

        # Check if files exist before loading
        if os.path.exists(feature_path) and os.path.exists(label_path):
            # Load the data
            features = np.load(feature_path)
            labels = np.load(label_path)[:, b]
            space_times = np.load(space_time_path)

        else:
            print(f"Warning: Files not found for fold {xfold} and month {month}")

    return features, labels, space_times
def root_mean_squared_error(y_true, y_pred):
    """Calculate root mean squared error."""
    m = tf.keras.metrics.RootMeanSquaredError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def MAE(y_true, y_pred):
    """Calculate mean absolute error."""
    m = tf.keras.metrics.MeanAbsoluteError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def mae_for_output(index):
    """Metric function to return MAE for a specific output index."""
    def mae(y_true, y_pred):
        return tf.keras.metrics.mean_absolute_error(y_true[:, index], y_pred[:, index])
    mae.__name__ = f'mae_{index+1}'
    return mae

def rmse_for_output(index):
    """Metric function to return RMSE for a specific output index."""
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:, index], y_pred[:, index]))
    rmse.__name__ = f'rmse_{index+1}'
    return rmse

def resize_preprocess(image, HEIGHT, WIDTH, method):
    """Resize and preprocess images using the specified method."""
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image

def normalize_channels(X, y):
    """Normalize the channel data for all samples in the dataset."""
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i, :, :, var].flat[np.abs(X[i, :, :, var]).argmax()]
            X[i, :, :, var] = X[i, :, :, var] / abs(maxvalue)
    print("Finish normalization...")
    return X, y
#
# MAIN CALL: Initialize dictionary to store results
#
datadict = {}
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

def normalize_Z(Z):
    Z[:,2] = (Z[:,2]+90) / 180
    Z[:,3] = (Z[:,3]+180) / 360
    return Z

#==============================================================================================
# Main call
#==============================================================================================

X, Y, Z = load_data_fold(data_dir, xfold)
X=np.transpose(X, (0, 2, 3, 1))

# Normalize the data before encoding
x, y = normalize_channels(X, Y)
z = normalize_Z(Z)
number_channels=x.shape[3]
# Load model and perform predictions
model = tf.keras.models.load_model(model_dir, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
name = model_name
predict = model.predict([x, z])

# Calculate metrics and store results
datadict[name + 'rmse'] = root_mean_squared_error(predict, y)
datadict[name + 'MAE'] = MAE(predict, y)
datadict[name] = predict

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
axs[0].boxplot([datadict[name].reshape(-1), y])
axs[0].grid(True)
axs[0].set_ylabel('Knots', fontsize=20)
axs[0].text(0.95, 0.05, '(a)', transform=axs[0].transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[0].set_xticklabels(['Predicted', 'Truth'], fontsize=20)


# Second subplot
axs[1].scatter(y, datadict[name].reshape(-1))
axs[1].grid()
axs[1].set_xlabel('Truth', fontsize=20)
axs[1].set_ylabel('Prediction', fontsize=20)
axs[1].text(0.95, 0.05, '(b)', transform=axs[1].transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
axs[1].plot(np.arange(min(y), max(y)), np.arange(min(y), max(y)), 'r-', alpha=0.8)
mae = datadict[name+'MAE']
rmse = datadict[name+'rmse']
axs[1].fill_between(np.arange(min(y), max(y)), np.arange(min(y), max(y)) + mae, np.arange(min(y), max(y)) - mae, color='red', alpha=0.3)
axs[1].tick_params(axis='both', which='major', labelsize=14)

# Legends with RMSE and MAE without markers
custom_lines = [
                Line2D([0], [0], color='red', lw=4, alpha=0.3),
                Line2D([0], [0], color='none', marker='', label=f'RMSE: {rmse:.2f}'),
                Line2D([0], [0], color='none', marker='', label=f'MAE: {mae:.2f}')]

axs[1].legend(custom_lines, [ 'MAE Area', f'RMSE: {rmse:.2f}', f'MAE: {mae:.2f}'], fontsize=12)

plt.savefig(directory + '/fig_' + str(name) + '.png')
print(f'Saved the result as Model:{name}.png')
print('RMSE = ' + str("{:.2f}".format(datadict[name + 'rmse'])) + ' and MAE = ' + str("{:.2f}".format(datadict[name + 'MAE'])))
print('Completed!')
