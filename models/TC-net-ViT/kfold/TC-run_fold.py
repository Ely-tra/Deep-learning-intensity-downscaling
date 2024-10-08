import numpy as np
import tensorflow as tf
import os
import re
import sys
# Define the working directory
if len(sys.argv) > 1:
    xfold = sys.argv[1]
else:
    xfold = ''
workdir = "/N/slate/kmluong/TC-net-cnn_workdir/Domain_data/exp_13features_18x18/kfold/"

# Define the file path for the VMAX model based on the workdir
#vmax_model_path = '/N/slate/kmluong/TC-net-cnn_workdir/Domain_data/training/modeltime'
vmax_model_path = os.path.join(workdir,f"model_VMAX13_18x18fold{xfold}")
def evaluate_model_on_fold(data_directory, xfold, vmax_model):
    months = range(1, 13)  # Months labeled 1 to 12

    # Loop over each month
    for month in months:
        feature_filename = f'test_features_fold{xfold}_18x18{month:02d}fixed.npy'
        label_filename = f'test_labels_fold{xfold}_18x18{month:02d}fixed.npy'
        #print(f'Loading {feature_filename}')
        # Construct full paths
        feature_path = os.path.join(data_directory, feature_filename)
        label_path = os.path.join(data_directory, label_filename)

        # Check if files exist before loading
        if os.path.exists(feature_path) and os.path.exists(label_path):
            # Load the data
            x_test = np.load(feature_path)
            x_test = np.transpose(x_test, (0, 2, 3, 1))
            y_true = np.load(label_path)[:,0]
            x_test = normalize_channels(x_test)
            x_test = resize_preprocess(x_test, 64, 64, 'lanczos5')

            # Make predictions using the loaded model
            y_pred = vmax_model.predict(x_test, verbose = 0)

            # Calculate RMSE and MAE
            rmse = root_mean_squared_error(y_true, y_pred)
            mae = MAE(y_true, y_pred)

            # Print the results
            print(f"{xfold}, {month} RMSE: {rmse:.2f} MAE: {mae:.2f}")
        else:
            print(f"Warning: Files not found for fold {xfold} and month {month}")

# Custom metrics and functions
def mae_for_output(index):
    def mae(y_true, y_pred):
        return tf.keras.metrics.mean_absolute_error(y_true[:, index], y_pred[:, index])
    mae.__name__ = f'mae_{index+1}'
    return mae

def rmse_for_output(index):
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:, index], y_pred[:, index]))
    rmse.__name__ = f'rmse_{index+1}'
    return rmse

def resize_preprocess(image, HEIGHT, WIDTH, method):
    image = tf.image.resize(image, (HEIGHT, WIDTH), method=method)
    return image

def normalize_channels(X):
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i, :, :, var].flat[np.abs(X[i, :, :, var]).argmax()]
            X[i, :, :, var] = X[i, :, :, var] / abs(maxvalue)
    #print("Finish normalization...")
    return X

# Load the VMAX model
custom_objects = {
    'mae_1': mae_for_output(0),
    'rmse_1': rmse_for_output(0)
}
vmax_model = tf.keras.models.load_model(vmax_model_path, custom_objects=custom_objects)
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

evaluate_model_on_fold(workdir, xfold, vmax_model)


print("Processing complete.")

