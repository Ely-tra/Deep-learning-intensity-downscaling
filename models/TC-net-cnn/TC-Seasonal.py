import numpy as np
import tensorflow as tf
import os
import re

# Define the working directory
workdir = "/N/slate/kmluong/TC-net-cnn_workdir/Domain_data/exp_13features_18x18/monthly/"

# Define the file path for the VMAX model based on the workdir
vmax_model_path = os.path.join(workdir, "model_VMAX13_18x18")

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
    print("Finish normalization...")
    return X

# Load the VMAX model
custom_objects = {
    'mae_1': mae_for_output(0),
    'rmse_1': rmse_for_output(0)
}
vmax_model = tf.keras.models.load_model(vmax_model_path, custom_objects=custom_objects)

# Function to find and process files
def process_files(workdir):
    for filename in os.listdir(workdir):
        if 'test' in filename and 'fixed' in filename and 'npy_x' in filename:
            match = re.search(r'test18x18(\d{2})', filename)
            if match:
                x = match.group(1)
                input_file_path = os.path.join(workdir, filename)
                input_data = np.load(input_file_path)
                print(input_data.shape)
                input_data = resize_preprocess(normalize_channels(np.transpose(input_data, (0, 2, 3, 1))), 64, 64, 'lanczos5')
                
                # Run the VMAX model and save the outputs
                output_data = vmax_model.predict(input_data)
                output_filename = f"resultVMAX{x}.npy"
                output_file_path = os.path.join(workdir, output_filename)
                np.save(output_file_path, output_data)
                print(f"Saved results to {output_file_path}")

# Run the file processing
process_files(workdir)

print("Processing complete.")

