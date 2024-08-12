import numpy as np
import os
from sklearn.utils import shuffle

#
# Set the path and parameters before running this script to split the data
#
workdir = '/N/slate/kmluong/TC-net-cnn_workdir/Domain_data/'
windowsize = [18,18]
split_ratio = 10
var_num = 13

def split_data(features, labels, test_percentage=10):
    """
    Shuffle and split the data into training and testing datasets.
    
    Parameters:
    - features (numpy.ndarray): Array of input features.
    - labels (numpy.ndarray): Array of corresponding labels.
    - test_percentage (int): Percentage of the data to be used as test set (default is 10).
    
    Returns:
    - tuple: Tuple containing:
        - train_features (numpy.ndarray): Features for the training set.
        - train_labels (numpy.ndarray): Labels for the training set.
        - test_features (numpy.ndarray): Features for the testing set.
        - test_labels (numpy.ndarray): Labels for the testing set.
    """
    # Shuffle the data together to maintain alignment
    indices = np.arange(len(features))
    shuffled_indices = shuffle(indices, random_state=0)
    features = features[shuffled_indices]
    labels = labels[shuffled_indices]

    # Compute the split index
    split_idx = int(len(features) * (test_percentage / 100))

    # Split the data into training and testing sets
    test_features = features[:split_idx]
    test_labels = labels[:split_idx]
    train_features = features[split_idx:]
    train_labels = labels[split_idx:]

    return train_features, train_labels, test_features, test_labels

# MAIN CALL:
windows = f"{windowsize[0]}x{windowsize[1]}"
data_directory = f"{workdir}/exp_{var_num}features_{windows}/monthly/"

# Ensure the data directory exists
if not os.path.exists(data_directory):
    print(f"Must have the input data by now....exit {data_directory}")
    exit()

all_train_features = []
all_train_labels = []

# Process each pair of feature and label files
for file in os.listdir(data_directory):
    if file.startswith('CNNfeatures') and 'fixed.npy' in file:
        feature_file = file
        label_file = file.replace('features', 'labels').replace('fixed.npy', '.npy')
        
        # Load the data
        features = np.load(data_directory + feature_file)
        labels = np.load(data_directory + label_file)
        
        # Split the data
        train_features, train_labels, test_features, test_labels = split_data(features, labels, test_percentage=split_ratio)
        
        # Append to list for later concatenation
        all_train_features.append(train_features)
        all_train_labels.append(train_labels)
        
        # Save the test data
        base_name = feature_file.split('_')[1]  # To get the variant number and use it in the saved filenames
        np.save(data_directory + f'test{base_name}_x.npy', test_features)
        np.save(data_directory + f'test{base_name}_y.npy', test_labels)

# Concatenate and save all training data
all_train_features = np.concatenate(all_train_features, axis=0)
all_train_labels = np.concatenate(all_train_labels, axis=0)
# Generate a random permutation of indices
indices = np.random.permutation(all_train_features.shape[0])

# Shuffle the features and labels using the same indices
all_train_features = all_train_features[indices]
all_train_labels = all_train_labels[indices]
np.save(data_directory + 'merged_train_features.npy', all_train_features)
np.save(data_directory + 'merged_train_labels.npy', all_train_labels)

