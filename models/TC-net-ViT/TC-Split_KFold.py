import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import argparse

def split_data(features, labels, spacetime=0, test_percentage=10):
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
    if spacetime.any() != 0:
        spacetime = spacetime[shuffled_indices]
    
    # Compute the split index
    split_idx = int(len(features) * (test_percentage / 100))

    # Split the data into training and testing sets
    test_features = features[:split_idx]
    test_labels = labels[:split_idx]
    train_features = features[split_idx:]
    train_labels = labels[split_idx:]
    if spacetime.any() != 0:
        test_spacetime = spacetime[:split_idx]
        train_spacetime = spacetime[split_idx:]
    if spacetime.all() == 0:
        return train_features, train_labels, test_features, test_labels
    else:
        return train_features, train_spacetime, train_labels, test_features, test_spacetime, test_labels

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Data splitting and K-Fold cross-validation.")
    parser.add_argument("--workdir", type=str, required=True, help="Directory to save output data")
    parser.add_argument("--windowsize", type=int, nargs=2, required=True, help="Window size for data extraction as two integers, e.g., '19 19'")
    parser.add_argument("--var_num", type=int, required=True, help="Number of variables to process")
    parser.add_argument("--kfold", type=int, required=True, help="K-fold value for cross-validation")
    
    # Parse arguments
    args = parser.parse_args()
    workdir = args.workdir
    windowsize = list(args.windowsize)
    var_num = args.var_num
    k = args.kfold
    
    # Set up paths and ensure the data directory exists
    windows = f"{windowsize[0]}x{windowsize[1]}"
    data_directory = f"{workdir}/exp_{var_num}features_{windows}/data/"

    if not os.path.exists(data_directory):
        print(f"Must have the input data by now....exit {data_directory}")
        exit()

    # Set up K-Fold
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    # Process each file in the data directory
    for file in os.listdir(data_directory):
        if file.startswith('CNNfeatures') and 'fixed.npy' in file:
            feature_file = file
            label_file = file.replace('features', 'labels').replace('fixed.npy', '.npy')
            spacetimefile = file.replace('features', 'space_time_info').replace('fixed.npy', '.npy')

            # Load the data
            features = np.load(data_directory + feature_file)
            labels = np.load(data_directory + label_file)
            spacetime = np.load(data_directory + spacetimefile)

            # Initialize lists to hold concatenated training data
            all_train_features = []
            all_train_labels = []
            all_train_spacetime = []

            # K-fold cross-validation
            fold = 1
            for train_index, test_index in kf.split(features):
                train_features, test_features = features[train_index], features[test_index]
                train_labels, test_labels = labels[train_index], labels[test_index]
                train_spacetime, test_spacetime = spacetime[train_index], spacetime[test_index]

                # Append to list for later concatenation
                all_train_features.append(train_features)
                all_train_labels.append(train_labels)
                all_train_spacetime.append(train_spacetime)

                # Save the test data
                base_name = feature_file.split('_')[1]  # Get the variant number for filenames
                np.save(data_directory + f'test_features_fold{fold}_{base_name}', test_features)
                np.save(data_directory + f'test_labels_fold{fold}_{base_name}', test_labels)
                np.save(data_directory + f'test_spacetime_fold{fold}_{base_name}', test_spacetime)
                fold += 1
    print('Completed')

