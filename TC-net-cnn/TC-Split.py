import numpy as np
from sklearn.utils import shuffle

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
    # Shuffle the data
    features, labels = shuffle(features, labels, random_state=0)

    # Compute the split index
    split_idx = int(len(features) * (test_percentage / 100))

    # Split the data into training and testing sets
    test_features = features[:split_idx]
    test_labels = labels[:split_idx]
    train_features = features[split_idx:]
    train_labels = labels[split_idx:]

    return train_features, train_labels, test_features, test_labels

# Set the path to the data
data_directory = '/N/slate/kmluong/Training_data/'
feature_file = 'CNNfeatures13.25x25fixed.npy'
label_file = 'CNNlabels13.25x25.npy'

# Load the data
features = np.load(data_directory + feature_file)
labels = np.load(data_directory + label_file)

# Split the data
train_features, train_labels, test_features, test_labels = split_data(features, labels, test_percentage=10)

# Save the split data
np.save(data_directory + 'Split/data/train13x.25x25.npy', train_features)
np.save(data_directory + 'Split/data/test13x.25x25.npy', test_features)
np.save(data_directory + 'Split/data/train13y.25x25.npy', train_labels)
np.save(data_directory + 'Split/data/test13y.25x25.npy', test_labels)

