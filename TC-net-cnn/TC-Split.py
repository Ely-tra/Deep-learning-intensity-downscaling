#
# DESCRIPTION: This script is to split the input data after fixing NaN and selecting
#       specific variables into a training and test dataset. It requires data from
#       Step 3(or 4) that is in the Numpy format with both features and labels. 
#
# HIST: - May 16, 2024: created by Khanh Luong
#       - May 18, 2024: cross-checked and cleaned up by CK
#
# USAGE: edit the main call with proper paths and parameters before running this script
#
# AUTH: Khanh Luong (kmluong@iu.edu)
#====================================================================================
import numpy as np
import os
from sklearn.utils import shuffle
#
# Set the path and parameters before running this script to split the data
#
workdir = '/N/project/Typhoon-deep-learning/output'
windowsize = [20,20]
split_ratio = 10
var_num = 13

#####################################################################################
# DO NOT EDIT BELOW UNLESS YOU WANT TO MODIFY THE SCRIPT
#####################################################################################
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
#
# MAIN CALL: 
#
windows = str(windowsize[0])+'x'+str(windowsize[1])
data_directory = workdir+'/exp_'+str(var_num)+'features_'+windows+'/'
feature_file = 'CNNfeatures'+str(var_num)+'_'+windows+'fixed.npy'
label_file = 'CNNlabels'+str(var_num)+'_'+windows+'.npy'
if not os.path.exists(data_directory):
    print("Must have the input data by now....exit",data_directory)
    exit

# Load the data
features = np.load(data_directory + feature_file)
labels = np.load(data_directory + label_file)

# Split the data
train_features, train_labels, test_features, test_labels = split_data(features, labels, test_percentage=split_ratio)

# Save the split data
np.save(data_directory + 'train'+str(var_num)+'x_'+windows+'.npy', train_features)
np.save(data_directory + 'test'+str(var_num)+'x_'+windows+'.npy', test_features)
np.save(data_directory + 'train'+str(var_num)+'y_'+windows+'.npy', train_labels)
np.save(data_directory + 'test'+str(var_num)+'y_'+windows+'.npy', test_labels)

