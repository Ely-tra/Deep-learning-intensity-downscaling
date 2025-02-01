import numpy as np
import time
import sys
import argparse
import os
import re
def parse_args():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer model for TC intensity correction.')
    parser.add_argument('-r', '--root', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Working directory path')
    parser.add_argument('-ws', '--windowsize', type=int, nargs=2, default=[19, 19], help='Window size as two integers (e.g., 19 19)')
    parser.add_argument('-vno', '--var_num', type=int, default=13, help='Number of variables')
    parser.add_argument('-st', '--st_embed', type=bool, default=False, help='Including space-time embedded')
    parser.add_argument('-vym', '--validation_year_merra', nargs='+', type=int, default=[2014], help='Year(s) taken for validation (MERRA2 dataset)')
    parser.add_argument('-tym', '--test_year_merra', nargs='+', type=int, default=[2017], help='Year(s) taken for test (MERRA2 dataset)')
    parser.add_argument('-tew', '--test_experiment_wrf', nargs='+', type=int, default=[5], help='Experiment taken for test (WRF dataset)')
    parser.add_argument('-vew', '--validation_experiment_wrf', nargs='+', type=int, default=None, help='Experiment taken for validation (WRF dataset)')
    parser.add_argument('-ss', '--data_source', type=str, default='MERRA', help='Data source to conduct experiment on')
    parser.add_argument('-dxx', '--dxx', type=str, default='d01', help='Label quality for WRF experiments, d01 is the best, d03 is the worst')
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Temporary working folder')
    return parser.parse_args()

def set_variables_from_args(args):
    # Dynamically set global variables based on args
    for arg in vars(args):
        globals()[arg] = getattr(args, arg)

args = parse_args()
set_variables_from_args(args)
def get_year_directories(data_directory):
    """
    List all directory names within a given directory that are formatted as four-digit years.

    Parameters:
    - data_directory (str): Path to the directory containing potential year-named subdirectories.

    Returns:
    - list: A list of directory names that match the four-digit year format.
    """
    all_entries = os.listdir(data_directory)
    year_directories = [
        int(entry) for entry in all_entries
        if os.path.isdir(os.path.join(data_directory, entry)) and re.match(r'^\d{4}$', entry)
    ]
    return year_directories

def load_merra_data(data_directory, windowsize, validation_year=None, test_year=None):
    """
    Loads data from a specified directory and organizes it into training, testing, and (optionally)
    validation datasets. Instead of skipping the years specified as test_year, the data from those
    years are loaded into a test set. The returned dictionary is structured with keys corresponding
    to file names, for example, 'train_x.npy', 'train_y.npy', 'train_space_times.npy', etc.
    
    Args:
        data_directory (str): The root directory where data files are stored.
        validation_year (list, optional): List of years to be used for validation. Defaults to None.
        test_year (list, optional): List of years to be used for testing. Defaults to None.
    
    Returns:
        dict: A dictionary containing the datasets with keys:
              'train_x.npy', 'train_y.npy', 'train_space_times.npy',
              'test_x.npy', 'test_y.npy', 'test_space_times.npy',
              and if applicable:
              'val_x.npy', 'val_y.npy', 'val_space_times.npy'
    """
    # If not provided, set to empty lists
    windows = f'{windowsize[0]}x{windowsize[1]}'
    if validation_year is None:
        validation_year = []
    if test_year is None:
        test_year = []
    
    # Retrieve the list of year directories.
    # Assumes a helper function get_year_directories() exists and returns a list of year identifiers.
    years = get_year_directories(data_directory)
    months = range(1, 13)  # Assume months are labeled 1 to 12

    
    # Initialize containers for the three sets
    train_features, train_labels, train_space_times = [], [], []
    test_features, test_labels, test_space_times = [], [], []
    val_features, val_labels, val_space_times = [], [], []
    
    # Loop over each year directory
    for year in years:
        is_val = (year in validation_year)
        is_test = (year in test_year)
        
        if is_val:
            print("Validation year:", year)
        if is_test:
            print("Test year:", year)
        
        # Loop over each month within the year
        for month in months:
            # Construct file names; note that var_num and windows are assumed to be defined globally.
            feature_filename = f'features{var_num}_{windows}{month:02d}fixed.npy'
            label_filename = f'labels{var_num}_{windows}{month:02d}.npy'
            space_time_filename = f'space_time_info{var_num}_{windows}{month:02d}.npy'
            
            # Construct full paths
            feature_path = os.path.join(data_directory, str(year), feature_filename)
            label_path = os.path.join(data_directory, str(year), label_filename)
            space_time_path = os.path.join(data_directory, str(year), space_time_filename)
            
            # Check that all files exist before loading
            if os.path.exists(feature_path) and os.path.exists(label_path) and os.path.exists(space_time_path):
                features = np.load(feature_path)
                labels = np.load(label_path)
                space_time = np.load(space_time_path)
                
                # Append to the correct list based on the year
                if is_val:
                    val_features.append(features)
                    val_labels.append(labels)
                    val_space_times.append(space_time)
                elif is_test:
                    test_features.append(features)
                    test_labels.append(labels)
                    test_space_times.append(space_time)
                else:
                    train_features.append(features)
                    train_labels.append(labels)
                    train_space_times.append(space_time)
            else:
                print(f"Warning: Files not found for year {year} and month {month}")
    
    # Concatenate each list along the first dimension if any data was loaded
    if train_features:
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_space_times = np.concatenate(train_space_times, axis=0)
    if test_features:
        test_features = np.concatenate(test_features, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        test_space_times = np.concatenate(test_space_times, axis=0)
    if val_features:
        val_features = np.concatenate(val_features, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_space_times = np.concatenate(val_space_times, axis=0)
    
    # Package the data into a dictionary with keys as file names
    results = {}
    results['train_x.npy'] = train_features
    results['train_y.npy'] = train_labels
    results['train_space_times.npy'] = train_space_times
    results['test_x.npy'] = test_features
    results['test_y.npy'] = test_labels
    results['test_space_times.npy'] = test_space_times
    if validation_year:
        results['val_x.npy'] = val_features
        results['val_y.npy'] = val_labels
        results['val_space_times.npy'] = val_space_times
    
    return results
def load_wrf_data(workdir, dxx, test=None, val=None):
    """
    Reads and processes WRF data files from a specific directory, separating them into training, testing,
    and validation datasets based on provided numeric identifiers. Any file whose identifier (the two digits
    following 'm') is not listed in the test or validation identifiers is treated as training data.
    The output is packaged in a dictionary with keys corresponding to file names, for example,
    'train_x.npy', 'train_y.npy', 'test_x.npy', 'test_y.npy', etc.
    
    Parameters:
        workdir (str): The base directory where the data files are stored.
        dxx (str): Descriptor or resolution identifier to match specific data files.
        test (list or None): Numeric identifiers for test data sets, if any.
        val (list or None): Numeric identifiers for validation data sets, if any.
    
    Returns:
        dict: A dictionary containing numpy arrays for train, test, and validation datasets with keys:
              'train_x.npy', 'train_y.npy', 'test_x.npy', 'test_y.npy'
              and if applicable:
              'val_x.npy', 'val_y.npy'
    """
    # Ensure test and val are lists
    if test is None:
        test = []
    else:
        test = list(test) if isinstance(test, (list, tuple)) else [test]

    if val is None:
        val = []
    else:
        val = list(val) if isinstance(val, (list, tuple)) else [val]

    output_dir = os.path.join(workdir, "wrf_data")
    all_files = os.listdir(output_dir)
    
    # Initialize containers for each dataset
    datasets = {
        'train_x': [],
        'train_y': [],
        'test_x': [],
        'test_y': [],
        'val_x': [],
        'val_y': []
    }
    
    # Compile regex to capture files with the pattern: mXX_<dxx> where XX are exactly two digits.
    mxx_pattern = re.compile(r"m(\d{2})_" + re.escape(dxx))
    
    for file_name in all_files:
        if '_variables.npy' in file_name:
            m = mxx_pattern.search(file_name)
            if m:
                # Extract the two-digit identifier and convert to an integer
                mxx = int(m.group(1))
                # Determine the dataset type based on the identifier
                if mxx in test:
                    dataset_type = 'test'
                elif mxx in val:
                    dataset_type = 'val'
                else:
                    dataset_type = 'train'
            else:
                continue  # Skip files that do not match the mXX_<dxx> pattern
            
            # Load the features (x)
            data_path = os.path.join(output_dir, file_name)
            data_array = np.load(data_path)
            datasets[f'{dataset_type}_x'].append(data_array)
            
            # Determine the corresponding label file by replacing '_variables.npy' with '_{dxx}_ys.npy'
            label_file_name = file_name.replace('_variables.npy', f'_{dxx}_ys.npy')
            label_path = os.path.join(output_dir, label_file_name)
            label_array = np.load(label_path)
            datasets[f'{dataset_type}_y'].append(label_array)
    
    # Concatenate the lists along the first dimension if they are non-empty
    for key in datasets:
        if datasets[key]:
            datasets[key] = np.concatenate(datasets[key], axis=0)
    
    # Package the data into a dictionary with keys mimicking file names
    results = {}
    results['train_x.npy'] = datasets['train_x']
    results['train_y.npy'] = datasets['train_y']
    results['test_x.npy'] = datasets['test_x']
    results['test_y.npy'] = datasets['test_y']
    if val:  # Only include validation data if provided
        results['val_x.npy'] = datasets['val_x']
        results['val_y.npy'] = datasets['val_y']
    
    return results


def write_data(data_dict, work_folder):
    """
    Saves all arrays contained in data_dict as .npy files in a 'temp' subdirectory within work_folder.
    
    Parameters:
        data_dict (dict): A dictionary where each key is a file name (e.g., 'train_x.npy', 'test_y.npy', etc.)
                          and each value is the corresponding numpy array to save.
        work_folder (str): The directory in which the 'temp' subfolder will be created and the files saved.
    
    Outputs:
        The function writes each numpy array to a file whose name is the dictionary key.
    """
    # Create the target directory if it does not exist
    temp_folder = os.path.join(work_folder, 'temp')
    os.makedirs(temp_folder, exist_ok=True)
    
    # Iterate over the dictionary and save each array with the given file name
    for file_name, array in data_dict.items():
        file_path = os.path.join(temp_folder, file_name)
        np.save(file_path, array)
        print(f"Saved {file_path}")


if ss == 'MERRA':
    data_dir = os.path.join(root, f'exp_{var_num}features_{windowsize[0]}x{windowsize[1]}', 'data')
    results = load_merra_data(data_dir, validation_year=validation_year_merra, test_year=test_year_merra)
if ss == 'WRF':
    data_dir = os.path.join(root, 'wrf_data')
    results = load_wrf_data(data_dir, dxx, test = test_experiment_wrf, val=validation_experiment_wrf)
write_data(results, work_folder)
