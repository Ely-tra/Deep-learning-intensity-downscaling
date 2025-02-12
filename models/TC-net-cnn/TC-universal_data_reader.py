import numpy as np
import time
import sys
import argparse
import os
from pathlib import Path
import re
def parse_args():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer model for TC intensity correction.')
    parser.add_argument('-r', '--root', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Working directory path')
    parser.add_argument('-ws', '--windowsize', type=int, nargs=2, default=[19, 19], help='Window size as two integers (e.g., 19 19)')
    parser.add_argument('-vno', '--var_num', type=int, default=13, help='Number of variables')
    parser.add_argument('-st', '--st_embed', type=int, default=0, help='Including space-time embedded')
    parser.add_argument('-vym', '--validation_year_merra', nargs='+', type=int, default=[2014], help='Year(s) taken for validation (MERRA2 dataset)')
    parser.add_argument('-tym', '--test_year_merra', nargs='+', type=int, default=[2017], help='Year(s) taken for test (MERRA2 dataset)')
    parser.add_argument('-tew', '--test_experiment_wrf', nargs='+', type=int, default=[5], help='Experiment taken for test (WRF dataset)')
    parser.add_argument('-vew', '--validation_experiment_wrf', nargs='+', type=int, default=None, help='Experiment taken for validation (WRF dataset)')
    parser.add_argument('-ss', '--data_source', type=str, default='MERRA2', help='Data source to conduct experiment on')
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Temporary working folder')
    parser.add_argument('-val_pc', '--validation_percentage', type=int, default=10, 
                        help='Validation set split percentage (based on train set), if a specific validation set is not provided')
    parser.add_argument('-wrf_eid', '--wrf_experiment_identification', type = str, default = 'H18h18', 
                        help =  'WRF experiment identification, H/L 18/06/02 h/l 18/06/02, please read wrf/extractor.py for a better understanding.')
    parser.add_argument('-wrf_ix', '--wrf_variables_imsize', type = int, nargs=2, default = [64,64], 
                        help = 'Image size for wrf variable data, for data identification only')
    parser.add_argument('-wrf_iy', '--wrf_labels_imsize', type = int, nargs=2, default = [64,64], 
                        help = 'Image size for wrf label data (data is extracted from this domain), for data identification only')
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
    results['train_z.npy'] = train_space_times
    results['test_x.npy'] = test_features
    results['test_y.npy'] = test_labels
    results['test_z.npy'] = test_space_times
    if validation_year:
        results['val_x.npy'] = val_features
        results['val_y.npy'] = val_labels
        results['val_z.npy'] = val_space_times
    
    return results
def load_wrf_data(workdir, eid, ix, iy, test=None, val=None):
    """
    Load and split paired x and y .npy files from the given working directory.
    
    Files are assumed to be in the subfolder: workdir / "wrf_data"
    
    The x-files must follow the pattern:
        x_{eid}_{ix[0]}_{ix[1]}_m<digits>.npy
    and the y-files must follow the pattern:
        y_{eid}_{iy[0]}_{iy[1]}_m<digits>.npy

    The <digits> part (extracted from after '_m' and before '.npy') is converted
    to an integer. If this integer is in the provided 'test' collection, the pair
    is added to the test split. If in the 'val' collection (if provided), it is
    added to the validation split. Otherwise, the pair is added to the training split.
    
    Parameters:
        workdir (str or Path): The base directory containing the "wrf_data" folder.
        eid (str or int): The experiment ID used in the file names.
        ix (list or tuple): Two elements for the x file indices.
        iy (list or tuple): Two elements for the y file indices.
        test (iterable of int, optional): m-numbers to assign to the test set.
        val (iterable of int, optional): m-numbers to assign to the validation set.
    
    Returns:
        dict: A dictionary with keys:
            'train_x.npy', 'train_y.npy', 'test_x.npy', 'test_y.npy'
        and if a validation set is provided:
            'val_x.npy', 'val_y.npy'
        Each key maps to a list of the loaded NumPy arrays.
    """
    # Ensure workdir is a Path object and point to the "wrf_data" subfolder.
    workdir = Path(workdir)
    data_dir = workdir

    # Prepare the test and validation sets.
    test_set = set(test) if test is not None else set()
    if val is not None:
        val_set = set(val)

    # Initialize dataset containers.
    datasets = {
        'train_x': [],
        'train_y': [],
        'test_x': [],
        'test_y': []
    }
    if val is not None:
        datasets['val_x'] = []
        datasets['val_y'] = []

    # Define the pattern for x files using the ix indices.
    pattern_x = f"x_{eid}_{ix[0]}x{ix[1]}_m*.npy"
    x_files = sorted(data_dir.glob(pattern_x))
    # Regular expression to extract the m-number from the filename.
    m_regex = re.compile(r'_m(\d+)\.npy$')

    # Process each x file.
    for x_file in x_files:
        match = m_regex.search(x_file.name)
        if not match:
            # Skip any file that doesn't match the expected pattern.
            continue
        m_str = match.group(1)
        try:
            m_number = int(m_str)
        except ValueError:
            continue  # skip if conversion fails

        # Construct the corresponding y file name using the iy indices.
        y_filename = f"y_{eid}_{iy[0]}x{iy[1]}_m{m_str}.npy"
        y_file = data_dir / y_filename

        if not y_file.exists():
            print(f"Warning: y file '{y_filename}' does not exist for x file '{x_file.name}'. Skipping this pair.")
            continue

        # Load the numpy arrays.
        try:
            x_data = np.load(x_file)
            y_data = np.load(y_file)
        except Exception as e:
            print(f"Error loading files '{x_file.name}' and/or '{y_filename}': {e}")
            continue

        # Assign to the appropriate split.
        if m_number in test_set:
            datasets['test_x'].append(x_data)
            datasets['test_y'].append(y_data)
        elif val is not None and m_number in val_set:
            datasets['val_x'].append(x_data)
            datasets['val_y'].append(y_data)
        else:
            datasets['train_x'].append(x_data)
            datasets['train_y'].append(y_data)

    # Prepare the results dictionary.
    results = {
        'train_x.npy': f_r(datasets['train_x']),
        'train_y.npy': f_r(datasets['train_y']),
        'test_x.npy': f_r(datasets['test_x']),
        'test_y.npy': f_r(datasets['test_y'])
    }
    if val is not None:
        results['val_x.npy'] = datasets['val_x']
        results['val_y.npy'] = datasets['val_y']

    return results

def f_r(ilist):
    ''' A quick reshaper'''
    return np.concatenate(ilist, axis=0)

def write_data(data_dict, work_folder, val_pc=20):
    """
    Saves all arrays contained in data_dict as .npy files in a 'temp' subdirectory within work_folder.
    If necessary validation files are missing, it shuffles and splits the training data to create them.
    
    Parameters:
        data_dict (dict): A dictionary where each key is a file name (e.g., 'train_x.npy', 'train_y.npy', etc.)
                          and each value is the corresponding numpy array to save.
        work_folder (str): The directory in which the 'temp' subfolder will be created and the files saved.
        val_pc (int): The percentage of data to be used as validation data if validation files are missing.
    
    Outputs:
        The function writes each numpy array to a file whose name is the dictionary key.
    """
    # Create the target directory if it does not exist
    temp_folder = os.path.join(work_folder, 'temp')
    os.makedirs(temp_folder, exist_ok=True)
    if os.listdir(temp_folder):  # This checks if the folder is not empty
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and contents
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    # Check for validation files
    if 'val_x.npy' not in data_dict or 'val_y.npy' not in data_dict:
        # Determine which arrays need splitting
        train_x = data_dict.get('train_x.npy')
        train_y = data_dict.get('train_y.npy')
        train_z = data_dict.get('train_z.npy') if 'train_z.npy' in data_dict else None
        
        # Shuffle data
        indices = np.random.permutation(len(train_x))
        split_idx = int(len(indices) * (1 - val_pc / 100))
        # Split and assign training and validation data
        data_dict['train_x.npy'] = train_x[indices[:split_idx]]
        data_dict['val_x.npy'] = train_x[indices[split_idx:]]
        data_dict['train_y.npy'] = train_y[indices[:split_idx]]
        data_dict['val_y.npy'] = train_y[indices[split_idx:]]
        
        if train_z is not None:
            data_dict['train_z.npy'] = train_z[indices[:split_idx]]
            data_dict['val_z.npy'] = train_z[indices[split_idx:]]

    # Iterate over the dictionary and save each array with the given file name
    for file_name, array in data_dict.items():
        file_path = os.path.join(temp_folder, file_name)
        np.save(file_path, array)
        print(f"Saved {file_path}")

if data_source == 'MERRA2':
    data_dir = os.path.join(root, f'exp_{var_num}features_{windowsize[0]}x{windowsize[1]}', 'data')
    results = load_merra_data(data_dir,windowsize, validation_year=validation_year_merra, test_year=test_year_merra)
if data_source == 'WRF':
    data_dir = os.path.join(root, 'wrf_data')
    results = load_wrf_data(data_dir, wrf_experiment_identification, wrf_variables_imsize, wrf_labels_imsize, test = test_experiment_wrf, val=validation_experiment_wrf)
write_data(results, work_folder, val_pc = validation_percentage)
