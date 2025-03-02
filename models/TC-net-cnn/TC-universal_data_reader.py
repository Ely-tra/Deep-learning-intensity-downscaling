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
    parser.add_argument('-xew', '--train_experiment_wrf', type=str, nargs='+', 
                        default=["exp_02km_m01", "exp_02km_m02", "exp_02km_m04", "exp_02km_m05"],
                        help='WRF experiment folders for training (inputs)')
    parser.add_argument('-tew', '--test_experiment_wrf', type=str, nargs='+', 
                        default=["exp_02km_m03"],
                        help='WRF experiment folders for testing (targets)')
    parser.add_argument('-vew', '--val_experiment_wrf', type=str, nargs='*', default=[], 
                        help='WRF experiment folders for validation (default: empty list)')
    parser.add_argument('-xd', '--X_resolution', type=str, default='d01', 
                        help='X resolution string in filename (e.g. d01)')
    parser.add_argument('-td', '--Y_resolution', type=str, default='d01', 
                        help='Y resolution string in filename (e.g. d01)')
    parser.add_argument('-ss', '--data_source', type=str, default='MERRA2', help='Data source to conduct experiment on')
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Temporary working folder')
    parser.add_argument('-val_pc', '--validation_percentage', type=int, default=10, 
                        help='Validation set split percentage (based on train set), if a specific validation set is not provided.')
    parser.add_argument('-test_pc', '--test_percentage', type=int, default=10, 
                        help='Test set split percentage (based on train set), if a specific test set is not provided.')
    parser.add_argument('-wrf_ix', '--wrf_variables_imsize', type = int, nargs=2, default = [64,64], 
                        help = 'Image size for wrf variable data, for data identification only')
    parser.add_argument('-wrf_iy', '--wrf_labels_imsize', type = int, nargs=2, default = [64,64], 
                        help = 'Image size for wrf label data (data is extracted from this domain), for data identification only')
    parser.add_argument('-tid', '--temp_id', type=str, default='testtemp')
    parser.add_argument('-r_split', '--random_split', type=int, default=0, help = 'Perform random split or not')
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

def load_merra_data_by_percentage(data_directory, windowsize, val_pc=20, test_pc=10):
    """
    Loads data from all available year directories in the given root directory, concatenates
    them, and then splits the combined dataset into training, validation, and test sets based on 
    the provided percentages.
    
    Args:
        data_directory (str): The root directory where data files are stored.
        windowsize (tuple): A tuple (height, width) defining the window size.
        val_pc (float): Fraction (between 0 and 1) of the total data to use as the validation set.
        test_pc (float): Fraction (between 0 and 1) of the total data to use as the test set.
        
    Returns:
        dict: A dictionary with the following keys and their corresponding data arrays:
              'train_x.npy', 'train_y.npy', 'train_z.npy',
              'test_x.npy', 'test_y.npy', 'test_z.npy',
              'val_x.npy', 'val_y.npy', 'val_z.npy'
    """
    windows = f'{windowsize[0]}x{windowsize[1]}'
    # Retrieve the list of year directories (assumes get_year_directories is defined)
    years = get_year_directories(data_directory)
    months = range(1, 13)  # Assumes months 1 to 12
    
    # Containers for all data
    all_features, all_labels, all_space_times = [], [], []
    
    # Loop over each year and month to load available data files
    for year in years:
        for month in months:
            feature_filename = f'features{var_num}_{windows}{month:02d}fixed.npy'
            label_filename = f'labels{var_num}_{windows}{month:02d}.npy'
            space_time_filename = f'space_time_info{var_num}_{windows}{month:02d}.npy'
            
            feature_path = os.path.join(data_directory, str(year), feature_filename)
            label_path = os.path.join(data_directory, str(year), label_filename)
            space_time_path = os.path.join(data_directory, str(year), space_time_filename)
            
            # Load the files only if they exist
            if os.path.exists(feature_path) and os.path.exists(label_path) and os.path.exists(space_time_path):
                features = np.load(feature_path)
                labels = np.load(label_path)
                space_time = np.load(space_time_path)
                
                all_features.append(features)
                all_labels.append(labels)
                all_space_times.append(space_time)
            else:
                print(f"Warning: Files not found for year {year} and month {month:02d}")
    
    # Concatenate the loaded arrays if any data was found
    if not all_features:
        print("No data loaded!")
        return {}
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_space_times = np.concatenate(all_space_times, axis=0)
    
    # Determine the total number of samples and compute split sizes
    total_samples = all_features.shape[0]
    test_samples = int(total_samples * test_pc/100)
    val_samples = int(total_samples * val_pc/100)
    
    # Shuffle indices to randomize the data split
    indices = np.random.permutation(total_samples)
    
    # Split indices for test, validation, and training sets
    test_indices = indices[:test_samples]
    val_indices = indices[test_samples:test_samples + val_samples]
    train_indices = indices[test_samples + val_samples:]
    
    # Partition the data using the computed indices
    train_x = all_features[train_indices]
    train_y = all_labels[train_indices]
    train_z = all_space_times[train_indices]
    
    test_x = all_features[test_indices]
    test_y = all_labels[test_indices]
    test_z = all_space_times[test_indices]
    
    val_x = all_features[val_indices]
    val_y = all_labels[val_indices]
    val_z = all_space_times[val_indices]
    
    # Package the split data into a dictionary
    results = {
        'train_x.npy': train_x,
        'train_y.npy': train_y,
        'train_z.npy': train_z,
        'test_x.npy': test_x,
        'test_y.npy': test_y,
        'test_z.npy': test_z,
        'val_x.npy': val_x,
        'val_y.npy': val_y,
        'val_z.npy': val_z
    }
    
    return results
def load_wrf_data(workdir, xd, td, ix, iy, train=None, test=None, val=None):
    """
    Load and split paired x and y .npy files from the given working directory.
    
    Files are assumed to be in the subfolder: workdir / "wrf_data"
    
    The x-files must follow the pattern:
        x_{xd}_{ix[0]}x{ix[1]}_{exp}.npy
    and the y-files must follow the pattern:
        y_{td}_{iy[0]}x{iy[1]}_{exp}.npy

    Here, xd and td denote the x and y resolutions, ix and iy the image sizes,
    and exp represents the experiment id. The exp value is used to assign each pair
    to the appropriate split: if exp is in the provided test collection, the pair
    is added to the test split; if in the validation collection (if provided), it is
    added to the validation split; otherwise, it is added to the training split.
    
    Parameters:
        workdir (str or Path): The base directory containing the "wrf_data" folder.
        xd (str): The x resolution used in the file names.
        td (str): The y resolution used in the file names.
        ix (list or tuple): Two elements for the x file image size.
        iy (list or tuple): Two elements for the y file image size.
        test (iterable, optional): Experiment ids to assign to the test set.
        val (iterable, optional): Experiment ids to assign to the validation set.
        train (iterable, optional): Experiment ids to assign to the training set (unused –
                                      any exp not in test or val goes to train).
    
    Returns:
        dict: A dictionary with keys:
            'train_x.npy', 'train_y.npy', 'test_x.npy', 'test_y.npy'
        and if a validation set is provided:
            'val_x.npy', 'val_y.npy'
        Each key maps to a list of the loaded NumPy arrays.
    """
    # Ensure workdir is a Path object and point to the "wrf_data" subfolder.
    workdir = Path(workdir)
    data_dir = workdir  # adjust if your "wrf_data" folder is in a subfolder, e.g., workdir / "wrf_data"
    
    # Prepare the test and validation sets.
    test_set = set(test) if test is not None else set()
    val_set = set(val) if val is not None else set()
    
    # Initialize dataset containers.
    datasets = {
        'train_x': [],
        'train_y': [],
        'test_x': [],
        'test_y': []
    }
    if val[0] != '':
        datasets['val_x'] = []
        datasets['val_y'] = []
    
    # Define the pattern for x files using the provided resolution and image size.
    pattern_x = f"x_{xd}_{ix[0]}x{ix[1]}_*.npy"
    x_files = sorted(data_dir.glob(pattern_x))
    
    # Regular expression to extract the experiment id (exp) from the filename.
    # This assumes the filename ends with _<exp>.npy
    exp_regex = re.compile(rf'_([^_]+)\.npy$')
     
    # Process each x file.
    for x_file in x_files:
        prefix = f"x_{xd}_{ix[0]}x{ix[1]}_"
        if x_file.name.startswith(prefix) and x_file.name.endswith(".npy"):
            exp = x_file.name[len(prefix):-4]  # Remove the prefix and the '.npy'
        y_filename = f"y_{td}_{iy[0]}x{iy[1]}_{exp}.npy"
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
        if exp in test_set:
            datasets['test_x'].append(x_data)
            datasets['test_y'].append(y_data)
        elif exp in val_set:
            datasets['val_x'].append(x_data)
            datasets['val_y'].append(y_data)
        else:
            datasets['train_x'].append(x_data)
            datasets['train_y'].append(y_data)
    
    # Prepare the results dictionary.
    results = {
        'train_x.npy': np.concatenate(datasets['train_x']),
        'train_y.npy': np.concatenate(datasets['train_y']),
        'test_x.npy': np.concatenate(datasets['test_x']),
        'test_y.npy': np.concatenate(datasets['test_y'])
    }
    if val[0] != '':
        results['val_x.npy'] = np.concatenate(datasets['val_x'])
        results['val_y.npy'] = np.concatenate(datasets['val_y'])
    
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
        file_path = os.path.join(temp_folder, f'{file_name[:-4]}_{temp_id}.npy')
        np.save(file_path, array)
        print(f"Saved {file_path}")

if data_source == 'MERRA2':
    data_dir = os.path.join(root, 'Domain_data', f'exp_{var_num}features_{windowsize[0]}x{windowsize[1]}', 'data')
    if random_split:
        results = load_merra_data_by_percentage(data_dir,windowsize, val_pc=validation_percentage, test_pc=test_percentage)
    else:
        results = load_merra_data(data_dir,windowsize, validation_year=validation_year_merra, test_year=test_year_merra)
if data_source == 'WRF':
    data_dir = os.path.join(root, 'wrf_data')
    results = load_wrf_data(data_dir, X_resolution, Y_resolution,wrf_variables_imsize, wrf_labels_imsize,train = train_experiment_wrf, test = test_experiment_wrf, val=val_experiment_wrf)
write_data(results, work_folder, val_pc = validation_percentage)
