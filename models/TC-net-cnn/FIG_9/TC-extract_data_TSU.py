# DESCRIPTION: 
#       This script is to read a post-processed data in the NETCDF format centered
#       on each TC from the previous script MERRA2tc_domain.py, and then seclect a specific
#       group of variables before writing them out in the Numpy array format. This latter
#       format will help the DL model to read in and train more efficiently without the 
#       memory issues, similar to the CNN-agumentation model system.
#
#       This script initializes all necessary libraries for data handling and defines a 
#       function to process atmospheric data from NetCDF files, structuring it into 
#       NumPy arrays suitable for analysis. It ensures efficient management 
#       of large datasets with an emphasis on maintaining data integrity through robust 
#       error handling and validation steps. The function supports custom configurations 
#       for selective data extraction and transformation, tailored for climate research.
#
# DEPENDENCIES:
#       - xarray: Utilized for data manipulation and accessing NetCDF file content.
#       - matplotlib: Incorporated for plotting capabilities.
#       - numpy: Essential for numerical computations and data structuring.
#       - glob: Facilitates file path handling for navigating directory structures.
#       - npy_append_array: efficient append operations to NumPy arrays, optimizing memory.
#       - math: Provides support for basic mathematical operations.
#
# USAGE: Edit all required input/output location and parameters right below before running 
#
# NOTE: This script selects specific level atmospheric data from the MERRA2 dataset,
#       which may also contain NaN. To add additional data layers, see variable
#       data_array_x for details.
#
# HIST: - May 14, 2024: created by Khanh Luong
#       - May 16, 2024: clean up and added more note by CK
#       - Oct 19, 2024: added a list of vars to be processed by CK
#       - Oct 26, 2024: added argument parsers by TN  
# AUTH: Minh Khanh Luong @ Indiana University Bloomington (email: kmluong@iu.edu)
#==========================================================================================
print('Initiating.', flush=True)
import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from npy_append_array import NpyAppendArray
import math
from datetime import datetime
import argparse
import re

#####################################################################################
# Arguments parser, arguments processing
#####################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='Process MERRA2 data for TC domain.')
    parser.add_argument('--workdir', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Working directory path')
    parser.add_argument('--windowsize', type=int, nargs=2, default=[19, 19], help='Window size as two integers (e.g., 19 19)')
    parser.add_argument('--force_rewrite', type=int, default=0, help='Overwrite previous dataset if this flag is set')
    parser.add_argument('--list_vars', type=str, nargs='+', default=['U850', 'V850', 'T850', 'RH850', 'U950', 'V950', 'T950', 'RH950', 'U750', 'V750', 'T750', 'RH750', 'SLP750'],
                        help='List of variables with levels, formatted as VarLevel (e.g., "V950")')
    return parser.parse_args()
args = get_args()
workdir = args.workdir  # Take working directory from command-line argument
inputpath = os.path.join(workdir,'TC_domain')  # Take input path from command-line argument
windowsize = args.windowsize  # Take window size from command-line argument
force_rewrite = args.force_rewrite  # Overwrite previous dataset option
list_vars = args.list_vars
def split_var_level(list_vars):
    """
    Splits each element in a list of combined variable-level strings into separate components.

    Given a list of strings where each string represents a variable and a level concatenated together
    (e.g., "V950"), this function separates the alphabetic part (e.g., "V") from the numeric part 
    (e.g., 950) and returns them as tuples.

    Parameters:
    list_vars (list of str): A list of strings with combined variable and level information.
                             Example: ["U850", "V850", "T850"]

    Returns:
    list of tuples: A list of tuples, where each tuple contains the alphabetic variable as a string 
                    and the level as an integer.
                    Example: [('U', 850), ('V', 850), ('T', 850)]

    Example:
    >>> list_vars_input = ['U850', 'V850', 'T850', 'RH850', 'U950', 'V950', 'T950', 'RH950', 'U750', 'V750', 'T750', 'RH750', 'SLP750']
    >>> split_var_level(list_vars_input)
    [('U', 850), ('V', 850), ('T', 850), ('RH', 850), ('U', 950), ('V', 950), ('T', 950), ('RH', 950),
     ('U', 750), ('V', 750), ('T', 750), ('RH', 750), ('SLP', 750)]
    """
    result = []
    pattern = re.compile(r"([a-zA-Z]+)(\d+)")
    for item in list_vars:
        match = pattern.match(item)
        if match:
            # Extract the alphabetic part and convert the numeric part to an integer
            var = match.group(1)
            level = int(match.group(2))
            result.append((var, level))
    return result

list_vars = split_var_level(list_vars)
var_num = len(list_vars)
#####################################################################################
# DO NOT EDIT BELOW UNLESS YOU WANT TO MODIFY THE SCRIPT
#####################################################################################
def build_data_array(data, var_levels):
    arrays = []

    for var, lev in var_levels:
        try:
            # Attempt to select the variable at the specified level
            selected_data = data[var].sel(lev=lev)
        except KeyError as e:
            selected_data = data[var]  # Select without using 'lev'

        # Convert to numpy array (add a new axis if needed based on your data shape)
        numpy_data = np.array(selected_data)
        
        # Append the numpy array to the list
        arrays.append(numpy_data)

    # Concatenate all arrays along the first axis (adjust axis if necessary based on data shape)
    data_array_x = np.stack(arrays, axis = 0)
    
    return data_array_x
def convert_date_to_cyclic(date_str):
    """
    Convert a date in 'YYYYMMDD' format to a cyclic representation using sine and cosine.
    
    Args:
    date_str (str): Date string in 'YYYYMMDD' format.
    
    Returns:
    tuple: A tuple containing the sine and cosine representations of the day of the year.
    """
    # Parse the date string to datetime object
    date = datetime.strptime(date_str, "%Y%m%d")
    
    # Calculate the day of the year
    day_of_year = date.timetuple().tm_yday
    
    # Number of days in the year (handling leap years)
    days_in_year = 366 if date.year % 4 == 0 and (date.year % 100 != 0 or date.year % 400 == 0) else 365
    
    # Convert to cyclic
    sin_component = np.sin(2 * np.pi * day_of_year / days_in_year)
    cos_component = np.cos(2 * np.pi * day_of_year / days_in_year)
    
    return sin_component, cos_component

def cold_delete(filepath):
    try:
        os.remove(filepath)
        print(f"File {filepath} has been successfully removed.")

    except FileNotFoundError:
        print("The file does not exist.")
    except PermissionError:
        print("You do not have permission to remove the file.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_file_year_and_month(filename, id1):
    position = filename.find(id1)
    if position != -1:
        filedate = filename[position + len(id1): position + len(id1) + 8]
        #print(filedate)
        year = int(filedate[:4])
        month = int(filedate[4:6])
        return year, month

def check_date_within_range(date_str):
    # Convert string to date object
    date = datetime.strptime(date_str, '%Y%m%d')
    
    # Create start and end date objects for May 1st and November 30th of the same year
    start_date = datetime(year=date.year, month=5, day=1)
    end_date = datetime(year=date.year, month=11, day=30)
    
    # Check if the date falls within the range
    return start_date <= date <= end_date

def dumping_data(root='', outdir='', outname=['features', 'labels'],
                 regionize=True, omit_percent=5, windowsize=[18,18], cold_start=False):
    """
    Select and convert data from NetCDF files to NumPy arrays organized by year and months.

    Parameters:
    - root (str): The root directory containing NetCDF files.
    - outdir (str): The output directory for the NumPy arrays.
    - outname (list): List containing the output names for features and labels.
    - regionize (bool): If True, output data for each basin separately, 
                    with output files named outname{basin}.npy
    - omit_percent (float, percentage): Defines the upper limit of acceptable NaN (missing data) 
                    percentage in the 850mb band.
    - windowsize (list of floats): Specifies the size of the rectangular domain for Tropical Cyclone 
                    data extraction in degrees. The function selects a domain with dimensions closest 
                    to, but not smaller than, the specified window size.
    - cold_start (bool): If True, clears previous data files on start.

    Returns:
    None
    """
    i = 0
    omit = 0
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for filename in glob.iglob(root + '*/**/*.nc', recursive=True):
        id1 = f"{windowsize[0]}x{windowsize[1]}"
        if id1 in filename:
            position = filename.find(id1) + len(id1)
            filedate = filename[position:position + 8]  # Assumes the format is correct
            year = filedate[:4]  # Extract the year
            month = filedate[4:6]  # Extract the month
            year_dir = os.path.join(outdir, year)
            if not os.path.exists(year_dir):
                os.makedirs(year_dir)
        else:
            continue
        if cold_start and i == 0:
            # Clear previous data if cold start is enabled
            for fname in glob.glob(os.path.join(year_dir, '*.npy')):
                cold_delete(fname)

        data = xr.open_dataset(filename)
        # Data extraction and processing logic
        data_array_x = build_data_array(data, list_vars)
        if np.sum(np.isnan(data_array_x[0:4])) / 4 > omit_percent / 100 * math.prod(data_array_x[0].shape):
            i += 1
            omit += 1
            continue
        sin_day, cos_day = convert_date_to_cyclic(filedate)
        data_array_x = data_array_x.reshape([1, data_array_x.shape[0], data_array_x.shape[1], data_array_x.shape[2]])
        data_array_z = np.array([sin_day, cos_day, data.CLAT, data.CLON]) #day in year to sincos, central lat lon
        data_array_y = np.array([data.VMAX, data.PMIN, data.RMW])  # knots, mb, nmile
        data_array_z = data_array_z.reshape([1, data_array_z.shape[0]])
        data_array_y = data_array_y.reshape([1, data_array_y.shape[0]])
        # Further implementation as needed...
        # Reshape and store the data arrays
        with NpyAppendArray(os.path.join(year_dir, outname[0] + month + '.npy')) as npaax:
            npaax.append(data_array_x)
        with NpyAppendArray(os.path.join(year_dir, outname[1] + month + '.npy')) as npaay:
            npaay.append(data_array_y)
        try:
            with NpyAppendArray(os.path.join(year_dir, outname[2] + month + '.npy')) as npaaz:
                npaaz.append(data_array_z)
        except Exception as e:
            print(f"⚠️  Failed to write z for {filename}: {e}", flush=True)
        i += 1
        if i % 1000 == 0:
            print(f"{i} dataset processed.", flush=True)
            print(f"{omit} dataset omitted due to NaNs.", flush=True)

    print(f'Total {i} dataset processed.', flush=True)
    print(f'With {omit} dataset omitted due to NaNs.', flush=True)



# MAIN CALL:
if __name__ == "__main__":
    print('Initiating.', flush=True)
    print('Initiation completed.', flush=True)

    outputpath = os.path.join(workdir, 'Domain_data', f'exp_{var_num}features_{windowsize[0]}x{windowsize[1]}', 'data')
    os.makedirs(outputpath, exist_ok=True)

    if not os.path.exists(inputpath):
        print(f"Must have the input data from Step 1 by now....exit {inputpath}")
        exit()

    # Check if output directory is empty
    directory_empty = True  # Assume directory is empty until proven otherwise
    try:
        for entry in os.scandir(outputpath):
            if entry.is_file():
                print(f"Output directory '{outputpath}' is not empty. Data has been processed before.", flush=True)
                directory_empty = False
                break
    except Exception as e:
        print(f"Error checking directory contents: {str(e)}")
        exit()

    if not directory_empty:
        if force_rewrite:
            print('Force rewrite is True, rewriting the whole dataset.', flush=True)
        else:
            print('Will use the processed dataset, terminating this step.', flush=True)
            exit()

    outname = [
        f'features{var_num}_{windowsize[0]}x{windowsize[1]}',
        f'labels{var_num}_{windowsize[0]}x{windowsize[1]}',
        f'space_time_info{var_num}_{windowsize[0]}x{windowsize[1]}'
    ]

    # Function to dump data - Placeholder for your actual function call
    dumping_data(root=inputpath, outdir=outputpath, windowsize=windowsize, outname=outname, cold_start=force_rewrite)
