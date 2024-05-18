#
# SCRIPT NAME: NetCDF Data Extraction and Preprocessing
#
# DESCRIPTION: This script initializes necessary libraries for data handling and defines a 
#              function to process atmospheric data from NetCDF files, structuring it into 
#              NumPy arrays suitable for analysis and model input. It ensures efficient management 
#              of large datasets with an emphasis on maintaining data integrity through robust 
#              error handling and validation steps. The function supports custom configurations 
#              for selective data extraction and transformation, tailored for climatological research.
#
# DEPENDENCIES:
#   - xarray: Utilized for data manipulation and accessing NetCDF file content.
#   - matplotlib: Incorporated for plotting capabilities (utilized upon extension of the script).
#   - numpy: Essential for numerical computations and data structuring.
#   - glob: Facilitates file path handling for navigating directory structures.
#   - npy_append_array: Employs efficient append operations to NumPy array files, optimizing memory usage.
#   - math: Provides support for basic mathematical operations necessary for data processing.
#
# USAGE: Execute the function 'dumping_data' with specified directory paths for input NetCDF files and 
#        output directories. Additional parameters allow customization of processing characteristics
#        such as window size and data omission thresholds. 
#
# NOTE: The initial layers of the output array contain 850mb level atmospheric data from the MERRA2 dataset,
#       which also determines the NaN threshold. For modifications or additional data layers, see line 79.
#
# NOTE: This script does not process the domain size directly, it rather extracts desired data from domains 
#       processed by MERRA2TC_domain.py script, with the corresponding domain size. 
#
# AUTHOR: Minh Khanh Luong @ Indiana University Bloomington
# EMAIL: kmluong@iu.edu
# CREATED DATE: May 14, 2024
#
#==============================================================================================

print('Initiating.', flush=True)
import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from npy_append_array import NpyAppendArray
import math

print('Initiation completed.', flush=True)

def dumping_data(root, outdir, outname=['CNNfeatures13.30x30test', 'CNNlabels13.30x30test'],regionize=True, omit_percent=5, windowsize=[18,18]):
    """
    Dump data from NetCDF files to NumPy arrays.

    Parameters:
    - root (str): The root directory containing NetCDF files.
    - outdir (str): The output directory for the NumPy arrays.
    - outname (list): List containing the output names for features and labels.
    - regionize (bool): if True, output data for each basin separately, with output files named outname{basin}.npy
    - omit_percent (float, percentage): Defines the upper limit of acceptable NaN (missing data due to the nature of MERRA2) percentage in the 850mb band.
    - windowsize (list of floats): Specifies the size of the rectangular domain for Tropical Cyclone data extraction in degrees.
                                   The function selects a domain with dimensions closest to, but not smaller than, the specified window size.

    Returns:
    None
    """

    # Counters for processed entries and skipped entries

    i = 0
    omit=0

    # Loop through the processed domains with raw data.
    # Note: This script processes files with the domain size specified in the `windowsize` parameter.
    #       Domains should be pre-processed using the MERRA2TC_domain.py script to set the desired windowsize.

    for filename in glob.iglob(root + '*/**/*.nc', recursive=True):
        if (str(windowsize[0])+'x'+str(windowsize[1])) in filename:
          pass
        else:
          continue
        data = xr.open_dataset(filename)
        
        # Choosing bands and level, data is taken from raw MERRA2 dataset, so the choice is not limited to atm level.

        data_array_x = np.array(data[['U', 'V', 'T', 'RH']].sel(lev=850).to_array())
        data_array_x = np.array(data[['U', 'V', 'T', 'RH']].sel(lev=950).to_array())
        data_array_x = np.append(data_array_x, np.array(data[['U', 'V', 'T', 'RH', 'SLP']].sel(lev=750).to_array()), axis=0)

        # Check for NaN percentage within the first level (which is 850mb)

        if np.sum(np.isnan(data_array_x[0:4])) / 4 > omit_percent / 100 * math.prod(data_array_x[0].shape):
            #print(data_array_x[0].shape, np.sum(np.isnan(data_array_x[0:4])), np.sum(np.isnan(data_array_x[0][12:51,10:41])), flush=True)
            i+=1
            #print(filename + ' omitted', flush=True)
            omit+=1
            if np.sum(np.isnan(data_array_x[0:4])) % 4 != 0:
                print('Oh no, am I cooked?', flush=True)            # This script works by assuming all variables share the same NaN locations, this line ensures that.

                # This script is unusable if the line show up in the terminal.

                break
            if i % 1000 == 0:
                print(str(i) + ' dataset processed.', flush=True)
            continue

        # Appending data directly to numpy savefile
        data_array_x = data_array_x.reshape([1, data_array_x.shape[0],
                                             data_array_x.shape[1], data_array_x.shape[2]])
        data_array_y = np.array([data.VMAX, data.PMIN, data.RMW])  # knots, mb, nmile
        data_array_y = data_array_y.reshape([1, data_array_y.shape[0]])
        if regionize:
          addon=filename[len(root):len(root)+2]
        else:
          addon=''
        print(addon)
        with NpyAppendArray(outdir + outname[0] + addon + '.npy', delete_if_exists=False) as npaax:
            npaax.append(data_array_x)

        with NpyAppendArray(outdir + outname[1] + addon + '.npy', delete_if_exists=False) as npaay:
            npaay.append(data_array_y)

        i += 1
        if i % 1000 == 0:
            print(str(i) + ' dataset processed.', flush=True)
            print(str(omit) + ' dataset omitted due to NaNs.', flush = True)
    print('Total ' + str(i) + ' dataset processed.', flush=True)
    print('With ' + str(omit) + ' dataset omitted due to NaNs.', flush = True)
dumping_data('/N/slate/kmluong/TC_domain/', '/N/slate/kmluong/Training_data/', windowsize=[30,30], regionize=True)
