#
# DESCRIPTION: This script is to read a post-processed data in the NETCDF format centered
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
#
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
#
# Edit the input data path and parameters before running this script.
# Note that all output will be stored under the same exp name.
#
inputpath='/N/project/Typhoon-deep-learning/output/TC_domain/'
workdir='/N/project/Typhoon-deep-learning/output/'
windowsize=[25,25]      # domain size (degree) centered on TC center
var_num = 13            # number of channels for input
force_rewrite = True    # overwrite previous dataset option
print('Initiation completed.', flush=True)

#####################################################################################
# DO NOT EDIT BELOW UNLESS YOU WANT TO MODIFY THE SCRIPT
#####################################################################################
def dumping_data(root='', outdir='', outname=['features', 'labels'],
                 regionize=True, omit_percent=5, windowsize=[18,18], cold_start=False):
    """
    Select and convert data from NetCDF files to NumPy arrays.

    Parameters:
    - root (str): The root directory containing NetCDF files.
    - outdir (str): The output directory for the NumPy arrays.
    - outname (list): List containing the output names for features and labels.
    - regionize (bool): if True, output data for each basin separately, 
                    with output files named outname{basin}.npy
    - omit_percent (float, percentage): Defines the upper limit of acceptable NaN (missing data due 
                    to the nature of MERRA2) percentage in the 850mb band.
    - windowsize (list of floats): Specifies the size of the rectangular domain for Tropical Cyclone 
                    data extraction in degrees. The function selects a domain with dimensions closest 
                    to, but not smaller than, the specified window size.

    Returns:
    None
    """
    #
    # Counters for processed entries and skipped entries
    #
    i = 0
    omit = 0
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #
    # Loop through the processed domains with raw data.
    # Note: This script processes files with the domain size specified in the `windowsize` parameter.
    #       Domains should be pre-processed using the MERRA2TC_domain.py script to set the desired windowsize.
    #
    for filename in glob.iglob(root + '*/**/*.nc', recursive=True):
        if (str(windowsize[0])+'x'+str(windowsize[1])) in filename:
          pass
        else:
          continue
        data = xr.open_dataset(filename)
        #    
        # Choosing bands and level, data is taken from raw MERRA2 dataset, so the choice is not limited to atm level.
        #
        data_array_x = np.array(data[['U', 'V', 'T', 'RH']].sel(lev=850).to_array())
        data_array_x = np.append(data_array_x, np.array(data[['U', 'V', 'T', 'RH']].sel(lev=950).to_array()), axis=0)
        data_array_x = np.append(data_array_x, np.array(data[['U', 'V', 'T', 'RH', 'SLP']].sel(lev=750).to_array()), axis=0)
        #
        # Check for NaN percentage within the first level (which is 850mb), set up for a cold start
        #
        delete_if_exists = False
        if cold_start and i == 0:
            delete_if_exists = True
        else:
            delete_if_exists = False
        if np.sum(np.isnan(data_array_x[0:4])) / 4 > omit_percent / 100 * math.prod(data_array_x[0].shape):
            #print(data_array_x[0].shape, np.sum(np.isnan(data_array_x[0:4])), 
            #      np.sum(np.isnan(data_array_x[0][12:51,10:41])), flush=True)
            i+=1
            #print(filename + ' omitted', flush=True)
            omit+=1
            if np.sum(np.isnan(data_array_x[0:4])) % 4 != 0:
                #
                # Assuming all variables share the same NaN locations, this line ensures that.
                # This script is unusable if the line show up in the terminal.
                # 
                print('Oh no, am I cooked?', flush=True)            
                break
            if i % 1000 == 0:
                print(str(i) + ' dataset processed.', flush=True)
            continue
        #
        # Appending data directly to numpy savefile
        #
        data_array_x = data_array_x.reshape([1, data_array_x.shape[0],
                                             data_array_x.shape[1], data_array_x.shape[2]])
        data_array_y = np.array([data.VMAX, data.PMIN, data.RMW])  # knots, mb, nmile
        data_array_y = data_array_y.reshape([1, data_array_y.shape[0]])
        if regionize:
            addon = filename[len(root):len(root)+2]
        else:
            addon = ''
        with NpyAppendArray(outdir + outname[0] + addon + '.npy', delete_if_exists=delete_if_exists) as npaax:
            npaax.append(data_array_x)

        with NpyAppendArray(outdir + outname[1] + addon + '.npy', delete_if_exists=delete_if_exists) as npaay:
            npaay.append(data_array_y)

        i += 1
        if i % 1000 == 0:
            print(str(i) + ' dataset processed.', flush=True)
            print(str(omit) + ' dataset omitted due to NaNs.', flush = True)
    print('Total ' + str(i) + ' dataset processed.', flush=True)
    print('With ' + str(omit) + ' dataset omitted due to NaNs.', flush = True)
#
# MAIN CALL: 
#
outputpath = workdir+'/exp_'+str(var_num)+'features_'+str(windowsize[0])+'x'+str(windowsize[1])+'/'
if not os.path.exists(inputpath):
    print("Must have the input data from Step 1 by now....exit",inputpath)
    exit

second_check = False
try:
    for entry in os.scandir(outputpath):
        if entry.is_file():  # Check for any file entry
            print(f"Output directory '{outputpath}' is not empty. Data is processed before.", flush = True)
            second_check = True
            break  # Exit loop after finding a file
        else:
            second_check = False
            continue
except:
    second_check = False


if second_check:
    if force_rewrite:
        print('Force rewrite is True, rewriting the whole dataset.', flush = True)
    else:
        print('Will use the processed dataset, terminating this step.', flush = True)
        exit()    
outname=['CNNfeatures'+str(var_num)+'_'+str(windowsize[0])+'x'+str(windowsize[1]),
         'CNNlabels'+str(var_num)+'_'+str(windowsize[0])+'x'+str(windowsize[1])]
dumping_data(root=inputpath, outdir=outputpath, windowsize=windowsize, 
             outname=outname, regionize=False, cold_start = force_rewrite)
