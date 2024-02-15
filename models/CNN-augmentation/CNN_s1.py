print('Initiating.', flush=True)
import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from npy_append_array import NpyAppendArray
import math

print('Initiation completed.', flush=True)

def dumping_data(root, outdir, outname=['CNNfeatures', 'CNNlabels'], omit_percent=5):
    """
    Dump data from NetCDF files to NumPy arrays.

    Parameters:
    - root (str): The root directory containing NetCDF files.
    - outdir (str): The output directory for the NumPy arrays.
    - outname (list): List containing the output names for features and labels.

    Returns:
    None
    """
    i = 0
    omit=0
    for filename in glob.iglob(root + '*/**/*.nc', recursive=True):
        data = xr.open_dataset(filename)
        data_array_x = np.array(data[['U', 'V', 'T', 'RH', 'SLP']].sel(lev=850).to_array())
        if np.sum(np.isnan(data_array_x)) / 4 > omit_percent / 100 * math.prod(data_array_x[0].shape):
            i+=1
            #print(filename + ' omitted', flush=True)
            omit+=1
            if np.sum(np.isnan(data_array_x)) % 4 != 0:
                print('Oh gawd', flush=True)
                break
            if i % 1000 == 0:
                print(str(i) + ' dataset processed.', flush=True)
                break
            continue
        data_array_x = data_array_x.reshape([1, data_array_x.shape[0],
                                             data_array_x.shape[1], data_array_x.shape[2]])
        data_array_y = np.array([data.VMAX, data.PMIN, data.RMW])  # knots, mb, nmile
        data_array_y = data_array_y.reshape([1, data_array_y.shape[0]])

        with NpyAppendArray(outdir + outname[0] + filename[27:29] + '.npy', delete_if_exists=False) as npaax:
            npaax.append(data_array_x)

        with NpyAppendArray(outdir + outname[1] + filename[27:29] + '.npy', delete_if_exists=False) as npaay:
            npaay.append(data_array_y)

        i += 1
        if i % 1000 == 0:
            print(str(i) + ' dataset processed.', flush=True)
            print(str(omit) + ' dataset omitted due to NaNs.', flush = True)
            break
    print('Total ' + str(i) + ' dataset processed.', flush=True)

dumping_data('/N/slate/kmluong/TC_domain/', '/N/slate/kmluong/Training_data/')

