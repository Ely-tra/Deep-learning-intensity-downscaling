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

    Returns:
    None
    """
    i = 0
    omit=0
    for filename in glob.iglob(root + '*/**/*.nc', recursive=True):
        if (str(windowsize[0])+'x'+str(windowsize[1])) in filename:
          pass
        else:
          continue
        data = xr.open_dataset(filename)
        
        data_array_x = np.array(data[['U', 'V', 'T', 'RH']].sel(lev=850).to_array())
        data_array_x = np.array(data[['U', 'V', 'T', 'RH']].sel(lev=950).to_array())
        data_array_x = np.append(data_array_x, np.array(data[['U', 'V', 'T', 'RH', 'SLP']].sel(lev=750).to_array()), axis=0)
        if np.sum(np.isnan(data_array_x[0:4])) / 4 > omit_percent / 100 * math.prod(data_array_x[0].shape):
            print(data_array_x[0].shape, np.sum(np.isnan(data_array_x[0:4])), np.sum(np.isnan(data_array_x[0][12:51,10:41])), flush=True)
            i+=1
            #print(filename + ' omitted', flush=True)
            omit+=1
            if np.sum(np.isnan(data_array_x[0:4])) % 4 != 0:
                print('Oh gawd', flush=True)
                break
            if i % 1000 == 0:
                print(str(i) + ' dataset processed.', flush=True)
            continue
        data_array_x = data_array_x.reshape([1, data_array_x.shape[0],
                                             data_array_x.shape[1], data_array_x.shape[2]])
        data_array_y = np.array([data.VMAX, data.PMIN, data.RMW])  # knots, mb, nmile
        data_array_y = data_array_y.reshape([1, data_array_y.shape[0]])
        if regionize==True:
          addon=filename[27:29]
        else:
          addon=''

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
dumping_data('/N/slate/kmluong/TC_domain/', '/N/slate/kmluong/Training_data/', windowsize=[30,30], regionize=False)

