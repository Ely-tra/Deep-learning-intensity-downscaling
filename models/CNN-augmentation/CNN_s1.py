print('Initiating.', flush=True)
import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from npy_append_array import NpyAppendArray
print('Initiation completed.', flush=True)
def dumping_data(root,outdir,outname=['CNNfeatures','CNNlabels']):
    i=0
    data_x=[]
    data_y=[]
    for filename in glob.iglob(root+'*/**/*.nc', recursive=True):
        data=xr.open_dataset(filename)
        data_array_x=np.array(data[['U','V', 'PS', 'T', 'H']].sel(lev=900).to_array())
        data_array_x=data_array_x.reshape([1,data_array_x.shape[0],
                                           data_array_x.shape[1],data_array_x.shape[2]])
        data_array_y=np.array([data.VMAX,data.PMIN, data.RMW]) #knot mb nmile
        data_array_y=data_array_y.reshape([1,data_array_y.shape[0]])
        with NpyAppendArray(outdir+outname[0]+filename[27:29]+'.npy', delete_if_exists=False) as npaax:
            npaax.append(data_array_x)
        with NpyAppendArray(outdir+outname[1]+filename[27:29]+'.npy', delete_if_exists=False) as npaay:
            npaay.append(data_array_y)
        i+=1
        if i%1000==0:
            print(str(i)+' dataset processed.', flush=True)
    print('Total ' + str(i)+ ' dataset processed.', flush=True)
dumping_data('/N/slate/kmluong/TC_domain/', '/N/slate/kmluong/Training_data/')
