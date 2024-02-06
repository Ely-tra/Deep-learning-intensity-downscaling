import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from npy_append_array import NpyAppendArray

def dumping_data(root,outdir,outname=['CNNfeatures','CNNlabels']):
    i=0
    data_x=[]
    data_y=[]
    for filename in glob.iglob(root+'*/**/*.nc', recursive=True):
        print(filename)
        print(filename.find('EP'))
        i+=1
        if i==10:
            break
dumping_data('/N/slate/kmluong/TC_domain/', '/N/slate/kmluong/Training_data/', outname=['Testx','Testy'])
