#
# Collection of functions that reads in NETCDF datapath and return an numpy array 
# of shape [batch,nx,ny,nc]
#
import cv2
import os
import netCDF4
import numpy as np
import libtcg_fnl as tcg_fnl
import libtcg_utils as tcg_utils
#
# This function reads in a NETCDF file and return a 3-channel numpy array with dimension
# of [new_ny,new_nx,nc] = [ndepth,nwidth,nchannel] = [nrow, ncols, 3]
#
def create3channels(file,new_nx=32,new_ny=16,number_channels=3):
    f = netCDF4.Dataset(file)
    nx = f.dimensions['lon'].size
    ny = f.dimensions['lat'].size
    nz = f.dimensions['lev'].size
    a2 = np.zeros((ny,nx,number_channels))

    tmp = f.variables['tmpprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,0] = tmp[3,j,i]    # temperature at 900 mb
    ugr = f.variables['ugrdprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,1] = ugr[3,j,i]    # u-wind at 900 mb
    vgr = f.variables['vgrdprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,2] = vgr[3,j,i]    # v-wind at 900 mb
    new_array = cv2.resize(a2, (new_nx, new_ny)) #cv2 resizes based on (width,height)
    return new_array
#
# This function reads in a NETCDF file and return an 12-channel numpy array with dimension
# of [new_ny,new_nx,nc] = [ndepth,nwidth,nchannel] = [nrow, ncols, 12]
#
def create12channels(file,new_nx=32,new_ny=16,number_channels=12):
    f = netCDF4.Dataset(file)
    nx = f.dimensions['lon'].size
    ny = f.dimensions['lat'].size
    nz = f.dimensions['lev'].size
    a2 = np.zeros((ny,nx,number_channels))
    #print("libtcg_netcdfloader.create12channels", file,nx,ny,nz)

    abv = f.variables['absvprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,0] = abv[1,j,i]    # abs vort at 950 mb
    rel = f.variables['rhprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,1] = rel[7,j,i]    # RH at 750 mb
    sfc = f.variables['pressfc']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,2] = sfc[j,i]      # surface pressure
    tmp = f.variables['tmpprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,3] = tmp[15,j,i]   # temperature at 400 mb
    tsf = f.variables['tmpsfc']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,4] = tsf[j,i]      # surface temperature
    ugr = f.variables['ugrdprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,5] = ugr[3,j,i]    # u-wind at 900 mb
            a2[j,i,6] = ugr[17,j,i]   # u-wind at 300 mb
    vgr = f.variables['vgrdprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,7] = vgr[3,j,i]    # v-wind at 900 mb
            a2[j,i,8] = vgr[17,j,i]   # v-wind at 300 mb
    hgt = f.variables['hgtprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,9] = hgt[3,j,i]    # geopotential at 850 mb
    wgr = f.variables['vvelprs']
    for i in range(nx):
        for j in range(ny):
            a2[j,i,10] = wgr[3,j,i]   # w-wind at 900 mb
            a2[j,i,11] = wgr[17,j,i]  # w-wind at 300 mb
    new_array = cv2.resize(a2, (new_nx, new_ny)) #cv2 resizes based on (width,height)
    #print("libtcg_netcdfloader.create12channels returns shape:",new_array.shape)
    return new_array
#
# This function returns a 12-channel data frame in terms of a numpy list with dim 
# of (nframe,nx,ny,nc), using cycle as the present time and cycle - 1*interval
# cycle - 2*interval,...,cycle - (frame-1)*interval as the lag times
#
def frame12channels(rootdir,cycle,interval=-6,nx=64,ny=16,number_channels=12,nframe=9):
    file = tcg_fnl.cycle2path(rootdir,cycle)
    data = np.zeros((nframe,ny,nx,number_channels)) 
    ncycle = cycle
    for i in range(nframe):
        print("---> libtcg_netcdfloader.frame12channels reads cycle", cycle,file) 
        tmp = create12channels(file,new_nx=nx,new_ny=ny,number_channels=number_channels)
        #print("libtcg_netcdfloader.frame12channels returns temp shape",tmp.shape,ncycle,interval)
        data[i,:,:,:] = tmp[:,:,:]
        ncycle = tcg_utils.add_hour(ncycle,interval)
        #print("libtcg_netcdfloader.frame12channels updates cycles",ncycle)
        file = tcg_fnl.cycle2path(rootdir,ncycle)   
        #print("libtcg_netcdfloader.frame12channels updates cycle file",file)
    return data   
#
# This function returns a 3-channel data frame in terms of a numpy list with dim
# of (nframe,nx,ny,nc), using cycle as the present time and cycle - 1*interval
# cycle - 2*interval,...,cycle - (frame-1)*interval as the lag times
#
def frame3channels(rootdir,cycle,interval=-6,nx=64,ny=16,number_channels=3,nframe=9):
    file = tcg_fnl.cycle2path(rootdir,cycle)
    data = np.zeros((nframe,ny,nx,number_channels))
    ncycle = cycle
    for i in range(nframe):
        print("---> libtcg_netcdfloader.frame12channels reads cycle", cycle,file)
        tmp = create3channels(file,new_nx=nx,new_ny=ny,number_channels=number_channels)
        #print("libtcg_netcdfloader.frame12channels returns temp shape",tmp.shape,ncycle,interval)
        data[i,:,:,:] = tmp[:,:,:]
        ncycle = tcg_utils.add_hour(ncycle,interval)
        #print("libtcg_netcdfloader.frame12channels updates cycles",ncycle)
        file = tcg_fnl.cycle2path(rootdir,ncycle)
        #print("libtcg_netcdfloader.frame12channels updates cycle file",file)
    return data
#
# This function reads in a NETCDF file and return an numpy array in full dimension
# of the size [nx,ny,nc] This is an old function and should be replaced by 
# create12channels as the latat allows for nx /= ny.
#
def read12channels(file,IMG_SIZE,number_channels=12):
    f = netCDF4.Dataset(file)
    abv = f.variables['absvprs']
    nx = np.size(abv[0,0,:])
    ny = np.size(abv[0,:,0])
    nz = np.size(abv[:,0,0])
    a2 = np.zeros((nx,ny,number_channels))
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,0] = abv[1,j,i]    # abs vort at 950 mb
    rel = f.variables['rhprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,1] = rel[7,j,i]    # RH at 750 mb
    sfc = f.variables['pressfc']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,2] = sfc[j,i]      # surface pressure
    tmp = f.variables['tmpprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,3] = tmp[15,j,i]   # temperature at 400 mb
    tsf = f.variables['tmpsfc']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,4] = tsf[j,i]      # surface temperature
    ugr = f.variables['ugrdprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,5] = ugr[3,j,i]    # u-wind at 900 mb
            a2[i,j,6] = ugr[17,j,i]   # u-wind at 300 mb
    vgr = f.variables['vgrdprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,7] = vgr[3,j,i]    # v-wind at 900 mb
            a2[i,j,8] = vgr[17,j,i]   # v-wind at 300 mb
    hgt = f.variables['hgtprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,9] = hgt[3,j,i]    # geopotential at 850 mb
    wgr = f.variables['vvelprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j,10] = wgr[3,j,i]   # w-wind at 900 mb
            a2[i,j,11] = wgr[17,j,i]  # w-wind at 300 mb
    new_array = cv2.resize(a2, (IMG_SIZE, IMG_SIZE))
    return new_array
#
# This function returns 2 outputs: 1) out_array is a non-dimensional numpy array
# of the form [batch,nx,ny,nc], and 2) output of the same shape but with full
# physics dimension.
#
def prepare12channels(filepath,IMG_SIZE=30):
    number_channels = 12
    new_array = read12channels(filepath,IMG_SIZE,number_channels)
    dim_array = new_array.copy()
    #
    # normalize the data
    #
    #print('Number of channels to normalize is: ',number_channels)
    for var in range(number_channels):
        maxvalue = new_array[:,:,var].flat[np.abs(new_array[:,:,var]).argmax()]
        #print('Normalization factor for channel',var,', is: ',abs(maxvalue))
        new_array[:,:,var] = new_array[:,:,var]/abs(maxvalue)
    out_array = np.reshape(new_array, (-1, IMG_SIZE, IMG_SIZE, number_channels))
    dim_array = np.reshape(dim_array, (-1, IMG_SIZE, IMG_SIZE, number_channels))
    #print('reshape new_array returns: ',out_array.shape)
    return out_array, dim_array

