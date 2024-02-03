import xarray as xr
import netCDF4 as nc
import os
count=0
windows_path = "/N/slate/kmluong/TC_domain"
oldlatdim=0
oldlondim=0
oldfile=0
err=0
print('Ini done', flush=True)
for folder in os.listdir(windows_path):
 print(folder)
 for folder2 in os.listdir(windows_path+'/'+folder):
  for file in os.listdir(windows_path+'/'+folder+'/'+folder2):
   filepath=os.path.join(windows_path+'/'+folder+'/'+folder2+'/'+file)
   dataset=xr.open_dataset(filepath)
   latdim=len(dataset.lat)
   londim=len(dataset.lon)
   if count >0:
    if latdim!=oldlatdim:
     print('Oh no, this is lat', flush=True)
     print(oldfile,filepath, oldlatdim, latdim)
     err+=1
     break
    if londim!=oldlondim:
     err+=1
     print('Oh no, this is lon', flush=True)
     print(oldfile, filepath, oldlondim, londim)
     break
   count+=1
   if count<11:
    print(count)
   oldfile=filepath
   oldlatdim=latdim
   oldlondim=londim
   if count%1000==0:
    print(str(count)+' checked, total '+str(err)+' error.', flush=True)
print('finish checking, total '+ str(err)+' error.')

