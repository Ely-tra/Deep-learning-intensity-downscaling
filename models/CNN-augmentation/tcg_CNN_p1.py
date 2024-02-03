#
# NOTE: This machine learning program is for predicting TC formation, using
#       input dataset in the NETCDF format. The program treats different 
#       2D input fields as different channels of an image. This specific 
#       program requires a set of 12 2D-variables (12-channel image) and 
#       consists of three stages
#       - Stage 1: reading NETCDF input and generating (X,y) data with a 
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,y) pair and build a CNN model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_CNN.model.
#       - Stage 3: import the trained model from Stage 2, and make a list
#                  of prediction from normalized test data. 
#
# INPUT: This Stage 1 script requires two specific input datasets, including
#       1. 7 meterological vars u, v,abs vort, tmp, RH, vvels, sst, cape  
#          corresponding to negative cases (i.e. no TC formation within the 
#          domain). 
#       2. Similar data but for positive cases (i.e., there is a TC centered
#          on the domain)  
#        Remarks: Note that these data must be on the standard 19 vertical
#        levels 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 
#        550, 500, 450, 400, 350, 300, 250, 200. Also, all field vars must
#        be resize to cover an area of 30x30 around the TC center for the 
#        positive data cases.
#
# OUTPUT: A set of pairs (X,y) needed for CNN training
#
# HIST: - 25, Oct 22: Created by CK
#       - 27, Oct 22: Added a classification loop to simplify the code
#       - 01, Nov 22: Modified to include more channels  
#       - 02, Feb 23: Revised for jupiter-notebook workflow
#       - 20, Jun 23: Updated for augmentation/dropout layers
#       - 11, Oct 23: revised for a better workflow for future upgrades
#                     and sharing
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu) 
#
#==========================================================================
import netCDF4
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import random
import pickle
import sys
import libtcg_netcdfloader as tcg_loader
import libtcg_utils as tcg_utils
#
# This function returns a kist composing of [numpy data, label] that reads from the
# set of NETCDF data under netcdf data path
#
def main(datapath,IMG_SIZE=30):
    tcg_class = ['pos','neg']
    array_raw = []
    for tcg in tcg_class:
        if tcg == "pos":
            datadir=datapath + 'pos'
        else:
            datadir=datapath + 'neg'
        print('Input data dir is: ',datadir)
        for img in tqdm(os.listdir(datadir)):
            try:
                file=datadir+'/'+img
                print('Processing file:', file)
                a3 = tcg_loader.read12channels(file,IMG_SIZE=IMG_SIZE,number_channels=12)
                print('Data shape is :',a3.shape)
                #input('Enter to continue...')
                if tcg == "pos":
                    array_raw.append([a3, 1])
                else:
                    array_raw.append([a3, 0])
            except Exception as e:
                pass
    return array_raw
#
# This function reads in a list of 4 dim and plot a random field for quick check
#
def check_visual(array_raw,plot_sample=1):
    print("Plotting one example from raw data input")
    temp = np.array(array_raw[plot_sample][0])
    plt.figure(figsize=(11, 3))
    plt.subplot(1,3,1)
    CS = plt.contour(temp[:,:,4])
    plt.clabel(CS, inline=True, fontsize=10)
    plt.title('SST')
    plt.grid()

    plt.subplot(1,3,2)
    CS = plt.contour(temp[:,:,1])
    plt.clabel(CS, inline=True, fontsize=10)
    plt.title('RH')
    plt.grid()

    plt.subplot(1,3,3)
    CS = plt.contour(temp[:,:,9])
    plt.clabel(CS, inline=True, fontsize=10)
    plt.title('850 mb geopotential')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    IMG_SIZE = 30
    datapath="/N/project/hurricane-deep-learning/data/ncep_extracted_binary_30x30/ncep_WP_binary_0h/"
    n = len(sys.argv)
    print("Total arguments input are:", n)
    print("Name of Python script:", sys.argv[0])
    if n < 2:
       print("Need an input data path argument for the year to process...Stop")
       print("+ Example: tcg_CNN_p1.py /N/project/hurricane-deep-learning/data/ncep_extracted_binary_30x30/ncep_WP_binary_0h/")
       exit()
    datapath = str(sys.argv[1])
    print("Input data path to run is: ",datapath)
    #sys.exit()

    array_raw = main(datapath=datapath,IMG_SIZE=IMG_SIZE)
    print("Raw input data shape (nsample,ny,nx,nchannel) is: ",len(array_raw),
      len(array_raw[0][0]),len(array_raw[0][0][0]),len(array_raw[0][0][0][0]))
    #
    # visualize a few variables for checking the input data
    #
    check_visualization = "no"
    if check_visualization== "yes":
        check_visual(array_raw,plot_sample=7)
    #
    # randomize data and generate training data (X,y)
    #
    np.random.shuffle(array_raw)
    X = []
    y = []
    for features,label in array_raw: 
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 12)
    if check_visualization== "yes":
        print(X.shape)
        print(y)
    #
    # save training data to an output for subsequent use
    #
    pickle_out = open("tcg_CNNaugment_X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open("tcg_CNNaugment_y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
