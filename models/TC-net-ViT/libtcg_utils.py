import numpy as np
import datetime
import time
import requests
import wget
import os
import pandas as pd
import csv
import sys
#
# build an F1-score function for later use
#
def F1_score(y_true,y_prediction,true_class,true_threshold):
    T = len(y_true)
    if len(y_prediction) != T:
        print("Prediction and true label arrays have different size. Stop")
        return 0, 0, 0
    P = 0
    TP = 0 
    FN = 0
    TN = 0
    FP = 0
    for i in range(T):
        if y_true[i] == true_class:
            P = P + 1       
            if y_prediction[i] >= true_threshold:
                TP += 1 
            else:
                FN += 1
        else:
            if y_prediction[i] >= true_threshold:
                FP += 1 
            else:
                TN += 1            
    N = T - P  
    Recall = 0  
    if TP == 0 and FP == 0 and FN == 0:
       F1 = 0
    else:
       F1 = 2.*TP/(2.*TP + FP + FN)
       Recall = TP/float(TP+FN)
    if TP == 0 and FP == 0: 
        Precision = 0.
    else:    
        Precision = TP/float(TP+FP)
    return F1, Recall, Precision
#
# function to add n hours to a cycle yyyymmddhh
#
def add_hour(yyyymmddhh,interval=3):
    yyyy = yyyymmddhh[:4]
    mm=yyyymmddhh[4:6]
    dd=yyyymmddhh[6:8]
    hh=yyyymmddhh[-2:]
    a = datetime.datetime(int(yyyy), int(mm), int(dd), int(hh),0,0,0)
    ts = datetime.datetime.timestamp(a)
    b = datetime.datetime.fromtimestamp(ts+interval*3600)
    if b.day < 10:
        new_day = "0" + str(b.day)
    else:
        new_day = str(b.day)
    if b.month < 10:
        new_month = "0" + str(b.month)
    else:
        new_month = str(b.month)
    if b.hour < 10:
        new_hour = "0" + str(b.hour)
    else:
        new_hour = str(b.hour)
    yyyymmddhh_updated = str(b.year) + new_month + new_day + new_hour
    #print("libtcg_utils.add_hour:",yyyymmddhh,interval,yyyymmddhh_updated)
    return yyyymmddhh_updated
#
# normalize the data by the max values
#
def normalize_channels(X,y):
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            #print('Normalization factor for sample and channel',i,var,', is: ',abs(maxvalue))
            X[i,:,:,var] = X[i,:,:,var]/abs(maxvalue)
            maxnew = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            #print('-->After normalization of sample and channel',i,var,', is: ',abs(maxnew))
            #input('Enter to continue...')
    print("Finish normalization...")
    return X,y

#
# normalize the data by the max values for all data frame at each channels
#
def normalize_frame_data(X):
    nsample = X.shape[0]
    number_channels = X.shape[4]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i,:,:,:,var].flat[np.abs(X[i,:,:,:,var]).argmax()]
            #print('----> lib_utils: Normalization factor for sample and channel',i,var,', is: ',abs(maxvalue))
            X[i,:,:,:,var] = X[i,:,:,:,var]/abs(maxvalue)
            maxnew = X[i,:,:,:,var].flat[np.abs(X[i,:,:,:,var]).argmax()]
            #print('----> lib_utils: After normalization of sample and channel',i,var,', is: ',abs(maxnew))
            #input('Enter to continue...')
    print("Finish normalization...")
    return X
#
# return the max value of data with a given sample and channel numbers
#
def maxval_framedata(X,ic,ib):
    maxval = X[ib,:,:,:,ic].flat[np.abs(X[ib,:,:,:,ic]).argmax()]
    return maxval 
