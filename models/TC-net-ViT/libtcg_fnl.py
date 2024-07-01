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
# function to turn a path contain fnl_yyyymmdd_hh_mm to a cycle yyyymmddhh
#
def path2cycle(path):
    #file = path.lstrip("-")
    #file = path.rstrip("-")
    file = path.split('/')
    temp = file[-1].split('_')
    #print(file[-1])
    #print(temp)
    cycle = temp[1]+temp[2]
    #print(cycle)
    return cycle
#
# function to turn a cycle of the form yyyymmddhh to a path contain 
# fnl_yyyymmdd_hh_mm file
#
def cycle2path(rootdir,cycle):
    filename = "fnl_" + cycle[0:8] + "_" + cycle[8:10] + "_00.nc"
    path = rootdir + '/' + filename
    #print(path)
    return path

