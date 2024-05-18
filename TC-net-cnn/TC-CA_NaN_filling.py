#
# SCRIPT NAME: Context-Aware NaN Filling for Atmospheric Data Arrays
#
# DESCRIPTION: This script initializes essential libraries and defines functions designed for the
#              sophisticated handling of atmospheric data. The primary functionality focuses on 
#              context-aware filling of NaN values in multidimensional arrays, where the filling 
#              strategy is guided by the surrounding wind field characteristics. This method uses 
#              calculations of vector fields, element-wise operations, and directional weightings to 
#              enhance the data filling accuracy. The script processes files within a specified directory,
#              applies the context-aware filling methods, and saves the modified data back to disk.
#
# FUNCTIONS:
#   - calfield: Calculates normalized vector fields from 2D arrays.
#   - elewise_dot: Computes element-wise dot products between two vectors.
#   - weight_field: Calculates weights for vector fields based on predefined directions.
#   - shift: Shifts array elements along specified axis and fills shifted positions with zeros.
#   - extract_bound: Identifies boundary elements in 2D arrays adjacent to NaN values.
#   - fill4, fill3, fill2: Fill NaN values using 4-cell, 3-cell, and 2-cell patterns, considering
#                          vector fields to maintain spatial coherence in wind data.
#   - fill_nan: Coordinates the sequence of filling functions to ensure comprehensive coverage of NaNs.
#   - fix_data: Applies NaN filling operations to data files, ensuring data consistency and reliability.
#
# USAGE: Adjust the root directory to point to your data files and run the script. It will automatically
#        find and process files that require NaN filling and save the corrected files.
#
# NOTE: The script fills NaN values by evaluating the wind field around each missing point within a 3x3 pixel 
#       area. The filling method assigns weights to neighboring pixels based on how closely their wind vectors 
#       align towards or away from the center pixel with missing data. This alignment is quantified using the 
#       dot product between wind vectors and predefined directional arrays, where a higher dot product indicates 
#       stronger alignment and results in a higher weight. By leveraging this directional weighting, the script 
#       ensures that filled values are consistent with the local wind patterns, enhancing the data's accuracy and relevance.
#
# NOTE: The algorithm is not completed, and should not be used for dataset with > 5% missing data. One should find a way
#       to handle large border missing patterns or any big rip near the center of the domain. Also loop filling is not
#       efficient, see line 225.
#
# AUTHOR: Minh Khanh Luong
# CREATED DATE: May 14 2024
#
#==============================================================================================

print('Initializing')
import os
import numpy as np
import copy
import glob

np.seterr(invalid='ignore')

#==============================================================================================
# Defining mathematical base functions
#==============================================================================================

def calfield(array):
    """
    Calculate the normalized vector field from a given array.

    Parameters:
    - array: 1D array with two elements representing x and y components.

    Returns:
    - vector: 1D array representing the normalized vector field.
    """
    uu = np.sqrt((array[0]**2) / (array[0]**2 + array[1]**2))
    vv = np.sqrt((array[1]**2) / (array[0]**2 + array[1]**2))
    vector = np.stack((uu, vv), axis=-1)
    return vector

def elewise_dot(vector1, vector2):
    """
    Calculate element-wise dot product between two vectors.

    Parameters:
    - vector1: 3D array representing the first vector.
    - vector2: 3D array representing the second vector.

    Returns:
    - result: 2D array representing the element-wise dot product.
    """
    return vector2[:, :, 0] * vector1[:, :, 0] + vector2[:, :, 1] * vector1[:, :, 1]

#==============================================================================================
# Filling algorithm function
#==============================================================================================

def weight_field(vector):
    """
    Calculate weights for a given vector field.

    Parameters:
    - vector: 3D array representing the vector field.

    Returns:
    - weight: 3D array representing the calculated weights.
    """
    C2 = np.sqrt(2) / 2
    direction_array = np.array([[[C2, -C2], [0, -1], [-C2, -C2]],
                                [[1, 0], [0, 0], [-1, 0]],
                                [[C2, C2], [0, 1], [-C2, C2]]])
    weight = np.abs(elewise_dot(vector, direction_array))
    weight = weight / np.nansum(weight)
    return weight

def shift(array, place, mode=0):
    """
    Shift the elements of a 2D array along the specified axis.

    Parameters:
    - array: 2D array to be shifted.
    - place: Number of positions to shift. Positive values shift to the right/down, negative to the left/up.
    - mode: Axis along which the shift is performed (0 for rows, 1 for columns).

    Returns:
    - new_arr: Shifted 2D array.
    """
    new_arr = np.roll(array, place, axis=mode)
    if place > 0:
        if mode == 0:
            new_arr[:place] = np.zeros((new_arr[:place].shape))
        else:
            new_arr[:, :place] = np.zeros((new_arr[:, :place].shape))
        return new_arr
    else:
        if mode == 0:
            new_arr[place:] = np.zeros((new_arr[place:].shape))
        else:
            new_arr[:, place:] = np.zeros((new_arr[:, place:].shape))
        return new_arr

def extract_bound(array):
    """
    Extract the boundary of a 2D array containing NaN values.

    Parameters:
    - array: 2D array containing NaN values.

    Returns:
    - bound: 2D boolean array representing the boundary.
    """
    nan = np.isnan(array)
    notnan = np.logical_not(nan)
    s1 = np.logical_and(notnan, shift(nan, 1))
    s2 = np.logical_and(notnan, shift(nan, -1))
    s3 = np.logical_and(notnan, shift(nan, 1, mode=1))
    s4 = np.logical_and(notnan, shift(nan, -1, mode=1))
    bound = np.logical_or(s1, np.logical_or(s2, np.logical_or(s3, s4)))
    return bound
def fill4(array):
    """
    Fills NaN values in the input array with a 4-cell pattern.

    Parameters:
    - array: numpy array

    Returns:
    - numpy array
    """
    bound = extract_bound(array[0])
    nan4 = np.zeros(bound.shape)
    nan4[1:-1, 1:-1] = nan4[1:-1, 1:-1] + bound[1:-1, :-2] + bound[1:-1, 2:] + bound[:-2, 1:-1] + bound[2:, 1:-1]
    nan4 = np.logical_and(nan4 == 4, np.isnan(array[0]))

    if np.sum(nan4) == 0:
        return array

    for i in np.transpose(nan4.nonzero()):
        vector = calfield(array)
        weight = weight_field(vector[i[0] - 1:i[0] + 2, i[1] - 1:i[1] + 2])

        for j in range(len(array) - 1):
            array[j, i[0], i[1]] = np.nansum(array[j, i[0] - 1:i[0] + 2, i[1] - 1:i[1] + 2] * weight)

    return array

def fill3(array):
    """
    Fills NaN values in the input array with a 3-cell pattern.

    Parameters:
    - array: numpy array

    Returns:
    - numpy array
    """
    bound = extract_bound(array[0])
    nan3 = np.zeros(bound.shape)
    nan3[1:-1, 1:-1] = nan3[1:-1, 1:-1] + bound[1:-1, :-2] + bound[1:-1, 2:] + bound[:-2, 1:-1] + bound[2:, 1:-1]
    nan3 = np.logical_and(nan3 == 3, np.isnan(array[0]))

    if np.sum(nan3) == 0:
        return array

    for i in np.transpose(nan3.nonzero()):
        vector = calfield(array)
        weight = weight_field(vector[i[0] - 1:i[0] + 2, i[1] - 1:i[1] + 2])

        for j in range(len(array) - 1):
            array[j, i[0], i[1]] = np.nansum(array[j, i[0] - 1:i[0] + 2, i[1] - 1:i[1] + 2] * weight)

    return array

def fill2(array):
    """
    Fills NaN values in the input array with a 2-cell pattern.

    Parameters:
    - array: numpy array

    Returns:
    - numpy array
    """
    bound = extract_bound(array[0])
    nan2 = np.zeros(bound.shape)
    nan2[1:-1, 1:-1] = nan2[1:-1, 1:-1] + bound[1:-1, :-2] + bound[1:-1, 2:] + bound[:-2, 1:-1] + bound[2:, 1:-1]
    nan2 = np.logical_and(nan2 == 2, np.isnan(array[0]))

    if np.sum(nan2) == 0:
        return array

    for i in np.transpose(nan2.nonzero()):
        vector = calfield(array)
        weight = weight_field(vector[i[0] - 1:i[0] + 2, i[1] - 1:i[1] + 2])

        for j in range(len(array) - 1):
            array[j, i[0], i[1]] = np.nansum(array[j, i[0] - 1:i[0] + 2, i[1] - 1:i[1] + 2] * weight)

    return array
def fill_nan(array):
    """
    Fills NaN values in the input array using a sequence of fill functions.

    Parameters:
    - array: numpy array

    Returns:
    - numpy array
    """
    hold1=0
    while np.sum(np.isnan(array[0, 1:-1, 1:-1])) > 0:
        array = fill4(array)
        array = fill3(array)
        array = fill2(array)
        hold1+=1
        if hold1==300:
            break
    array = np.pad(array, [[0, 0], [1, 1], [1, 1]])

    while np.sum(np.isnan(array[0, 1:-1, 1:-1])) > 0:
        array = fill4(array)
        array = fill3(array)
        array = fill2(array)

    array = array[:, 1:-1, 1:-1]
    return array
def fix_data(file):
    """
    Fixes NaN values in the data stored in the given file using the fill_nan function.

    Parameters:
    - file: str
        The file path for the data.

    Returns:
    - None
    """
    xa = np.load(file)
    #print(xa.shape[1],flush=True)
    if xa.shape[1] == 5:
        fillmode = 0
    else:
        fillmode = 1
    for i in range(len(xa)):
        if np.isnan(np.sum(xa[i])):
            for j in range(len(xa[i])//4):
                xa[i,j*4:4*j+5] = fill_nan(xa[i,j*4:4*j+5])
    #print(np.sum(np.isnan(xa)), flush=True)
    np.save(file[:-4]+'fixed'+'.npy', xa)
print('Initialization completed')

#==============================================================================================
# Execution
#==============================================================================================

root='/N/slate/kmluong/Training_data/'
for file in glob.iglob(root + '**/CNNfea*', recursive=True):
    print(file)
    if 'fixed' in file:
        continue
    fix_data(file)
print('Completed')
