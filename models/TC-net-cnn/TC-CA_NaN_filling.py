print('Initializing')
import argparse
import os
import numpy as np
import copy
import glob
np.seterr(invalid='ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Context-aware NaN filling for multidimensional arrays.")
    parser.add_argument("--workdir", type=str, default='/N/project/Typhoon-deep-learning/output/', help="Working directory where data files are stored.")
    parser.add_argument("--windowsize", type=int, nargs=2, default=[19, 19], help="Window size for filling method [width, height].")
    parser.add_argument("--var_num", type=int, default=13, help="Number of variables.")
    parser.add_argument("--channel_map", type=str, default="0,1:0,1,2,3;4,5:4,5,6,7;8,9:8,9,10,11", help="Mapping of reference channels to channels that need fixing.")
    args = parser.parse_args()
    return args

#####################################################################################
# DO NOT EDIT BELOW UNLESS YOU WANT TO MODIFY THE SCRIPT
#####################################################################################
def parse_channel_map(channel_map_str):
    """
    Parse the channel mapping string into a dictionary.
    
    Example input: "0,1:0,1,2,3;4,5:4,5,6,7;8,9:8,9,10,11"
    Returns: { (0, 1): [0, 1, 2, 3], (4, 5): [4, 5, 6, 7], (8, 9): [8, 9, 10, 11] }
    """
    mapping = {}
    groups = channel_map_str.split(';')
    for group in groups:
        try:
            ref_str, fix_str = group.split(':')
            ref_channels = tuple(int(x) for x in ref_str.split(','))
            fix_channels = [int(x) for x in fix_str.split(',')]
            mapping[ref_channels] = fix_channels
        except Exception as e:
            raise ValueError(f"Invalid channel mapping format: {group}") from e
    return mapping
    
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
    """
    return vector2[:, :, 0] * vector1[:, :, 0] + vector2[:, :, 1] * vector1[:, :, 1]

def weight_field(vector):
    """
    Calculate weights for a given vector field.
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
    """
    new_arr = np.roll(array, place, axis=mode)
    if place > 0:
        if mode == 0:
            new_arr[:place] = np.zeros(new_arr[:place].shape)
        else:
            new_arr[:, :place] = np.zeros(new_arr[:, :place].shape)
    else:
        if mode == 0:
            new_arr[place:] = np.zeros(new_arr[place:].shape)
        else:
            new_arr[:, place:] = np.zeros(new_arr[:, place:].shape)
    return new_arr

def extract_bound(array):
    """
    Extract the boundary of a 2D array containing NaN values.
    """
    nan = np.isnan(array)
    notnan = np.logical_not(nan)
    s1 = np.logical_and(notnan, shift(nan, 1))
    s2 = np.logical_and(notnan, shift(nan, -1))
    s3 = np.logical_and(notnan, shift(nan, 1, mode=1))
    s4 = np.logical_and(notnan, shift(nan, -1, mode=1))
    bound = np.logical_or(s1, np.logical_or(s2, np.logical_or(s3, s4)))
    return bound

def fill4(array, ref, fix):
    bound = extract_bound(array[ref[0]])
    pattern = np.zeros(bound.shape)
    pattern[1:-1, 1:-1] = pattern[1:-1, 1:-1] + bound[1:-1, :-2] + bound[1:-1, 2:] + bound[:-2, 1:-1] + bound[2:, 1:-1]
    nan4 = np.logical_and(pattern == 4, np.isnan(array[ref[0]]))
    if np.sum(nan4) == 0:
        return array

    # Calculate vector field once from the reference channels.
    vector = calfield(array[list(ref)])
    for i in np.transpose(nan4.nonzero()):
        weight = weight_field(vector[i[0]-1:i[0]+2, i[1]-1:i[1]+2])
        for j in fix:
            if np.isnan(array[j, i[0], i[1]]):
                array[j, i[0], i[1]] = np.nansum(array[j, i[0]-1:i[0]+2, i[1]-1:i[1]+2] * weight)
    return array

def fill3(array, ref, fix):
    bound = extract_bound(array[ref[0]])
    pattern = np.zeros(bound.shape)
    pattern[1:-1, 1:-1] = pattern[1:-1, 1:-1] + bound[1:-1, :-2] + bound[1:-1, 2:] + bound[:-2, 1:-1] + bound[2:, 1:-1]
    nan3 = np.logical_and(pattern == 3, np.isnan(array[ref[0]]))
    if np.sum(nan3) == 0:
        return array

    vector = calfield(array[list(ref)])
    for i in np.transpose(nan3.nonzero()):
        weight = weight_field(vector[i[0]-1:i[0]+2, i[1]-1:i[1]+2])
        for j in fix:
            if np.isnan(array[j, i[0], i[1]]):
                array[j, i[0], i[1]] = np.nansum(array[j, i[0]-1:i[0]+2, i[1]-1:i[1]+2] * weight)
    return array

def fill2(array, ref, fix):
    bound = extract_bound(array[ref[0]])
    pattern = np.zeros(bound.shape)
    pattern[1:-1, 1:-1] = pattern[1:-1, 1:-1] + bound[1:-1, :-2] + bound[1:-1, 2:] + bound[:-2, 1:-1] + bound[2:, 1:-1]
    nan2 = np.logical_and(pattern == 2, np.isnan(array[ref[0]]))
    if np.sum(nan2) == 0:
        return array

    vector = calfield(array[list(ref)])
    for i in np.transpose(nan2.nonzero()):
        weight = weight_field(vector[i[0]-1:i[0]+2, i[1]-1:i[1]+2])
        for j in fix:
            if np.isnan(array[j, i[0], i[1]]):
                array[j, i[0], i[1]] = np.nansum(array[j, i[0]-1:i[0]+2, i[1]-1:i[1]+2] * weight)
    return array

def fill_nan(array, ref, fix):
    hold1 = 0
    # Instead of checking array[fix[0]] (which might be a reference channel),
    # we check all channels that need fixing (channels after the reference channels).
    while np.sum(np.isnan(array[list(fix)[0], 1:-1, 1:-1])) > 0:
        array = fill4(array, ref, fix)
        array = fill3(array, ref, fix)
        array = fill2(array, ref, fix)
        hold1 += 1
        if hold1 == 300:
            break
    array = np.pad(array, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    while np.sum(np.isnan(array[list(fix)[0], 1:-1, 1:-1])) > 0:
        array = fill4(array, ref, fix)
        array = fill3(array, ref, fix)
        array = fill2(array, ref, fix)
    array = array[:, 1:-1, 1:-1]
    return array

def fix_data(file, channel_mapping):
    """
    Fixes NaN values in the data stored in the given file using the fill_nan function
    and a user-specified channel mapping.
    
    Parameters:
    - file: str
        The file path for the data.
    - channel_mapping: dict
        Dictionary where keys are tuples of reference channel indices and
        values are lists of channels to fix (which include the reference channels).
        
        For example: { (0, 1): [0, 1, 2, 3], (4, 5): [4, 5, 6, 7], (8, 9): [8, 9, 10, 11] }
        
    Returns:
    - None
    """
    xa = np.load(file)
    for i in range(len(xa)):
        if np.isnan(np.sum(xa[i])):
            for ref_channels, fix_channels in channel_mapping.items():
                xa[i] = fill_nan(xa[i], ref_channels, fix_channels)
    np.save(file[:-4] + 'fixed' + '.npy', xa)
#
# MAIN CALL: 
#
def main():
    args = get_args()
    # Initialize parameters
    workdir = args.workdir
    windowsize = list(args.windowsize)
    var_num = args.var_num
    channel_mapping = parse_channel_map(args.channel_map)
    windows = f"{windowsize[0]}x{windowsize[1]}"
    root = os.path.join(workdir, 'Domain_data', f'exp_{var_num}features_{windows}/')
    pattern = f"{root}**/features*.npy"

    # Find and process the files
    for file in glob.glob(pattern, recursive=True):
        print("Checking: ", file)
        if 'fixed' in file:
            continue
        fix_data(file, channel_mapping)

    print('Processing completed.')

if __name__ == "__main__":
    main()
