#
# DESCRIPTION: This script is to read in MERRA dataset and produces a domain selection
#       around TC center from the best track data set. The domain will have a
#       fixed size, and incorporate the TC information as global attribute. This
#       script will produce output data centered on the best track data with a
#       given domain size, with the same as, e.g., "MERRA_TC30x302021090818_4.nc"
#
#       Because there may have more than 1 storm each time, the prefix "_n.nc" in
#       each file name is needed to indicate different storms at each time. 
#
# HIST: - Jan 26, 2024: created by Khanh Luong
#       - May 14, 2024: cross-checked and cleaned up by CK
#
# USAGE: please edit the data paths and parameters under right below before running
#
# NOTE: This script will produce some error messages related to cfgrib and eccodes
#       libraries. However, it still runs properly as can be checked from the output
#       location.
#
# AUTH: Khanh Luong (kmluong@iu.edu)
#====================================================================================
import warnings
warnings.filterwarnings("ignore") # Suppress all warnings
import os
import pandas as pd
import numpy as np
import xarray as xr               # use xarray as it is better than netCDF4 
from datetime import datetime     # Use datetime to name the output files
from timeit import default_timer as timer
import argparse
#
# Users need to edit input data paths for both reanlysis
# and best track data, and an output data path here.
#
#datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'
#workdir='/N/project/Typhoon-deep-learning/output/'
#csvdataset='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
#windowsize=[19,19]
#regions=['EP', 'NA', 'WP']
def parse_args():
    parser = argparse.ArgumentParser(description="Merge datasets of tropical cyclones, apply filters, and process data.")
    parser.add_argument("--csvdataset", type=str, default='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv', help="The path to the dataset in CSV format.")
    parser.add_argument("--tc_name", type=str, default='', help="Names of the tropical cyclones to search for. Empty by default.")
    parser.add_argument("--years", type=str, default='', help="Years to search for. Empty by default.")
    parser.add_argument("--minlat", type=float, default=-90.0, help="Minimum latitude.")
    parser.add_argument("--maxlat", type=float, default=90.0, help="Maximum latitude.")
    parser.add_argument("--minlon", type=float, default=-180.0, help="Minimum longitude.")
    parser.add_argument("--maxlon", type=float, default=180.0, help="Maximum longitude.")
    parser.add_argument("--regions", nargs='+', default=['EP', 'NA', 'WP'], help="Regions to search for. Specify multiple regions separated by spaces.")
    parser.add_argument("--maxwind", type=int, default=10000, help="Maximum wind speed in knots.")
    parser.add_argument("--minwind", type=int, default=0, help="Minimum wind speed in knots.")
    parser.add_argument("--maxpres", type=int, default=10000, help="Maximum pressure.")
    parser.add_argument("--minpres", type=int, default=0, help="Minimum pressure.")
    parser.add_argument("--maxrmw", type=int, default=10000, help="Maximum radius of maximum wind.")
    parser.add_argument("--minrmw", type=int, default=0, help="Minimum radius of maximum wind.")
    parser.add_argument("--windowsize", type=int, nargs=2, default=[19, 19], help="The window size around the TC center, specified as two integers for latitude and longitude.")
    parser.add_argument("--datapath", type=str, default='/N/project/Typhoon-deep-learning/data/nasa-merra2/', help="Path to the data files.")
    parser.add_argument("--outputpath", type=str, default='/N/slate/kmluong/TC-net-cnn_workdir/', help="Path to save the output files.")

    return parser.parse_args()
#####################################################################################
# DO NOT EDIT BELOW UNLESS YOU WANT TO MODIFY THE SCRIPT
#####################################################################################
def process_entries(per):
    """
    Print progress information based on the count and specified interval.

    Parameters:
    - per (int): The interval for printing progress information.

    Returns:
    None
    """
    global count
    global starttime
    global endtime
    global faulty
    if per == 0:
        per = 100
    if (count + faulty) % per == 0:
        endtime = timer()
        entries_processed = count + faulty
        progress_percentage = (entries_processed / entries) * 100
        print(f'{entries_processed} entries processed over {entries}, {progress_percentage:.2f}% done.', flush=True)

        time_used = endtime - starttime
        print(f'Time used for the last {per} entries: {time_used:.2f}', flush=True)

        estimate = (entries - count - faulty) / per * time_used
        starttime = timer()
        print(f'Time left: {estimate:.2f}', flush=True)

#####################################
def get_runid(formatted_time, datapath=''):
    """
    Convert formatted_time to a numeric representation (assuming it's a date in YYYYMMDD format)
    and return a runid based on predefined date ranges.

    Parameters:
    - formatted_time (str): A string representing the formatted time in YYYYMMDD format.

    Returns:
    - int or None: The runid (100, 200, 300, 400, 401) corresponding to the date range of formatted_time.
                  Returns None if the date doesn't fall into any defined range.

    Example:
    >>> formatted_time = "19950115"
    >>> runid = get_runid(formatted_time)
    >>> print(runid)
    100
    """
    # Convert formatted_time to a numeric representation (assuming it's a date in YYYYMMDD format)
    numeric_time = int(formatted_time)

    # Define date ranges
    range_1 = (0, 19911231)
    range_2 = (19920101, 20001231)
    range_3 = (20010101, 20101231)
    filename = datapath + f"MERRA2_401.inst3_3d_asm_Np.{formatted_time}.nc4"
    # Check the date range and return the corresponding runid
    if range_1[0] <= numeric_time <= range_1[1]:
        return 100
    elif range_2[0] <= numeric_time <= range_2[1]:
        return 200
    elif range_3[0] <= numeric_time <= range_3[1]:
        return 300
    elif os.path.isfile(filename):
        return 401
    else:
        return 400  # Return None if the date doesn't fall into any defined range

######################################
def process_years(years, df):
    """
    Process the 'SEASON' column in the DataFrame based on the provided years.

    Parameters:
    - years (str, int or list): The year(s) to search for in the 'SEASON' column.
    - df (DataFrame): The input DataFrame.

    Returns:
    - DataFrame: The filtered DataFrame based on the specified years.
    """
    if years == '':  
        # Pass if no search value is given
        pass
    else:
        if isinstance(years, list):  # If search value is a list, then search for a list
            years_int = [int(year) for year in years]  # Convert search value into int
            df = df[df['SEASON'].isin(years_int)]  # Filter the DataFrame
        elif years is not None:  # If it is not a list, then search for a year
            df = df[df['SEASON'] == int(years)]
    return df


#######################################
def process_name(names, df):
    """
    Process the 'NAME' column in the DataFrame based on the provided names.

    Parameters:
    - names (str or list): The name(s) to search for in the 'NAME' column.
    - df (DataFrame): The input DataFrame.

    Returns:
    - DataFrame: The filtered DataFrame based on the specified names.
    """ 
    if names == '':
        pass  # Pass if no search value is given
    else:
        if isinstance(names, list) and all(isinstance(elem, str) for elem in names):
            df = df[df['NAME'].isin(names)]  # Filter the DataFrame by list of names
        if isinstance(names, str):
            df = df[df['NAME'] == names]  # Filter the DataFrame by a single name
    return df

#########################################
def trim_area(df, maxlat, minlat, maxlon, minlon):
    """
    Trim the DataFrame based on specified latitude and longitude bounds.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - maxlat (float): Maximum latitude.
    - minlat (float): Minimum latitude.
    - maxlon (float): Maximum longitude.
    - minlon (float): Minimum longitude.

    Returns:
    - DataFrame: The trimmed DataFrame based on the specified bounds.
    """
    if minlat == -90.0 and maxlat == 90.0 and minlon == -180.0 and maxlon == 180.0:  # Default value, pass to save computational power
        pass
    else:
        df = df[(df['LAT'] <= maxlat) & (df['LAT'] >= minlat) & (df['LON'] >= minlon) & (df['LON'] <= maxlon)]
    return df


#########################################
def process_regions(regions, df):
    """
    Filter the DataFrame based on specified regions in the 'BASIN' column.

    Parameters:
    - regions (str or list): The region(s) to search for in the 'BASIN' column.
    - df (DataFrame): The input DataFrame.

    Returns:
    - DataFrame: The filtered DataFrame based on the specified regions.
    """
    if regions == '':
        pass  # Do nothing if no region is specified
    else:
        if isinstance(regions, list) and all(isinstance(elem, str) for elem in regions):
            df = df[df['BASIN'].isin(regions)]  # Filter by a list of regions
        elif isinstance(regions, str):
            df = df[df['BASIN'] == regions]  # Filter by a single region
    return df

#########################################
def trim_wind_range(df, maxwind, minwind):
    """
    Trim the DataFrame based on specified maximum and minimum wind speeds.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - maxwind (int): Maximum "maximum" wind speed in knots.
    - minwind (int): Minimum "maximum" wind speed in knots.

    Returns:
    - DataFrame: The trimmed DataFrame based on the specified wind speed range.
    """
    if maxwind == 10000 and minwind == 0:
        pass  # Do nothing if default extreme values are used
    else:
        df = df[(df['WMO_WIND'] <= maxwind) & (df['WMO_WIND'] >= minwind)]
    return df

#########################################
def trim_pressure_range(df, maxpres, minpres):
    """
    Trim the DataFrame based on specified maximum and minimum pressure values.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - maxpres (int): Maximum pressure value.
    - minpres (int): Minimum pressure value.

    Returns:
    - DataFrame: The trimmed DataFrame based on the specified pressure range.
    """
    if maxpres == 10000 and minpres == 0:
        pass  # Do nothing if default extreme values are used
    else:
        df = df[(df['WMO_PRES'] <= maxpres) & (df['WMO_PRES'] >= minpres)]
    return df

#########################################
def trim_rmw_range(df, maxrmw, minrmw):
    """
    Trim the DataFrame based on specified maximum and minimum Radius of Maximum Wind values.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - maxrmw (int): Maximum RMW value.
    - minrmw (int): Minimum RMW value.

    Returns:
    - DataFrame: The trimmed DataFrame based on the specified USA RMW range.
    """
    if maxrmw == 10000 and minrmw == 0:
        pass  # Do nothing if default extreme values are used
    else:
        df = df[(df['USA_RMW'] <= maxrmw) & (df['USA_RMW'] >= minrmw)]
    return df

#########################################
def merge_data(csvdataset, tc_name='', years='', minlat = -90.0, maxlat = 90.0, minlon = -180.0, maxlon = 180.0,
               regions='', maxwind=10000, minwind=0, maxpres=10000, minpres=0, maxrmw=10000, minrmw=0, windowsize=[18,18],
               datapath='', completed=0, outputpath=''): 
    """
    Merge two datasets of tropical cyclones, create a window for a TC domain with additional attributes, and write to files.

    Args:
        csvdataset (str): The link to the dataset in CSV format.

    Kwargs:
        tc_name (str or None): Names to search for. Default is None.
        years (str or None): Years to search for. Default is None.
        minlat (float): Minimum latitude. Default is -90.0.
        maxlat (float): Maximum latitude. Default is 90.0.
        minlon (float): Minimum longitude. Default is -180.0.
        maxlon (float): Maximum longitude. Default is 180.0.
        regions (str or None): Regions to search for. Default is None.
        maxwind (int): Maximum wind speed in knots. Default is 10000.
        minwind (int): Minimum wind speed in knots. Default is 0.
        maxpres (int): Maximum pressure. Default is 10000.
        minpres (int): Minimum pressure. Default is 0.
        maxrmw (int): Maximum USA RMW. Default is 10000.
        minrmw (int): Minimum USA RMW. Default is 0.
        windowsize (tuple): The intended output window around the TC center. Default is (18, 18), lat, lon.
        datapath (str): Path to the data. Default is ''.
        outputpath (str): Path to the output. Default is ''.

    Returns:
        None

    Additional Attributes Assigned:
        - VMAX (float): Maximum wind speed.
        - PMIN (float): Minimum pressure.
        - RMW (float): Radius of maximum wind.
        - Center Latitude (float): Latitude of the TC center.
        - Center Longitude (float): Longitude of the TC center.
    """
    # 
    # read CSV data and process the datatype, selecting only TC with defined characteristics
    #
    selected_columns = ["SEASON", "BASIN", "NAME", "LAT", 
                        "LON", "ISO_TIME", "WMO_WIND", 
                        "WMO_PRES", "USA_RMW"]                          # define the important columns, some for search bar, some for interest
    df = pd.read_csv(csvdataset, usecols=selected_columns, keep_default_na=False)  # read data using pandas read csv
    filtered_df = df[
        (df['WMO_WIND'].apply(lambda x: str(x).isnumeric())) & 
        (df['WMO_PRES'].apply(lambda x: str(x).isnumeric())) & 
        (df['USA_RMW'].apply(lambda x: str(x).isnumeric()))][selected_columns]  # pick only where max wind speed, min pressure, and RMW are numbers
    filtered_df["WMO_WIND"] = filtered_df["WMO_WIND"].astype(float) 
    # still need to convert them to number, because >>some<< entries are strings, that's why I have to use isnumeric.
    filtered_df["WMO_PRES"] = filtered_df["WMO_PRES"].astype(float)
    filtered_df["USA_RMW"] = filtered_df["USA_RMW"].astype(float)
    del df  # liberate some data, for other users
    filtered_df = process_years(years, filtered_df)  # search bar
    filtered_df = process_name(tc_name, filtered_df)
    filtered_df = process_regions(regions, filtered_df)
    filtered_df = trim_area(filtered_df, maxlat=maxlat, minlat=minlat, maxlon=maxlon, minlon=minlon)
    filtered_df = trim_wind_range(filtered_df, maxwind=maxwind, minwind=minwind)
    filtered_df = trim_pressure_range(filtered_df, maxpres=maxpres, minpres=minpres)
    filtered_df = trim_rmw_range(filtered_df, maxrmw=maxrmw, minrmw=minrmw)
    filtered_df = filtered_df.sort_values('ISO_TIME')
    #
    # Initiate counter value
    #
    global count
    count = 0
    global starttime
    starttime = timer()
    global faulty
    faulty = 0
    global suffix
    suffix = 0
    global previous_time
    previous_time = 0  # There can be 2 or more TCs happen at the same time
    global entries
    entries = len(filtered_df)
    global endtime
    endtime = 0
    print('Total: ' + str(entries), flush=True)
    latsize = int(np.ceil((np.ceil(windowsize[0]/0.5)+1)/2))  # change 0.625 and 0.5 for other dataset
    lonsize = int(np.ceil((np.ceil(windowsize[1]/0.625)+1)/2))
    filtered_df['LON'] = filtered_df['LON'] - 360*np.logical_and(filtered_df['BASIN'].isin(['EP','SP']), filtered_df['LON'] > 0)
    #
    # Loop through filtered data
    #
    for index, row in filtered_df.iterrows():
        window_df = row
        time = row['ISO_TIME']
        if time == previous_time:
            suffix += 1
        elif time != previous_time:
            suffix = 0
        previous_time = time
        if count < completed:
            count += 1
            continue
        #
        # Filter out data with unregistered time
        # 
        formatted_datetime = datetime.strptime(time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H')  # take YYYYMMDDHH format to build filename 
        if datetime.strptime(time, '%Y-%m-%d %H:%M:%S').minute != 0:
            print('Faulty entry ' + time + ' unexpected minute.', flush=True)
            faulty += 1
            process_entries(round(entries * 0.01 / 100) * 100)
            continue
        if datetime.strptime(time, '%Y-%m-%d %H:%M:%S').second != 0:
            print('Faulty entry ' + time + ' unexpected second.', flush=True)
            faulty += 1
            process_entries(round(entries * 0.01 / 100) * 100)
            continue
        if formatted_datetime[-2:] not in ['00', '03', '06', '09', '12', '15', '18', '21']:
            print('Faulty entry ' + time + ' unexpected hour.', flush=True)
            faulty += 1
            process_entries(round(entries * 0.01 / 100) * 100)
            pass
        #
        # Ensure the windows are of the same size, deal with antimeridian data
        #
        else:
            gblat = (window_df['LAT'] + 90) // 0.5
            lower_index_lat = int(gblat - latsize + 1)
            upper_index_lat = int(gblat + latsize + 1)
            formatted_time = pd.to_datetime(time).strftime('%Y%m%d')  # read corresponding data file
            dataname = datapath + 'MERRA2_' + str(get_runid(formatted_time, datapath)) + '.inst3_3d_asm_Np.' + formatted_time + '.nc4'
            dataset = xr.open_dataset(dataname)
            if lower_index_lat < 0 or upper_index_lat > len(dataset.lat) - 1:
                faulty += 1
                print('Cannot create a window of designed size for this TC, outside of map.', flush=True)
                print('ASDFGHJ' + window_df['ISO_TIME'] + window_df['NAME'] + window_df['BASIN'], flush=True)
                process_entries(round(entries * 0.01 / 100) * 100)
                continue
            gblon = (window_df['LON'] + 180) // 0.625
            # Calculate the lower and upper longitude indices
            lower_index_lon = int(gblon - lonsize + 1)
            upper_index_lon = int(gblon + lonsize + 1)
            window = dataset.sel(time=time)  # cut the window
            full_length = len(window.lon)
            if upper_index_lon > (full_length - 1):
                window = window.isel(lon=(window.lon < window.lon[upper_index_lon - full_length]) | (window.lon > window.lon[lower_index_lon - 1]))
                window = window.isel(lat=slice(lower_index_lat, upper_index_lat))
                window = window.roll(lon=full_length - lower_index_lon)
            elif lower_index_lon < 0:
                window = window.isel(lon=(window.lon > window.lon[lower_index_lon - 1]) | (window.lon < window.lon[upper_index_lon]))
                window = window.isel(lat=slice(lower_index_lat, upper_index_lat))
                window = window.roll(lon=-lower_index_lon)
            else:
                window = window.isel(lat=slice(lower_index_lat, upper_index_lat), lon=slice(lower_index_lon, upper_index_lon))
            #
            # assign attrs and print out
            #
            window = window.assign_attrs(VMAX=window_df['WMO_WIND'], 
                                         PMIN=window_df['WMO_PRES'], 
                                         RMW=window_df['USA_RMW'], 
                                         CLAT=window_df['LAT'], 
                                         CLON=window_df['LON'], 
                                         TCNAME=window_df['NAME']) 
                                         # assign new attributes, Max wind speed, Min pressure and radius of maximum wind
            formatted_datetime = datetime.strptime(time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H')  # take YYYYMMDDHH format to build filename
            basin = window_df['BASIN']
            outname = outputpath
            outname = outname + 'TC_domain/' + basin 
            if not os.path.exists(outname):
                os.makedirs(outname)
            outname = outname + '/' + formatted_datetime[:4]
            if not os.path.exists(outname):
                os.makedirs(outname)
            outname = outname + '/' + 'MERRA_TC' + str(windowsize[0]) + 'x' + str(windowsize[1]) + formatted_datetime + '_' + str(suffix) + '.nc'
            outname = str(outname)
            window.to_netcdf(outname)  # print out the new file, its name is MERRA_TCW1xW2YYYYMMDDHH.nc
            count = count + 1
            process_entries(round(entries * 0.01 / 100) * 100)
    print('Total: ' + str(entries) + ' entries processed.', flush=True)
    print('With ' + str(faulty) + ' faulty entries.', flush=True)
    print('Generated ' + str(count) + ' windows.', flush=True)
#
# MAIN CALL: 
#
if __name__ == "__main__":
    
    # Arguments Input
    args = parse_args()
    print("Window size", args.windowsize)
    merge_data(**vars(args))
#For processing faulty window size, use minlon=171-0.625*3 and maxlon=-171+0.625*3 >>>max-windowsize+3gridsize<<<
#tc_name (str or None), years (str or None), minlat (float), maxlat (float), minlon (float), maxlon (float), regions (str or None), maxwind (int), minwind (int), maxpres (int), minpres (int), maxrmw (int), minrmw (int), windowsize (tuple) default [18,18], datapath (str)  
#Define parameters, only csvdataset is required, if no keyword argument is given, the function search for the whole domain            
