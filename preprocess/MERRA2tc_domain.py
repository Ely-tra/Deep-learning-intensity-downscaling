import os
import pandas as pd
import xarray as xr #use xarray because it is far more better than netCDF4 
from datetime import datetime  #Use datetime to name the output files

#####################################

def get_runid(formatted_time):
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
    filename = f"MERRA2_401.inst3_3d_asm_Np.{formatted_time}.nc4"
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
def process_years(years,df):
  """
    Process the 'SEASON' column in the DataFrame based on the provided years.

    Parameters:
    - years (str, int or list): The year(s) to search for in the 'SEASON' column.
    - df (DataFrame): The input DataFrame.

    Returns:
    - DataFrame: The filtered DataFrame based on the specified years.
  """
  if years=='':  
	#pass if no search value is given
    pass
  else:
    if isinstance(years, list):  #if search value is a list, then search for a list
        years_int = [int(year) for year in years]  #convert search value into int
        df = df[df['SEASON'].isin(years_int)] #filter the dataframe
    elif years is not None: #if it is not a list, then search for a year
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
  if names=='':
    pass
	#pass if no search value is given
  else:
    if isinstance(names, list) and all(isinstance(elem, str) for elem in names):
      df = df[df['NAME'].isin(names)]
    if isinstance(names, str):
      df = df[df['NAME'] == names]
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
  if minlat == -90.0 and maxlat == 90.0 and minlon == -180.0 and maxlon == 180.0:  #default value, pass to save computational power
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
        pass
    else:
        if isinstance(regions, list) and all(isinstance(elem, str) for elem in regions):
            df = df[df['BASIN'].isin(regions)]
        elif isinstance(regions, str):
            df = df[df['BASIN'] == regions]
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
  if maxwind==10000 and minwind==0:
    pass
  else:
    df = df[(df['WMO_WIND'] <= maxwind) & (df['WMO_WIND'] >= minwind)]
  return df

#########################################


def trim_pressure_range(df,maxpres,minpres):
    """
    Trim the DataFrame based on specified maximum and minimum pressure values.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - maxpres (int): Maximum pressure value.
    - minpres (int): Minimum pressure value.

    Returns:
    - DataFrame: The trimmed DataFrame based on the specified pressure range.
    """
    if maxpres==10000 and minpres==0:
        pass
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
        pass
    else:
        df = df[(df['USA_RMW'] <= maxrmw) & (df['USA_RMW'] >= minrmw)]
    return df

#########################################


def merge_data(csvdataset, tc_name='', years='', minlat = -90.0, maxlat = 90.0, minlon = -180.0, maxlon = 180.0, regions='', maxwind=10000, minwind=0, maxpres=10000, minpres=0, maxrmw=10000, minrmw=0, windowsize=[18,18], datapath=''): #define a search bar for you, csvdataset is the link to the dataset, tc_name are names to search for, years are years to search for, .... Window size is the intended output around the TC center, 18 degree means center+/- 9 degree.
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
        windowsize (tuple): The intended output window around the TC center. Default is (18, 18).
        datapath (str): Path to the data. Default is ''.

    Returns:
        None

    Additional Attributes Assigned:
        - VMAX (float): Maximum wind speed.
        - PMIN (float): Minimum pressure.
        - RMW (float): Radius of maximum wind.
        - Center Latitude (float): Latitude of the TC center.
        - Center Longitude (float): Longitude of the TC center.
  """
  selected_columns = ["SEASON", "BASIN", "NAME", "LAT", 
                      "LON", "ISO_TIME", "WMO_WIND", 
                      "WMO_PRES", "USA_RMW"]                          #define the important columns, some for search bar, some for interest
  df=pd.read_csv(csvdataset, usecols=selected_columns) #read data using pandas read csv
  filtered_df = df[
    (df['WMO_WIND'].apply(lambda x: str(x).isnumeric())) & 
    (df['WMO_PRES'].apply(lambda x: str(x).isnumeric())) & 
    (df['USA_RMW'].apply(lambda x: str(x).isnumeric()))][selected_columns] #pick only where max wind speed, min pressure, and RMW are numbers
  filtered_df["WMO_WIND"] = filtered_df["WMO_WIND"].astype(float) 
  #still need to convert them to number, because >>some<< entries are strings, that's why I have to use isnumeric.
  filtered_df["WMO_PRES"] = filtered_df["WMO_PRES"].astype(float)
  filtered_df["USA_RMW"] = filtered_df["USA_RMW"].astype(float)
  del df #liberate some data, for other users
  filtered_df=process_years(years, filtered_df) #search bar
  filtered_df=process_name(tc_name, filtered_df)
  filtered_df=process_regions(regions, filtered_df)
  filtered_df=trim_area(filtered_df, maxlat=maxlat,minlat=minlat,maxlon=maxlon,minlon=minlon)
  filtered_df=trim_wind_range(filtered_df, maxwind=maxwind, minwind=minwind)
  filtered_df=trim_pressure_range(filtered_df, maxpres=maxpres, minpres=minpres)
  filtered_df=trim_rmw_range(filtered_df, maxrmw=maxrmw, minrmw=minrmw)
  for time in filtered_df['ISO_TIME']: 
    lowerlat=max(-90,filtered_df[filtered_df['ISO_TIME']==time]['LAT'].values[0]-windowsize[0]/2) #define the window
    upperlat=min(90, lowerlat+windowsize[0])
    lowerlon=max(-180,filtered_df[filtered_df['ISO_TIME']==time]['LON'].values[0]-windowsize[1]/2)
    upperlon=min(180, lowerlon+windowsize[1])
    formatted_time = pd.to_datetime(time).strftime('%Y%m%d') #read corresponding data file
    dataname=datapath+'MERRA2_'+str(get_runid(formatted_time))+'.inst3_3d_asm_Np.'+formatted_time+'.nc4'
    dataset = xr.open_dataset(dataname)
    window=dataset.sel(time=time, lat=slice(lowerlat, upperlat), lon=slice(lowerlon,upperlon)) #cut the window
    window=window.assign_attrs(VMAX=filtered_df[filtered_df['ISO_TIME'] == time]['WMO_WIND'].values[0], 
                               PMIN=filtered_df[filtered_df['ISO_TIME'] == time]['WMO_PRES'].values[0], 
    			       RMW=filtered_df[filtered_df['ISO_TIME'] == time]['USA_RMW'].values[0], 
			       CLAT=filtered_df[filtered_df['ISO_TIME'] == time]['LAT'].values[0], 
			       CLON=filtered_df[filtered_df['ISO_TIME'] == time]['LON'].values[0], 
			       TCNAME=filtered_df[filtered_df['ISO_TIME'] == time]['NAME'].values[0] ) 
			       #assign new attributes, Max wind speed, Min pressure and radius of maximum wind
    formatted_datetime = datetime.strptime(time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H') #take YYYYMMDDHH format to build filename
    outname='TC_domain/MERRA_TC'+str(windowsize[0])+'x'+str(windowsize[1])+formatted_datetime[:-2]+formatted_datetime[-2:]+'.nc' 
    window.to_netcdf(outname) #print out the new file, its name is MERRA_TCW1xW2YYYYMMDDHH.nc
datapath='/N/scratch/tqluu/merra2-nasa/full/'

csvdataset='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
merge_data(csvdataset, tc_name='HAIYAN',  datapath=datapath) 
#tc_name (str or None), years (str or None), minlat (float), maxlat (float), minlon (float), maxlon (float), regions (str or None), maxwind (int), minwind (int), maxpres (int), minpres (int), maxrmw (int), minrmw (int), windowsize (tuple), datapath (str)  
#Define parameters, only csvdataset is required, if no keyword argument is given, the function search for the whole domain            
