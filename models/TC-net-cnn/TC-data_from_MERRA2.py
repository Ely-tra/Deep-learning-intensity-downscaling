import os
import glob
import argparse
import xarray as xr
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Process NetCDF files and compute TC metrics.")
    parser.add_argument("--input_path", "-i", type=str, default="/N/project/Typhoon-deep-learning/output/TC_domain",
                        help="Root folder containing NetCDF files to process.")
    parser.add_argument("--output_path", "-o", type=str, default="/N/project/Typhoon-deep-learning/output/Processed",
                        help="Output folder to save processed results.")
    parser.add_argument("--level", "-l", type=int, default=1000,
                        help="The level at which to extract wind components (e.g., 1000).")
    return parser.parse_args()

def haversine(lon1, lat1, lon2, lat2):
    """
    Compute the great-circle distance between two points on Earth using the haversine formula.
    
    Parameters:
      lon1, lat1: float
          Longitude and latitude of the first point (in degrees).
      lon2, lat2: float
          Longitude and latitude of the second point (in degrees).
    
    Returns:
      distance_km: float
          Distance between the two points in kilometers.
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Differences in coordinates
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    
    # Haversine formula
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    radius_km = 6371
    distance_km = radius_km * c
    return distance_km

def extract_tc_metrics(ds, lev):
    """
    Extract tropical cyclone metrics from an xarray dataset.
    
    This function performs two sets of extractions:
    
    1. Computed metrics based on U and V components and SLP:
       - Reads U and V wind components at the specified level.
       - Computes the wind speed (in m/s) then converts it to knots (1 m/s = 1.94384 knots).
       - Determines the location of the maximum wind speed.
       - Computes the distance (in nautical miles) between this location and the storm center 
         (given by ds attributes "CLAT" and "CLON").
       - Finds the minimum sea level pressure (SLP) from the dataset and converts it from Pascals to millibars.
    
    2. Extracted attributes from the dataset:
       - Retrieves stored attributes "VMAX", "PMIN", and "RMW".
    
    Parameters:
      ds : xarray.Dataset
          Dataset containing variables "U", "V", "SLP", and coordinates "lat" and "lon".
          The dataset should include attributes:
             "CLAT" (center latitude),
             "CLON" (center longitude),
             "VMAX" (atmospheric maximum wind),
             "PMIN" (atmospheric minimum sea level pressure),
             "RMW"  (radius of maximum wind).
      lev : int or float
          The level at which to extract the wind components.
    
    Returns:
      tuple:
         (vmax, pmin, rmw, attr_vmax, attr_pmin, attr_rmw)
         where:
             vmax      : Computed maximum wind speed in knots.
             pmin      : Computed minimum sea level pressure in millibars.
             rmw       : Computed radius of maximum wind in nautical miles.
             attr_vmax : Extracted "VMAX" attribute.
             attr_pmin : Extracted "PMIN" attribute.
             attr_rmw  : Extracted "RMW" attribute.
    """
    # 1. Extract U and V at the specified level
    try:
        u_level = ds["U"].sel(lev=lev)
        v_level = ds["V"].sel(lev=lev)
    except KeyError:
        raise ValueError("The specified level is not available for the U and/or V components.")
    
    # Compute wind speed (m/s) at each grid point
    wind_speed = np.sqrt(u_level.values**2 + v_level.values**2)
    # Convert wind speed to knots (1 m/s = 1.94384 knots)
    wind_speed_knots = wind_speed * 1.94384
    
    # Determine the grid point with the maximum wind speed
    max_idx = np.unravel_index(np.nanargmax(wind_speed_knots), wind_speed_knots.shape)
    vmax = wind_speed_knots[max_idx]
    
    # 2. Retrieve latitude and longitude coordinates
    if "lat" in ds.coords and "lon" in ds.coords:
        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values
    else:
        raise ValueError("Dataset must include 'lat' and 'lon' coordinates.")
    
    # Get the lat, lon of the grid point with max wind speed.
    lat_maxwind = lat_vals[max_idx[0]]
    lon_maxwind = lon_vals[max_idx[1]]
    
    # 3. Retrieve storm center coordinates from dataset attributes
    try:
        center_lat = ds.attrs["CLAT"]
        center_lon = ds.attrs["CLON"]
    except KeyError:
        raise ValueError("Storm center coordinates 'CLAT' and 'CLON' must be specified in the dataset attributes.")
    
    # Compute the distance (in km) between storm center and max wind location using the haversine formula
    distance_km = haversine(center_lon, center_lat, lon_maxwind, lat_maxwind)
    # Convert km to nautical miles (1 nautical mile ≈ 1.852 km)
    rmw = distance_km / 1.852
    
    # 4. Compute minimum sea level pressure (SLP)
    if "SLP" in ds:
        # Convert SLP from Pascals to millibars (1 mb = 100 Pa)
        pmin = np.nanmin(ds["SLP"].values) / 100.0
    else:
        raise ValueError("Dataset must contain the 'SLP' variable for sea level pressure.")
    
    # 5. Extract the attributes for VMAX, PMIN, and RMW stored in the dataset attributes
    try:
        attr_vmax = float(ds.attrs["VMAX"])
        attr_pmin = float(ds.attrs["PMIN"])
        attr_rmw  = float(ds.attrs["RMW"])
    except KeyError as e:
        raise ValueError("The dataset must include 'VMAX', 'PMIN', and 'RMW' attributes. Missing: " + str(e))
    
    return vmax, pmin, rmw, attr_vmax, attr_pmin, attr_rmw

def compute_errors(computed, stored):
    """
    Compute the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
    between computed and stored arrays.
    """
    mae = np.mean(np.abs(computed - stored))
    rmse = np.sqrt(np.mean((computed - stored) ** 2))
    return mae, rmse

def main():
    args = parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    level = args.level  # Use the parsed level value

    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Construct a recursive glob pattern to match all NetCDF files.
    file_pattern = os.path.join(input_path, '**', '*.nc')
    file_counter = 0
    samples = []  # List to collect the 6-number metrics from each file

    # Loop over every matching file in the input directory tree.
    for file_path in glob.iglob(file_pattern, recursive=True):
        try:
            ds = xr.open_dataset(file_path)
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            continue

        try:
            # Extract six metrics using extract_tc_metrics, using the parsed level value.
            sample = extract_tc_metrics(ds, level)
            samples.append(sample)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        file_counter += 1

    # Convert list of samples into a NumPy array with shape (n_samples, 6)
    samples = np.array(samples)
    
    # Save the concatenated array to a NumPy binary file.
    samples_file = os.path.join(output_path, "samples.npy")
    np.save(samples_file, samples)
    print(f"Saved concatenated samples to {samples_file}")

    # Check if any samples were collected.
    if samples.shape[0] == 0:
        print("No valid samples were processed.")
        return

    # Assuming the order is:
    # [0]: computed vmax, [1]: computed pmin, [2]: computed rmw,
    # [3]: attribute VMAX, [4]: attribute PMIN, [5]: attribute RMW.
    mae_vmax, rmse_vmax = compute_errors(samples[:, 0], samples[:, 3])
    mae_pmin, rmse_pmin = compute_errors(samples[:, 1], samples[:, 4])
    mae_rmw,  rmse_rmw  = compute_errors(samples[:, 2], samples[:, 5])
    
    # Report the errors.
    print(f"Total {file_counter} file(s) processed.")
    #print("MAE and RMSE for computed vmax vs stored VMAX:")
    #print(f"  MAE: {mae_vmax:.3f}, RMSE: {rmse_vmax:.3f}")
    #print("MAE and RMSE for computed pmin vs stored PMIN:")
    #print(f"  MAE: {mae_pmin:.3f}, RMSE: {rmse_pmin:.3f}")
    #print("MAE and RMSE for computed rmw vs stored RMW:")
    #print(f"  MAE: {mae_rmw:.3f}, RMSE: {rmse_rmw:.3f}")
    print("Done.")

if __name__ == "__main__":
    main()
