import os
import numpy as np
import xarray as xr
import argparse
print('Starting')
def parse_args():
    parser = argparse.ArgumentParser(description='Extract WRF data.')
    parser.add_argument('--base', '-b', type=str, default = '18km', help='Base resolution used by the model.')
    parser.add_argument('-d', '--dinput_resolution', type=str, default = 'd01', help='Label quality (d01 = high, d03 = low).')
    parser.add_argument('-i', '--imsize', type=int, nargs=2, default = [64,64], help='Image size after extraction')
    parser.add_argument('-or', '--output_resolution', type=int, default = 18, help='Output grid resolution')
    parser.add_argument('-wd', '--workdir', type = str, default = '/N/project/Typhoon-deep-learning/output/', help = 'Working directory')
args = parse_args()
imsize = args.imsize
dxx = args.dinput_resolution
base = args.base
output_resolution = args.output_resultion
workdir = args.workdir
def extract_core_variables(ds):
    # Define the middle index for XLAT and XLONG assuming they are square and have the same dimensions
    mid_x = ds.sizes['west_east'] // 2
    mid_y = ds.sizes['south_north'] // 2

    # Calculate start and end indices for a 64x64 subset
    start_x = mid_x - imsize[0]//2
    end_x = mid_x + imsize[0]//2
    start_y = mid_y - imsize[1]//2
    end_y = mid_y + imsize[1]//2

    # Selecting specific variables and applying conditions
    u = ds.U.isel(bottom_top=slice(1, 4), south_north=slice(start_y, end_y), west_east_stag=slice(start_x, end_x))
    v = ds.V.isel(bottom_top=slice(1, 4), south_north_stag=slice(start_y, end_y), west_east=slice(start_x, end_x))
    t = ds.T.isel(bottom_top=slice(1, 4), south_north=slice(start_y, end_y), west_east=slice(start_x, end_x))
    qvapor = ds.QVAPOR.isel(bottom_top=slice(1, 4), south_north=slice(start_y, end_y), west_east=slice(start_x, end_x))
    psfc = ds.PSFC.isel(south_north=slice(start_y, end_y), west_east=slice(start_x, end_x))

    # Reshape for a uniform shape
    u = u.values.reshape(3, imsize[1], imsize[0])
    v = v.values.reshape(3, imsize[1], imsize[0])
    t = t.values.reshape(3, imsize[1], imsize[0])
    qvapor = qvapor.values.reshape(3, imsize[1], imsize[0])
    psfc = psfc.values.reshape(imsize[1], imsize[0])  # This line needed correction for shape

    combined = np.concatenate([u, v, t, qvapor], axis=0)  # This results in a (12, imsize[1], imsize[0]) array
    final_result = np.concatenate([combined, np.expand_dims(psfc, axis=0)], axis=0)  # Shape (13, imsize[1], imsize[0])

    # Calculate y
    
    u10 = ds.U10.isel(south_north=slice(start_y, end_y), west_east=slice(start_x, end_x))
    v10 = ds.V10.isel(south_north=slice(start_y, end_y), west_east=slice(start_x, end_x))
    wind_speed = np.sqrt(u10**2 + v10**2)

    # Convert DataArray to numpy array before finding the argmax
    wind_speed_np = wind_speed.values.reshape(64,64)  # Convert xarray DataArray to numpy array
    max_wind_loc = np.unravel_index(np.argmax(wind_speed_np, axis=None), wind_speed_np.shape)
    min_psfc_loc = np.unravel_index(np.argmin(psfc, axis=None), psfc.shape)

    # Calculating distances in grid points
    dist_x = np.abs(max_wind_loc[1] - min_psfc_loc[1])
    dist_y = np.abs(max_wind_loc[0] - min_psfc_loc[0])

    # Computing the Euclidean distance and converting it to kilometers
    distance_km = np.sqrt(dist_x**2 + dist_y**2) * output_resolution

    y = np.zeros((1, 3))
    y[0, 0] = np.max(wind_speed_np)  # Use numpy max for maximum wind speed
    y[0, 1] = np.min(psfc)  # Ensure using numpy array for psfc as well
    
    # Calculate distance to the center from the max wind speed location
    
    # Convert grid distance to kilometers (assuming dx=dy=18 km)
    y[0, 2] = distance_km  # Distance in kilometers   
    return final_result.reshape(1, 13, imsize[1], imsize[0]), y

base_path = "/N/project/Typhoon-deep-learning/data/tc-wrf/"
pattern = os.path.join(base_path, f"exp_{base}*")
exp_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
output_dir = os.path.join(workdir, "wrf_data")
os.makedirs(output_dir, exist_ok=True)
for exp_dir in exp_dirs:
    full_path = os.path.join(base_path, exp_dir)
    results = []
    ys = []  # List to hold y values

    for file_name in sorted(os.listdir(full_path)):
        if 'wrfout' in file_name and dxx in file_name:
            file_path = os.path.join(full_path, file_name)
            print("Processing file:", file_path)
            ds = xr.open_dataset(file_path)
            result, y = extract_core_variables(ds)  # Assume this function is defined elsewhere
            results.append(result)
            ys.append(y)
            ds.close()  # Close the dataset after processing

    # Concatenate all results and y values
    final_data = np.concatenate(results, axis=0)
    final_y = np.concatenate(ys, axis=0)

    # Write y values to text files and save numpy arrays
    for i in range(final_y.shape[1]):
        filename = os.path.join(output_dir, f"{exp_dir}_{dxx}_ys_{i}.txt")
        with open(filename, 'w') as file:
            file.write(','.join(map(str, final_y[:, i])))
            file.write('\n')  # Ensure newline at the end of the file

    # Save the final array and y values as .npy files in the specified directory
    np.save(os.path.join(output_dir, f"{exp_dir}_variables.npy"), final_data)
    np.save(os.path.join(output_dir, f"{exp_dir}_{dxx}_ys.npy"), final_y)

    print(f"Saved data for {exp_dir}, input resolution {dxx}, saved at {output_dir}")
