import glob
import re
import os
import numpy as np
import xarray as xr
import argparse

print('Starting')

def parse_args():
    parser = argparse.ArgumentParser(description='Extract WRF data without eid and m_id.')
    parser.add_argument('-ix', '--imsize_variables', type=int, nargs=2, default=[64, 64],
                        help='Image size for variable extraction (width height)')
    parser.add_argument('-iy', '--imsize_labels', type=int, nargs=2, default=[64, 64],
                        help='Image size for label extraction (width height)')
    parser.add_argument('-r', '--root', type=str, default='/N/project/Typhoon-deep-learning/output/',
                        help='Output directory')
    parser.add_argument('-b', '--wrf_base', type=str, default="/N/project/Typhoon-deep-learning/data/tc-wrf/",
                        help='Base directory to read from')
    parser.add_argument('-vl', '--var_levels', type=str, nargs='+', 
                        default=["U01", "U02", "U03", 
                                 "V01", "V02", "V03",
                                 "T01", "T02", "T03",
                                 "QVAPOR01", "QVAPOR02", "QVAPOR03",
                                 "PSFC"],
                        help="List of variable levels to extract. Example usage: ('U01' 'U02' 'V01')")
    parser.add_argument('-xew', '--train_experiment_wrf', type=str, nargs='+', 
                        default=["exp_02km_m01:exp_02km_m01", "exp_02km_m02:exp_02km_m02", 
                                 "exp_02km_m04:exp_02km_m04", "exp_02km_m05:exp_02km_m05",
                                 "exp_02km_m06:exp_02km_m06", "exp_02km_m07:exp_02km_m07", 
                                 "exp_02km_m08:exp_02km_m08", "exp_02km_m09:exp_02km_m09",
                                 "exp_02km_m10:exp_02km_m10"],
                        help='WRF experiment folders for training (inputs), form x:y')
    parser.add_argument('-tew', '--test_experiment_wrf', type=str, nargs='+', 
                        default=["exp_02km_m03:exp_02km_m03"],
                        help='WRF experiment folders for testing (targets)')
    parser.add_argument('-vew', '--val_experiment_wrf', type=str, nargs='*', default=[], 
                        help='WRF experiment folders for validation (default: empty list)')
    parser.add_argument('-xd', '--X_resolution', type=str, default='d01', 
                        help='X resolution string in filename (e.g. d01)')
    parser.add_argument('-td', '--Y_resolution', type=str, default='d01', 
                        help='Y resolution string in filename (e.g. d01)')
    parser.add_argument('-or', '--output_resolution', type=float, default=18.0,
                        help='Output resolution (km/grid point) used in computing target distances')
    return parser.parse_args()


def extract_core_variables(ds1, ds2, imsize1=(64, 64), imsize2=(64, 64),
                           output_resolution=18, var_levels=None):
    """
    Extract core variables from ds1 and compute the target array y from ds2.
    """
    if var_levels is None:
        var_levels = [('U', 1), ('U', 2), ('U', 3),
                      ('V', 1), ('V', 2), ('V', 3),
                      ('T', 1), ('T', 2), ('T', 3),
                      ('QVAPOR', 1), ('QVAPOR', 2), ('QVAPOR', 3),
                      ('PSFC', None)]
    
    # Process ds1: extract and slice variables
    mid_x1 = ds1.sizes['west_east'] // 2
    mid_y1 = ds1.sizes['south_north'] // 2
    start_x1 = mid_x1 - imsize1[0] // 2
    end_x1   = mid_x1 + imsize1[0] // 2
    start_y1 = mid_y1 - imsize1[1] // 2
    end_y1   = mid_y1 + imsize1[1] // 2

    arrays = []
    for var, lev in var_levels:
        try:
            if lev is not None and 'bottom_top' in ds1[var].dims:
                selected_data = ds1[var].isel(bottom_top=lev)
            elif lev is not None and 'lev' in ds1[var].coords:
                selected_data = ds1[var].sel(lev=lev)
            elif lev is not None and 'bottom_top_stag' in ds1[var].dims:
                selected_data = ds1[var].isel(bottom_top_stag=lev)
            else:
                selected_data = ds1[var]
        except Exception:
            selected_data = ds1[var]

        if 'south_north' in selected_data.dims:
            selected_data = selected_data.isel(south_north=slice(start_y1, end_y1))
        elif 'south_north_stag' in selected_data.dims:
            selected_data = selected_data.isel(south_north_stag=slice(start_y1, end_y1))
        if 'west_east_stag' in selected_data.dims:
            selected_data = selected_data.isel(west_east_stag=slice(start_x1, end_x1))
        elif 'west_east' in selected_data.dims:
            selected_data = selected_data.isel(west_east=slice(start_x1, end_x1))
        
        arr = np.squeeze(selected_data.values, axis=0)
        arrays.append(arr)
    
    final_result = np.stack(arrays, axis=0)
    final_result = final_result[np.newaxis, ...]  # shape: (1, channels, height, width)

    # Process ds2: compute target y
    mid_x2 = ds2.sizes['west_east'] // 2
    mid_y2 = ds2.sizes['south_north'] // 2
    start_x2 = mid_x2 - imsize2[0] // 2
    end_x2   = mid_x2 + imsize2[0] // 2
    start_y2 = mid_y2 - imsize2[1] // 2
    end_y2   = mid_y2 + imsize2[1] // 2

    u10 = ds2.U10.isel(south_north=slice(start_y2, end_y2), west_east=slice(start_x2, end_x2))
    v10 = ds2.V10.isel(south_north=slice(start_y2, end_y2), west_east=slice(start_x2, end_x2))
    wind_speed = np.sqrt(u10**2 + v10**2)
    max_wind_speed = np.max(wind_speed.values)

    psfc_ds2 = ds2.PSFC.isel(south_north=slice(start_y2, end_y2), west_east=slice(start_x2, end_x2))
    min_psfc = np.min(psfc_ds2.values)

    max_wind_loc = np.unravel_index(np.argmax(wind_speed.values, axis=None), wind_speed.shape)
    min_psfc_loc = np.unravel_index(np.argmin(psfc_ds2.values, axis=None), psfc_ds2.shape)
    dist_x = np.abs(max_wind_loc[2] - min_psfc_loc[2])
    dist_y = np.abs(max_wind_loc[1] - min_psfc_loc[1])
    distance_km = np.sqrt(dist_x**2 + dist_y**2) * output_resolution

    y = np.array([[max_wind_speed, min_psfc, distance_km]])
    return final_result, y

def parse_experiment_pairs(experiment_list):
    """
    Converts a list of strings in the format "exp_xx:exp_xx" into a list of tuples (exp_xx, exp_xx).
    If the input is [''], it returns [''] unchanged.

    :param experiment_list: List of strings in "exp_xx:exp_xx" format.
    :return: List of tuples (exp_xx, exp_xx) or [''] if input is [''].
    """
    if experiment_list == ['']:
        return experiment_list  # Return unchanged if input is ['']
    
    return [tuple(exp.split(":")) for exp in experiment_list]

def natural_sort_key(s):
    """
    Produce a sort key that sorts strings naturally.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def collect_file_pairs(exp_folder1, exp_folder2, x_res, y_res):
    """
    Given an experiment folder, collect all files whose names contain the X resolution substring.
    For each such file, generate the corresponding Y file by replacing the X resolution with the Y resolution.
    Returns a list of (x_file, y_file) tuples sorted naturally by the X file's basename.
    """
    all_files = glob.glob(os.path.join(exp_folder1, "*"))
    pairs = []
    for f in all_files:
        basename = os.path.basename(f)
        if x_res in basename:
            y_file = f.replace(x_res, y_res).replace(exp_folder1, exp_folder2)
            if os.path.exists(y_file):
                pairs.append((f, y_file))
            else:
                print(f"Warning: Corresponding Y file not found for {f} (expected {y_file})")
    pairs.sort(key=lambda pair: natural_sort_key(os.path.basename(pair[0])))
    return pairs


def process_experiments(exp_list, base_path, imsize_x, imsize_y, root, x_res, y_res,
                        var_levels, output_resolution=18):
    """
    Process a list of experiment folders. For each folder, collect file pairs by looking for the X resolution
    substring in the filename, then generate the corresponding Y file by replacing it with the Y resolution.
    Process each file pair and save the concatenated results using the original folder name.
    """
    for exp in exp_list:
        exp_folder1 = os.path.join(base_path, exp[0])
        exp_folder2 = os.path.join(base_path, exp[0])
        file_pairs = collect_file_pairs(exp_folder1, exp_folder2, x_res, y_res)
        if not file_pairs:
            print(f"No file pairs found in {exp} for X resolution '{x_res}' and Y resolution '{y_res}'.")
            continue
        
        results = []
        ys = []
        for x_file, y_file in file_pairs:
            ds1 = xr.open_dataset(x_file)
            ds2 = xr.open_dataset(y_file)
            
            # Convert var_levels strings (e.g. "U01") to tuples: ("U", 1) or ("PSFC", None)
            levels = [(var[:-2], int(var[-2:])) if var[-2:].isdigit() else (var, None)
                      for var in var_levels]
            result, y_val = extract_core_variables(ds1, ds2, imsize1=imsize_x, imsize2=imsize_y,
                                                    output_resolution=output_resolution,
                                                    var_levels=levels)
            # Optionally skip pairs based on target value criteria
            if y_val[0, 0] in [0, 1]:
                continue
            
            results.append(result)
            ys.append(y_val)
        
        if results:
            concatenated_result = np.concatenate(results, axis=0)
            concatenated_y = np.concatenate(ys, axis=0)
            x_filename = f"x_{x_res}_{imsize_x[0]}x{imsize_x[1]}_{exp[0]}.npy"
            y_filename = f"y_{y_res}_{imsize_y[0]}x{imsize_y[1]}_{exp[1]}.npy"
            
            np.save(os.path.join(root, x_filename), concatenated_result)
            np.save(os.path.join(root, y_filename), concatenated_y)
            np.savetxt(os.path.join(root, f"{y_filename[:-4]}.txt"), concatenated_y, fmt="%.6f")
            print(f"Saved concatenated {x_filename} and {y_filename} in {root}.")
        else:
            print(f"No valid file pairs processed for folder {exp}.")
    print("All experiment folders have been processed and saved with concatenated data.")


if __name__ == '__main__':
    args = parse_args()
    imsize_x = args.imsize_variables
    imsize_y = args.imsize_labels
    root = os.path.join(args.root, 'wrf_data')
    base_path = args.wrf_base
    var_levels = args.var_levels

    os.makedirs(root, exist_ok=True)
    
    if args.train_experiment_wrf[0]:
        print("Processing training experiments:")
        train_experiment_wrf = parse_experiment_pairs(args.train_experiment_wrf)
        process_experiments(train_experiment_wrf, base_path, imsize_x, imsize_y, root,
                            x_res=args.X_resolution, y_res=args.Y_resolution,
                            var_levels=var_levels, output_resolution=args.output_resolution)
    
    if args.test_experiment_wrf[0]:
        test_experiment_wrf = parse_experiment_pairs(args.test_experiment_wrf)
        print("Processing testing experiments:")
        process_experiments(test_experiment_wrf, base_path, imsize_x, imsize_y, root,
                            x_res=args.X_resolution, y_res=args.Y_resolution,
                            var_levels=var_levels, output_resolution=args.output_resolution)
    
    if args.val_experiment_wrf[0]:
        val_experiment_wrf = parse_experiment_pairs(args.val_experiment_wrf)
        print("Processing validation experiments:")
        process_experiments(val_experiment_wrf, base_path, imsize_x, imsize_y, root,
                            x_res=args.X_resolution, y_res=args.Y_resolution,
                            var_levels=var_levels, output_resolution=args.output_resolution)
