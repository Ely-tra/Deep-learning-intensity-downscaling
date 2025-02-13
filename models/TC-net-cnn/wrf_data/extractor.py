import glob
import re
import os
import numpy as np
import xarray as xr
print('Starting')
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Extract WRF data.')
    parser.add_argument('-exp_id', '--experiment_identification', type=str, default = 'H18h18', help = 'Experiment ID code')
    parser.add_argument('-iy', '--imsize_variables', type=int, nargs=2, default = [64,64], help='Image size after extraction')
    parser.add_argument('-ix', '--imsize_labels', type=int, nargs=2, default = [64,64], help='Extractor domain for y')
    parser.add_argument('-r', '--root', type=str, default = '/N/project/Typhoon-deep-learning/output/', help='Output directory')
    parser.add_argument('-b', '--wrf_base', type=str, default = "/N/project/Typhoon-deep-learning/data/tc-wrf/" , help='Read from')
    parser.add_argument('-vl', '--var_levels', type=str, nargs='+', 
                        default=["U01", "U02", "U03", 
                        "V01", "V02", "V03",
                        "T01", "T02", "T03",
                        "QVAPOR01", "QVAPOR02", "QVAPOR03",
                        "PSFC"],
                        help="List of variable levels to extract. Example usage in Bash: ('U01' 'U02' 'V01')")
    return parser.parse_args()
               
args = parse_args()
imsize_x = args.imsize_variables
imsize_y = args.imsize_labels
eid = args.experiment_identification
root = os.path.join(args.root, 'wrf_data')
base_path = args.wrf_base
var_levels = [(var[:-2], int(var[-2:])) if var[-2:].isdigit() else (var, None) for var in args.var_levels]
os.makedirs(root, exist_ok=True)
def extract_core_variables(ds1, ds2, imsize1=(64, 64), imsize2=(64, 64), output_resolution=int(eid[-2:])):
    """
    Extract core variables from ds1 and compute the target array y from ds2.

    Parameters:
        ds1 (xarray.Dataset): Dataset to extract the 13-channel final result.
        ds2 (xarray.Dataset): Dataset to compute y (e.g., wind and surface pressure metrics).
        imsize1 (tuple): Size of the extracted subset for ds1 (width, height). Default is (64, 64).
        imsize2 (tuple): Size of the extracted subset for ds2 (width, height). Default is (64, 64).
        output_resolution (float): Resolution in km per grid point. Default is 18.
    
    Returns:
        final_result (numpy.ndarray): Array of shape (1, 13, imsize1[1], imsize1[0]) extracted from ds1.
        y (numpy.ndarray): Array of shape (1, 3) computed from ds2.
    """
    # ======= Process ds1: extract final_result =======
    # Compute the center indices for ds1
    mid_x1 = ds1.sizes['west_east'] // 2
    mid_y1 = ds1.sizes['south_north'] // 2

    # Determine the start and end indices for the subset from ds1
    start_x1 = mid_x1 - imsize1[0] // 2
    end_x1   = mid_x1 + imsize1[0] // 2
    start_y1 = mid_y1 - imsize1[1] // 2
    end_y1   = mid_y1 + imsize1[1] // 2

    # Extract the core variables from ds1
    u = ds1.U.isel(
        bottom_top=slice(1, 4),
        south_north=slice(start_y1, end_y1),
        west_east_stag=slice(start_x1, end_x1)
    )
    v = ds1.V.isel(
        bottom_top=slice(1, 4),
        south_north_stag=slice(start_y1, end_y1),
        west_east=slice(start_x1, end_x1)
    )
    t = ds1.T.isel(
        bottom_top=slice(1, 4),
        south_north=slice(start_y1, end_y1),
        west_east=slice(start_x1, end_x1)
    )
    qvapor = ds1.QVAPOR.isel(
        bottom_top=slice(1, 4),
        south_north=slice(start_y1, end_y1),
        west_east=slice(start_x1, end_x1)
    )
    psfc = ds1.PSFC.isel(
        south_north=slice(start_y1, end_y1),
        west_east=slice(start_x1, end_x1)
    )
    u_reshaped = np.squeeze(u.values, axis=0) 
    v_reshaped = np.squeeze(v.values, axis=0) 
    t_reshaped = np.squeeze(t.values, axis=0)
    q_reshaped = np.squeeze(qvapor.values, axis=0)
    # Reshape and concatenate variables to form the final_result
    final_result = np.concatenate([
        u_reshaped, v_reshaped, t_reshaped, q_reshaped, psfc.values
    ], axis=0).reshape(1, 13, imsize1[1], imsize1[0])

    # ======= Process ds2: compute y =======
    # Compute the center indices for ds2
    mid_x2 = ds2.sizes['west_east'] // 2
    mid_y2 = ds2.sizes['south_north'] // 2

    # Determine the start and end indices for the subset in ds2
    start_x2 = mid_x2 - imsize2[0] // 2
    end_x2   = mid_x2 + imsize2[0] // 2
    start_y2 = mid_y2 - imsize2[1] // 2
    end_y2   = mid_y2 + imsize2[1] // 2

    # Extract U10, V10, and calculate wind speed
    u10 = ds2.U10.isel(south_north=slice(start_y2, end_y2), west_east=slice(start_x2, end_x2))
    v10 = ds2.V10.isel(south_north=slice(start_y2, end_y2), west_east=slice(start_x2, end_x2))
    wind_speed = np.sqrt(u10**2 + v10**2)
    max_wind_speed = np.max(wind_speed.values)

    # Extract PSFC and find the minimum pressure
    psfc_ds2 = ds2.PSFC.isel(south_north=slice(start_y2, end_y2), west_east=slice(start_x2, end_x2))
    min_psfc = np.min(psfc_ds2.values)

    # Calculate distance in km between max wind speed and min pressure locations
    max_wind_loc = np.unravel_index(np.argmax(wind_speed.values, axis=None), wind_speed.shape)
    min_psfc_loc = np.unravel_index(np.argmin(psfc_ds2.values, axis=None), psfc_ds2.shape)
    dist_x = np.abs(max_wind_loc[1] - min_psfc_loc[1])
    dist_y = np.abs(max_wind_loc[0] - min_psfc_loc[0])
    distance_km = np.sqrt(dist_x**2 + dist_y**2) * output_resolution

    # Create the target array y
    y = np.array([[max_wind_speed, min_psfc, distance_km]])

    return final_result, y

def extract_core_variables(ds1, ds2, imsize1=(64, 64), imsize2=(64, 64),
                           output_resolution=18, var_levels=None):
    """
    Extract core variables from ds1 and compute the target array y from ds2.

    Parameters:
        ds1 (xarray.Dataset): Dataset from which to extract the multi‐channel input.
        ds2 (xarray.Dataset): Dataset from which to compute y (e.g., wind and surface pressure metrics).
        imsize1 (tuple): (width, height) for ds1 extraction.
        imsize2 (tuple): (width, height) for ds2 extraction.
        output_resolution (float): Resolution (km/grid point).
        var_levels (list of tuples): Each tuple is (variable_name, level) where:
            - variable_name (str): Name of the variable in ds1.
            - level (int or None): The desired level for vertical selection.
              For example, for a variable with a 'bottom_top' dimension, a value of 1 selects ds1[var].isel(bottom_top=1).
              If level is None, no vertical selection is done (useful for 2D variables like PSFC).
            
            Default:
              [('U', 1), ('U', 2), ('U', 3),
               ('V', 1), ('V', 2), ('V', 3),
               ('T', 1), ('T', 2), ('T', 3),
               ('QVAPOR', 1), ('QVAPOR', 2), ('QVAPOR', 3),
               ('PSFC', None)]
    
    Returns:
        final_result (numpy.ndarray): Array of shape (1, total_channels, imsize1[1], imsize1[0])
                                      containing the concatenated extracted ds1 variables.
        y (numpy.ndarray): Array of shape (1, 3) computed from ds2.
    """
    # Set default var_levels if none provided.
    if var_levels is None:
        var_levels = [('U', 1), ('U', 2), ('U', 3),
                      ('V', 1), ('V', 2), ('V', 3),
                      ('T', 1), ('T', 2), ('T', 3),
                      ('QVAPOR', 1), ('QVAPOR', 2), ('QVAPOR', 3),
                      ('PSFC', None)]
    
    # ======= Process ds1: extract and concatenate core variables =======
    # Compute center indices for ds1 horizontal dimensions.
    mid_x1 = ds1.sizes['west_east'] // 2
    mid_y1 = ds1.sizes['south_north'] // 2
    start_x1 = mid_x1 - imsize1[0] // 2
    end_x1   = mid_x1 + imsize1[0] // 2
    start_y1 = mid_y1 - imsize1[1] // 2
    end_y1   = mid_y1 + imsize1[1] // 2

    arrays = []  # List to store extracted arrays

    for var, lev in var_levels:
        try:
            # Attempt to select the variable at the specified vertical level.
            # First, try with "bottom_top" coordinate if available.
            if lev is not None and 'bottom_top' in ds1[var].dims:
                selected_data = ds1[var].isel(bottom_top=lev)
            # Otherwise, try with a coordinate named "lev"
            elif lev is not None and 'lev' in ds1[var].coords:
                selected_data = ds1[var].sel(lev=lev)
            else:
                selected_data = ds1[var]
        except Exception as e:
            # Fallback: if any error occurs, select the variable without level selection.
            selected_data = ds1[var]

        # Now, slice horizontally based on the computed indices.
        # If the data has a 'south_north' dimension, slice it.
        if 'south_north' in selected_data.dims:
            selected_data = selected_data.isel(south_north=slice(start_y1, end_y1))
        # For the east-west direction, check for either "west_east_stag" or "west_east"
        if 'west_east_stag' in selected_data.dims:
            selected_data = selected_data.isel(west_east_stag=slice(start_x1, end_x1))
        elif 'west_east' in selected_data.dims:
            selected_data = selected_data.isel(west_east=slice(start_x1, end_x1))
        
        # Remove the time dimension if it exists (assuming it's the first dimension)
        arr = np.squeeze(selected_data.values, axis=0)
        # If the resulting array is 2D (i.e. no explicit channel dimension), add one.
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        arrays.append(arr)
    
    # Stack all extracted arrays along a new axis (channels) and add a batch dimension.
    final_result = np.stack(arrays, axis=0)  # Shape: (total_channels, height, width)
    final_result = final_result[np.newaxis, ...]  # Final shape: (1, total_channels, height, width)

    # ======= Process ds2: compute y (unchanged) =======
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
    dist_x = np.abs(max_wind_loc[1] - min_psfc_loc[1])
    dist_y = np.abs(max_wind_loc[0] - min_psfc_loc[0])
    distance_km = np.sqrt(dist_x**2 + dist_y**2) * output_resolution

    y = np.array([[max_wind_speed, min_psfc, distance_km]])

    return final_result, y
#exp_dirs = ["exp_18km_m05"]
def natural_sort_key(s):
    """
    Produce a sort key that sorts strings "naturally", i.e. splits numbers and
    converts them to integers. For example, "m2" will come before "m10".
    """
    # Split string into numeric and non-numeric parts.
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def extract_m_id(exp_dir_name):
    """
    Given an experiment directory name (e.g. "exp_02km_m01" or "exp_18km_m01"),
    extract the m-identifier (e.g. "m01").
    """
    # We assume the directory name starts with "exp_<number>km_"
    m = re.search(r"exp_\d+km_(m\d+)", exp_dir_name)
    if m:
        return m.group(1)
    # Fallback: return the whole name
    return exp_dir_name

def collect_files(pattern, file_subpattern):
    """
    Given a pattern like "<base>/exp_XXkm_m*/<file_subpattern>",
    return a dictionary keyed by (m_id, suffix) where:
      - m_id is extracted from the experiment directory name.
      - suffix is extracted from the file name (the part after wrfout_dXX_).
    The value is the full file path.
    """
    # Get all matching files (unsorted)
    files = glob.glob(pattern)
    # Regular expression to extract the suffix after "wrfout_dXX_"
    regex = re.compile(r"wrfout_d\d\d_(.+)$")
    file_dict = {}
    for f in files:
        # Get the experiment directory name from the parent directory
        exp_dir = os.path.basename(os.path.dirname(f))
        m_id = extract_m_id(exp_dir)
        # Extract the suffix from the file name
        fname = os.path.basename(f)
        m_suffix = regex.search(fname)
        if m_suffix:
            suffix = m_suffix.group(1)
            key = (m_id, suffix)
            file_dict[key] = f
    return file_dict




def process_eid(eid, base_path, imsize_x, imsize_y, root):
    """
    Process an experiment id (eid) with format AxxByy, load matching ds1 and ds2 files 
    (ensuring that the m** (experiment id) and the suffix after wrfout_dXX_ match 
    between ds1 and ds2), extract core variables, and save concatenated results.
    
    Rules:
      - eid has format: AxxByy
         • A is either 'H' or 'L'. If A=='L', then xx must be '18'.
         • xx must be one of: '18', '06', '02'.
         • B is either 'h' or 'l'. If B=='l', then yy must be '18'.
         • yy must be one of: '18', '06', '02'.
      - For ds1: if A=='H' → experiment dirs: "exp_02km_m*", else if A=='L' → "exp_18km_m*".
      - For ds2: if B=='h' → experiment dirs: "exp_02km_m*", else if B=='l' → "exp_18km_m*".
      - The two-digit codes select the file subpattern:
         '18' -> "wrfout_d01_*"
         '06' -> "wrfout_d02_*"
         '02' -> "wrfout_d03_*"
      - Instead of matching the full experiment directory, we only match the m** portion and
        the suffix (the text after wrfout_dXX_) between ds1 and ds2 files.
    """
    import os
    import numpy as np
    import xarray as xr

    # --- Parse and validate the eid string ---
    if len(eid) != 6:
        raise ValueError(f"eid must have 6 characters. Got: {eid}")
    
    letter_A = eid[0]       # First character: H or L
    xx = eid[1:3]           # Two digits after letter A
    letter_B = eid[3]       # Fourth character: h or l
    yy = eid[4:6]           # Last two digits

    if letter_A not in ['H', 'L']:
        raise ValueError(f"First character must be 'H' or 'L'. Got: {letter_A}")
    if letter_B not in ['h', 'l']:
        raise ValueError(f"Fourth character must be 'h' or 'l'. Got: {letter_B}")
    valid_codes = ['18', '06', '02']
    if xx not in valid_codes:
        raise ValueError(f"Invalid xx value. Must be one of {valid_codes}. Got: {xx}")
    if yy not in valid_codes:
        raise ValueError(f"Invalid yy value. Must be one of {valid_codes}. Got: {yy}")

    # Enforce that L (or l) is always followed by 18 only
    if letter_A == 'L' and xx != '18':
        raise ValueError(f"For letter A 'L', xx must be '18'. Got: {xx}")
    if letter_B == 'l' and yy != '18':
        raise ValueError(f"For letter B 'l', yy must be '18'. Got: {yy}")

    # --- Determine experiment directory patterns (for file lookup) ---
    # ds1: depends on letter_A
    if letter_A == 'H':
        ds1_exp_pattern = "exp_02km_m*"
    else:
        ds1_exp_pattern = "exp_18km_m*"
    # ds2: depends on letter_B
    if letter_B == 'h':
        ds2_exp_pattern = "exp_02km_m*"
    else:
        ds2_exp_pattern = "exp_18km_m*"
    
    # --- Determine file subpatterns based on the two-digit codes ---
    subfolder_map = {
        '18': "wrfout_d01_*",
        '06': "wrfout_d02_*",
        '02': "wrfout_d03_*"
    }
    ds1_subpattern = subfolder_map[xx]
    ds2_subpattern = subfolder_map[yy]
    
    # Build full patterns to retrieve files.
    ds1_pattern = os.path.join(base_path, ds1_exp_pattern, ds1_subpattern)
    ds2_pattern = os.path.join(base_path, ds2_exp_pattern, ds2_subpattern)
    
    # --- Collect files into dictionaries keyed by (m_id, suffix) ---
    ds1_files_dict = collect_files(ds1_pattern, ds1_subpattern)
    ds2_files_dict = collect_files(ds2_pattern, ds2_subpattern)
    
    # Get the common keys (i.e. same m_id and suffix)
    common_keys = set(ds1_files_dict.keys()).intersection(ds2_files_dict.keys())
    if not common_keys:
        raise ValueError("No common experiment files found between ds1 and ds2 based on m_id and suffix.")
    
    # Sort common keys naturally (first by m_id, then by suffix)
    common_keys = sorted(common_keys, key=lambda k: (natural_sort_key(k[0]), natural_sort_key(k[1])))
    
    # --- Prepare dictionaries to accumulate extracted results by m_id ---
    results_by_m = {}
    ys_by_m = {}

    # Process each matching file pair.
    for key in common_keys:
        ds1_file = ds1_files_dict[key]
        ds2_file = ds2_files_dict[key]
        print(f"Processing {ds1_file} and {ds2_file} with matching m_id and suffix")
        ds1 = xr.open_dataset(ds1_file)
        ds2 = xr.open_dataset(ds2_file)
        
        # Extract core variables.
        result, y = extract_core_variables(ds1, ds2, imsize1=imsize_x, imsize2=imsize_y, var_levels=var_levels)
        if y[0,0]==0 or y[0,1]==0:
            print('AAAAAAAAAAA')
            continue

        # Extract m_id from key (assumed to be the first element of the key tuple).
        m_id = key[0]
        
        # Initialize the lists for this m_id if they don't exist yet.
        if m_id not in results_by_m:
            results_by_m[m_id] = []
            ys_by_m[m_id] = []
        
        # Append the extracted data to the lists.
        results_by_m[m_id].append(result)
        ys_by_m[m_id].append(y)
    
    # Concatenate and save the data for each m_id.
    for m_id in results_by_m:
        # Concatenate along axis 0 (adjust axis if needed).
        concatenated_result = np.concatenate(results_by_m[m_id], axis=0)
        concatenated_y = np.concatenate(ys_by_m[m_id], axis=0)
        
        # Build filenames including the m_id.
        x_filename = f"x_{eid}_{imsize_x[0]}x{imsize_x[1]}_{m_id}.npy"
        y_filename = f"y_{eid}_{imsize_y[0]}x{imsize_y[1]}_{m_id}.npy"
        
        # Save the concatenated arrays.
        np.save(os.path.join(root, x_filename), concatenated_result)
        np.save(os.path.join(root, y_filename), concatenated_y)
        print(f"Saved concatenated {x_filename} and {y_filename} in {root}.")

    print("All m_id directories have been processed and saved with concatenated data.")




process_eid(eid, base_path, imsize_x, imsize_y, root)

