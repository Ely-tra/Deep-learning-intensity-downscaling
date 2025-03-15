# Deep-learning Retrieval of Tropical Cyclone Intensity and Structure

---

## MODEL DESCRIPTION
### **Introduction**
This workflow is designed to retrieve tropical cyclone (TC) intensity and structure from gridded datasets. Two particular datasets are supported, which include NASA's MERRA2 datasets and the output from WRF model in NETCDF format. The workflow is based on CNN architectures, which can be used for either TC intensity forecasting from numerical model outputs or TC intensity downscaling from climate projection outputs.

### **History**
+ Jan 26, 2024: created by Khanh Luong from the CNN workflow in CK's previous TCG model
+ May 14, 2024: cross-checked and cleaned up by CK.
+ Oct 19, 2024: added a job script, further test, and cleaned up by CK to be consistent with VIT model.
+ Mar 11, 2025: finished/validated a new workflow that include also WRF idealized experiments.
+ Mar 13, 2025: updated by KL and CK to be consistent with TCNN (V1.0) on Zenodo repository.

### **Contributors**
+ Khanh Luong: IU Earth and Atmospheric Sciences (kmluong@iu.edu or mk223338@gmail.com)
+ Chanh Kieu: IU Earth and Atmospheric Sciences (ckieu@iu.edu)
 
---

## MODEL PARAMETERS 
The whole workflow for this system is contained in a bash Shell job script `job_br200.sh`. In this script, the workflow is divided into several sections that do the data preprocessing, model training, and test/analyis. In this section, we will provide all details about these components so users can make any changes tailored to their system.

### **Input data paths/output settings**
The first check of the job script defines, input data paths, preprocessing outputs, and model outputs for our tropical cyclone prediction workflows. 
 
| Parameter       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `besttrack`     | Path to IBTrACS dataset (CSV format) containing ground truth labels (e.g., storm intensity, location). |
| `datapath`      | Directory storing raw MERRA2 data files.                                   |
| `wrf_base`      | Root folder for WRF model output files (alternative data source to MERRA2). |
| `config`        | Path to CNN configuration file (defines model architecture/hyperparameters). |
| `workdir`       | Main directory for workflow outputs. |

Once these data path and input files are set, the workflow will automatically generates these subfolders under the `workdir` as follows:  

| Subdirectory         | Purpose                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------|
| `TC_domain`      | Output from first MERRA2 preprocessing step:<br> - Stores storm-centered domains with all MERRA2 variables/levels.<br> - Organized as `TC_domain/[basin]/[year]/` (e.g., `TC_domain/NA/2017/`). |
| `Domain_data`    | Output from second MERRA2 preprocessing step:<br> - Contains subsetted variables/levels for training.<br> - Structured as `Domain_data/[experiment_name]/data/`. |
| `wrf_data`       | Preprocessed WRF data (resampled resolution, selected variables/levels).                 |
| `model`          | Stores trained model files. Names include the base `model_name`, label (e.g., `VMAX`), data source (e.g., `MERRA2`), and spatiotemporal flag. <br> Example: `hurricane_intensity_MERRA2_VMAX(_st)`. |
| `text_report`    | Evaluation outputs:<br> - Graphs comparing predictions vs. labels.<br> - TXT reports (`prediction` and `label` columns). |  

Note also that the workflow will produce a temporary directory under the location `temporary_folder` that contains intermediate for unified model input files. Note that files in this location are automatically deleted after each workflow execution. To toggle it off, comment the last line in the job script.   

### **Experiment settings**

To help design properly each experiment, we provide also several variables in the second block to configure an experiment with some specific naming conventions for outputs and models as follows: 

| Parameter                   | Description                                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------------|
| `text_report_name`          | Name of the report comparing predictions and labels. Saved to `workdir/text_report/[name]`. Example: `vmax_validation.txt` creates `vmax_validation.txt`. |
| `experiment_identification` | Placeholder for human-readable experiment notes (no functional impact). Example: `merra2_vmax_test1`. |
| `model_name`                | Base name for saved models. Final name includes:<br> - This base name<br> _ Data source (`MERRA2`, `WRF`)<br> _ Target label (`VMAX`, `PMIN`, `RMW`)<br> - Spatiotemporal flag. <br> Example: `baseline` → `baseline_MERRA2_VMAX(_st)`. |
| `plot_unit`                 | Explicit unit for evaluation plots (no auto-detection). Match to label source:<br> - IBTrACS: `knots` (VMAX), `millibars` (PMIN)<be>, `nmile` (RMW) - WRF: `m/s` (VMAX), `Pa` (PMIN), `km` (RMW). |


In addition to these naming variables, users need to also configure  several core experiment parameters and data source handling as follows:  


| Parameter       | Options                          | Description                                                                 |
|-----------------|----------------------------------|-----------------------------------------------------------------------------|
| `mode`          | `VMAX`, `PMIN`, `RMW`           | Target cyclone metric:<br>- `VMAX`: Maximum sustained wind speed<br>- `PMIN`: Minimum sea level pressure<br>- `RMW`: Radius of maximum winds |
| `data_source`   | `MERRA2`, `WRF`                 | Input data type. Automatically disables incompatible preprocessing steps.   |
| `val_pc`        | 0-100 (default: `20`)           | Percentage of training data reserved for validation (if no explicit validation set exists). |
| `test_pc`       | 0-100 (default: `10`)           | Percentage of training data reserved for testing (requires `random_split` flag at line 82). |

To help run multiple experiment, we also have a variable `temp_id`, which set an unique identifier generated for each run to prevent file conflict under `temporary_folder`. This parameter should not be changed.

### **MERRA-2 parameters**
Specific for the NASA's MERRA-2 data, the workflow need several parameters for storm-centered data extraction and preprocessing as follows:  

| Parameter              | Type/Format              | Description                                                                 |
|------------------------|--------------------------|-----------------------------------------------------------------------------|
| `regions`              | Space-separated strings  | Basins to analyze: `EP` (Eastern Pacific), `NA` (North Atlantic), `WP` (Western Pacific). |
| `st_embed`             | `0` or `1`               | Toggle space-time embedding:<br>- `0`: Disabled (default)<br>- `1`: Adds time-aware layers to model input. |
| `force_rewrite`        | `0` or `1`               | File overwrite control:<br>- `0`: Skip existing files (default)<br>- `1`: Recreate all files. |
| `list_vars`            | Array of variables       | MERRA2 meteorological variables to extract. Retains original units.<br>Example: `("U850" "V850" "SLP750")` = zonal wind, meridional wind (850 hPa), sea-level pressure (750 hPa). |
| `windowsize_x`<br>`windowsize_y` | Degrees (integer) | Spatial extraction window size around storm center. Default: `18°×18°`. |

For data sampling method that is need to divide data into training and testing, we provide two different options   

- Year-Based Splitting (Default)  
```bash
validation_years=(2014)  # Reserved for validation  
test_years=(2017)        # Reserved for testing  
random_split=0           # Disable percentage-based splitting  
```

- Random Percentage Splitting
```bash
random_split=1           # Enable percentage-based splits  
val_pc=20                # Line 50, 20% of training data for validation  
test_pc=10               # Line 62, 10% for testing  
```

Note that for this option of year-based splitting, all years  **not**  in  `validation_years`  or  `test_years` will be considered to be training data. This sampling method is very suitable for TC problem because the random splitting may introduce time-series autocorrelation artifacts.

Due to the nature of the system used to generate MERRA2 data, extrapolation is not performed for pressure levels greater than the surface pressure, leading to missing data points[^1]. The nan_fill_map parameter provides a mechanism to handle these missing values effectively. By default, datasets where less than 5% of the total pixels are NaN will be filled using an adaptive algorithm. While this 5% threshold is not a fixed value, it is a recommended guideline to maintain the scientific integrity of the workflow. Adjusting this value beyond the recommended limit may impact the underlying scientific principles, making it a parameter intended for advanced users. To modify this setting, an explicit input argument must be added at **line 159** in the script. This algorithm uses the referenced wind field to interpolate data in the missing grid points. The dependent fields need to be on the same pressure level as the referenced wind fields, for example, a 850mb wind field in the input data can only be used to fill 850mb fields. 

An example of thie NaN data filling is given as
```bash
nan_fill_map="0,1:0,1,2,3;4,5:4,5,6,7;8,9:8,9,10,11"  
```
where the format is defined as  `"referenced_wind_fields: dependent_fields_to_impute;"`. For example, `0,1:0,1,2,3` would mean if wind fields (indices 0,1) have NaNs, impute them and fields 2,3. Note that we index field according to  `list_vars`  order (e.g.,  `U850=0`,  `V850=1`,  `T850=2`, etc.).

Several usage examples for the MERRA-2 options are:

- Example 1: North Atlantic basin with 3 input channels, using 2015 for vlidation, and 2018 for testing, and the rest for training:
```bash
regions="NA"  
list_vars=("U850" "V850" "SLP750")  
validation_years=(2015)  
test_years=(2018)  
```

- Example 2: Multi-basin with random splitting with 20\% for validation, 15\% for testing and the rest for training:
```bash
regions="EP NA"  
random_split=1  
val_pc=30  
test_pc=15  
```

### **WRF parameters**
This workflow also provides a support for using  WRF model output to retrieve TC intensity and structure. The key parameters for processing WRF idealized outputs are:  

| Parameter                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `imsize_variables`         | Spatial dimensions (width height) for input variable images. Example: `64 64` = 64×64 pixels. |
| `imsize_labels`            | Spatial dimensions to extract labels. Must align with grid resolution:<br>- Example 1: 20×20 pixels (18km grid) → 180×180 pixels (2km grid)<br>- Example 2: 60×60 pixels (6km grid)<br>- Example 3: 20×20 pixels (18km grid) |
| `VAR_LEVELS_WRF`           | WRF variables to extract. **Names must match WRF outputs**, with the last 2 digits indicating vertical levels:<br>- `U01`, `V01`: Horizontal wind components (level 1)<br>- `T01`: Temperature (level 1)<br>- `QVAPOR01`: Water vapor mixing ratio (level 1)<br>- `PSFC`: Surface pressure (no level suffix). |
| `X_resolution_wrf`<br>`Y_resolution_wrf` | Domain identifiers for nested grids:<br>- `d01` = 18km parent grid<br>- `d02` = 6km nested grid<br>- `d03` = 2km nested grid<br>Defines input (X) and label (Y) grid resolutions. |
| `output_resolution`        | Resolution (km) for RMW calculation. Matches `Y_resolution_wrf` (e.g., 18 for d01). |

To generate data pair for the supervised training using the WRF model, we match each set of variables from one data grid with a label derived from another grid according the following rules:    

| Parameter                  | Format & Purpose                                                                 | Example                                                                 |
|----------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `train_experiment_wrf`     | **Format**: `"exp_X:exp_Y"` pairs (variables folder:labels folder).<br>**Purpose**: Defines pairs of WRF output folders. The script extracts variables from `exp_X` and labels from `exp_Y`, using grid resolutions defined by `X/Y_resolution_wrf`. Folder names are arbitrary - actual resolutions are determined by domain identifiers (`d01`=18km, `d02`=6km, `d03`=2km). | `"nested:non_nested"` with:<br>- `X_resolution_wrf='d01'` (18km parent grid for variables)<br>- `Y_resolution_wrf='d03'` (2km nested grid for labels)<br>*Result*: Uses 18km data from "nested" folder as inputs to predict 2km labels from "non_nested" folder. |
| `test_experiment_wrf`      | **Format**: `"exp_X:exp_Y"` pairs.<br>**Purpose**: Defines test datasets using same resolution mapping logic as training pairs. | `"exp_02km_m03:exp_02km_m03"` = Test on 2km data (requires `Y_resolution_wrf='d03'`). |
| `val_experiment_wrf`       | **Format**: `"exp_X:exp_Y"` pairs (optional).<br>**Purpose**: Explicit validation datasets. Overrides `val_pc` percentage splitting if provided. | `"exp_val:exp_val"` with `Y_resolution_wrf='d02'` = 6km validation data. |

We note here several important points about the data pair (X,y) generated from the WRF output.

-  Folder names containing WRF output (e.g., "exp_02km") are  _descriptive labels only_  - actual grid resolution is defined by:
    
    + `X_resolution_wrf`: Domain ID for variables (`d01`-`d03`)
        
    + `Y_resolution_wrf`: Domain ID for labels
        
-  One can build an X data from one resolution with y data from another resolution by setting:
    
    -   `"coarse:fine"`  +  `X_resolution_wrf='d01'`/`Y_resolution_wrf='d03'`  = 18km→2km mapping
        
-  For current WRF output support, our domain supports 3 specific resolutions as follows:
```bash
d01 = 18km    # Parent grid
d02 = 6km     # First nested grid
d03 = 2km     # Second nested grid
```

Several example settings for WRF output are given below:
- Same resolution training configuration:
```bash
# Train on 18km experiments m01-m10, nested grids (in this example, is characterized by the flag _02km_), test on m03  
train_experiment_wrf=("exp_02km_m01:exp_02km_m01" "exp_02km_m02:exp_02km_m02" ...)  
test_experiment_wrf=("exp_02km_m03:exp_02km_m03")  
X_resolution_wrf='d01'  
Y_resolution_wrf='d01'  
output_resolution=18  
```
- Cross-resolution training configuration
Use 18km input (d01), non-nested grid (in our setup, _18km means non-nested grid, and _02km means nested grid), to estimate 6km labels (d02) , nested grid.
```bash
train_experiment_wrf=("exp_18km_m01:exp_02km_m01")  
X_resolution_wrf='d01'  
Y_resolution_wrf='d02'  
```

### **CNN model parameters**
For the deep-learning model based on CNN, we design model using a json file provided under the directory `model_core`. The other  hyperparameters are set as follows:   

| Parameter          | Description                                                                 | Default Value | Example        |
|--------------------|-----------------------------------------------------------------------------|---------------|----------------|
| `learning_rate`    | Initial learning rate for the Adam optimizer.                               | 0.0001        | 0.00005        |
| `batch_size`       | Number of samples per training batch. Impacts memory usage and convergence. | 256           | 128 or 512     |
| `num_epochs`       | Total number of training iterations over the entire dataset.                | 300           | 500            |
| `image_size`       | Spatial dimensions of input images. Input images will be resized to fit the CNN architecture (height × width in pixels).              | 64            | 128            |

Users can modify the model design by editing (or creating a new design) the default configuration `model_core/77.json`. Any new json file needs to be included in the job script `job_br200.sh` so it can be properly called.

---

## HOW TO RUN

After setting all parameters and experiment designs as mentioned above, users can run the entire workflow by executing the main job script `job_br200.sh`, which centralizes all configuration and parameters. For more control, users may want to read and edit additional parameters within individual Python scripts, although this is generally not required. This job script will do the following steps: 

1. setup workflow parameters.  
2. running preprocessing, either WRF or MERRA2 depending on your choice  
3. build model    
4. save results in degsinated location.   

The control for each of these steps is given by the following block of lines in the job script. Users can turn on/off any step if needed (note that MERRA2 and WRF option are mutually exclusive).  
```bash
# Example control sequence  
merra=(1 1 1)  # MERRA2 preprocessing steps (1=enabled, 0=disabled)  
wrf=1           # WRF preprocessing (1=enabled)  
build=(1 1 1)  # Model building, data loading, result saving
```

Few examples of controling the workflow are below 

- Runing only the final MERRA2 step, build a model, and plot results: 
```bash
merra=(0 0 1)  # Skips MERRA2 Steps 1 and 2
wrf=0          # No WRF preprocessing (1=enabled)  
build=(1 1 1)  # Model building, data loading, result saving
```  

- Just building a model and plot results: 
```bash
merra=(0 0 0)  # Skips MERRA2 Steps 1-3
wrf=0          # No WRF preprocessing (1=enabled)  
build=(1 1 1)  # Model building, data loading, result saving
```  

- Running WRF experiment, build a model, and plot results: 
```bash
merra=(0 0 0)  # Skips MERRA2 Steps 1-3
wrf=1          # Activate WRF preprocessing (1=enabled)  
build=(1 1 1)  # Model building, data loading, result saving
```  

---

## ACKNOWLEDGEMENT
This project is funded by the U.S. National Science Foundation (NSF, Chanh Kieu, PI). Any dissemination of information, code, or data associated with this project must comply with open source terms, NSF guidelines and regulations.
