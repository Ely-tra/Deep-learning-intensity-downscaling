
# Workflow Description  

## Introduction
This workflow preprocesses MERRA2 datasets and WRF idealized storm simulations to generate **multi-channel NumPy arrays** and **corresponding cyclone labels**. Outputs are structured for direct use in machine learning pipelines.  


### MERRA2 Data Preprocessing  
- **Input**:  
  - MERRA2 atmospheric reanalysis data.  
  - IBTrACS historical cyclone tracks (CSV format).  
- **Output**:  
  - Variables: storm-centered multi-channel NumPy images (user-defined MERRA2 variables in original units).  
  - Labels: Tropical cyclone intensity parameters extracted from IBTrACS:  
    - `VMAX`: Maximum wind speed (**knots**)  
    - `PMIN`: Minimum sea level pressure (**millibars**)  
    - `RMW`: Radius of maximum wind (**nautical miles**)  



### WRF Idealized Data Preprocessing  
- **Input**:  
  - WRF simulation outputs (e.g., idealized tropical cyclones).  
- **Output**:  
  - Variables: storm-centered multi-channel NumPy images (user-specified WRF channels in original units).  
  - Labels extracted from:  
    - `VMAX`: Maximum wind speed (**m/s**)  
    - `PMIN`: Minimum sea level pressure (**Pa**)  
    - `RMW`: Radius of maximum wind (**km**)  



### WRF Cross-Experiment Variable-Label Mapping  
This script enables pairing variables from one experiment with labels from another. Cross-experimenting allows users to investigate complex nested-grid feedback mechanism from the WRF model.
**Example**:  
- **Input Variables**: Extracted from a WRF 18km × 18km resolution simulation.  
- **Output Labels**: Derived from a nested WRF experiment (18km outer grid with 6km/2km nested grids).




## HISTORY:
+ Jan 26, 2024: created by Khanh Luong from the CNN workflow in CK's previous TCG model
+ May 14, 2024: cross-checked and cleaned up by CK
+ Oct 19, 2024: added a job script, further test, and cleaned up by CK to be consistent with VIT model

## CONTRIBUTION:
+ Khanh Luong: IU Earth and Atmospheric Sciences (kmluong@iu.edu or mk223338@gmail.com)
+ Chanh Kieu: IU Earth and Atmospheric Sciences (ckieu@iu.edu)


# Model parameters (job_br200.sh) 


## Common input files/data path
**Location**: Line 28 of the job script  

Configure input data paths, preprocessing outputs, and model results for tropical cyclone prediction workflows.  

---

### Input Data Paths  
| Parameter       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `besttrack`     | Path to IBTrACS dataset (CSV format) containing ground truth labels (e.g., storm intensity, location). |
| `datapath`      | Directory storing raw MERRA2 data files.                                   |
| `wrf_base`      | Root folder for WRF model output files (alternative data source to MERRA2). |
| `config`        | Path to CNN configuration file (defines model architecture/hyperparameters). |


---

### Working Directory (`workdir`)  
Main directory for workflow outputs. Automatically generates these subfolders:  

| Subdirectory         | Purpose                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------|
| **`TC_domain`**      | Output from **first MERRA2 preprocessing step**:<br> - Stores storm-centered domains with *all* MERRA2 variables/levels.<br> - Organized as `TC_domain/[basin]/[year]/` (e.g., `TC_domain/NA/2017/`). |
| **`Domain_data`**    | Output from **second MERRA2 preprocessing step**:<br> - Contains subsetted variables/levels for training.<br> - Structured as `Domain_data/[experiment_name]/data/`. |
| **`wrf_data`**       | Preprocessed WRF data (resampled resolution, selected variables/levels).                 |
| **`model`**          | Stores trained model files. Names include the base `model_name`, label (e.g., `VMAX`), data source (e.g., `MERRA2`), and spatiotemporal flag. <br> Example: `hurricane_intensity_MERRA2_VMAX(_st)`. |
| **`text_report`**    | Evaluation outputs:<br> - Graphs comparing predictions vs. labels.<br> - TXT reports (`prediction` and `label` columns). |  

---

### Temporary Folder (`temporary_folder`)  
**Purpose**: Intermediate storage for unified model input files.  

#### Workflow Integration  
1. Converts preprocessed MERRA2/WRF data into a standardized format.  
2. Ensures cross-source compatibility before model ingestion.  

**NOTE**: Files here are **automatically deleted** after workflow execution. To toggle it off, comment the last line in the job script.   

---
## Experiment Naming Configuration  
**Location**: Line 38 of the job script  

Configure experiment-specific naming conventions for outputs and models.  

---

### Parameters  

| Parameter                   | Description                                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------------|
| `text_report_name`          | Name of the report comparing predictions and labels. Saved to `workdir/text_report/[name]`. Example: `vmax_validation.txt` creates `vmax_validation.txt`. |
| `experiment_identification` | Placeholder for human-readable experiment notes (no functional impact). Example: `merra2_vmax_test1`. |
| `model_name`                | Base name for saved models. Final name includes:<br> - This base name<br> _ Data source (`MERRA2`, `WRF`)<br> _ Target label (`VMAX`, `PMIN`, `RMW`)<br> - Spatiotemporal flag. <br> Example: `baseline` → `baseline_MERRA2_VMAX(_st)`. |
| `plot_unit`                 | Explicit unit for evaluation plots (no auto-detection). Match to label source:<br> - IBTrACS: `knots` (VMAX), `millibars` (PMIN)<be>, `nmile` (RMW) - WRF: `m/s` (VMAX), `Pa` (PMIN), `km` (RMW). |

---

### Usage Examples  
1. **Basic Configuration**:  
   ```bash
   text_report_name="pmin_predictions.txt"  
   model_name="hurricane_intensity"  
   plot_unit="millibars"
   ```
Generates: `workdir/text_report/pmin_predictions.txt`

Model saved as: `hurricane_intensity_MERRA2_PMIN`
Or `hurricane_intensity_MERRA2_PMIN_st` if `st_embed` is set to 1 (line 68)

---
## Common Parameters
**Location**: Line 46 of the job script  

Configure core experiment parameters and data source handling.  

---

### Core Parameters  

| Parameter       | Options                          | Description                                                                 |
|-----------------|----------------------------------|-----------------------------------------------------------------------------|
| `mode`          | `VMAX`, `PMIN`, `RMW`           | Target cyclone metric:<br>- `VMAX`: Maximum sustained wind speed<br>- `PMIN`: Minimum sea level pressure<br>- `RMW`: Radius of maximum winds |
| `data_source`   | `MERRA2`, `WRF`                 | Input data type. Automatically disables incompatible preprocessing steps.   |
| `val_pc`        | 0-100 (default: `20`)           | Percentage of training data reserved for validation (if no explicit validation set exists). |
| `test_pc`       | 0-100 (default: `10`)           | Percentage of training data reserved for testing (requires `random_split` flag at line 82). |

**`temp_id`**: Unique identifier generated for each run to prevent file conflicts, will affect files inside  **`temporary_folder`** (line 34). This parameter should not be changed.

---
## MERRA2 Parameters
**Location**: Line 65 of the job script  

Configure MERRA2-specific processing parameters for storm-centered data extraction and preprocessing.  

---

### Core Parameters  

| Parameter              | Type/Format              | Description                                                                 |
|------------------------|--------------------------|-----------------------------------------------------------------------------|
| `regions`              | Space-separated strings  | Basins to analyze: `EP` (Eastern Pacific), `NA` (North Atlantic), `WP` (Western Pacific). |
| `st_embed`             | `0` or `1`               | Toggle space-time embedding:<br>- `0`: Disabled (default)<br>- `1`: Adds time-aware layers to model input. |
| `force_rewrite`        | `0` or `1`               | File overwrite control:<br>- `0`: Skip existing files (default)<br>- `1`: Recreate all files. |
| `list_vars`            | Array of variables       | MERRA2 meteorological variables to extract. Retains original units.<br>**Example**: `("U850" "V850" "SLP750")` = zonal wind, meridional wind (850 hPa), sea-level pressure (750 hPa). |
| `windowsize_x`<br>`windowsize_y` | Degrees (integer) | Spatial extraction window size around storm center. Default: `18°×18°`. |


---
### Data Splitting Configuration  

#### Year-Based Splitting (Default)  
```bash
validation_years=(2014)  # Reserved for validation  
test_years=(2017)        # Reserved for testing  
random_split=0           # Disable percentage-based splitting  
```
-   Training data: All years  **not**  in  `validation_years`  or  `test_years`.
#### Random Percentage Splitting
```bash
random_split=1           # Enable percentage-based splits  
val_pc=20                # Line 50, 20% of training data for validation  
test_pc=10               # Line 62, 10% for testing  
```
**Warning**: Random splitting may introduce time-series autocorrelation artifacts.

---
### NaN Value Handling
```bash
nan_fill_map="0,1:0,1,2,3;4,5:4,5,6,7;8,9:8,9,10,11"  
```
Due to the nature of the system used to generate MERRA2 data, extrapolation is not performed for pressure levels greater than the surface pressure, leading to missing data points[^1]. The nan_fill_map parameter provides a mechanism to handle these missing values effectively. By default, datasets where less than 5% of the total pixels are NaN will be filled using an adaptive algorithm. While this 5% threshold is not a fixed value, it is a recommended guideline to maintain the scientific integrity of the workflow. Adjusting this value beyond the recommended limit may impact the underlying scientific principles, making it a parameter intended for advanced users. To modify this setting, an explicit input argument must be added at **line 159** in the script. This algorithm uses the referenced wind field to interpolate data in the missing grid points. The dependent fields need to be on the same pressure level as the referenced wind fields, for example, a 850mb wind field in the input data can only be used to fill 850mb fields. 

-   **Format**:  `"referenced_wind_fields: dependent_fields_to_impute;"`
    
-   **Example**:
    
    -   `0,1:0,1,2,3`  → If wind fields (indices 0,1) have NaNs, impute them and fields 2,3.
        
    -   Indices correspond to  `list_vars`  order (e.g.,  `U850=0`,  `V850=1`,  `T850=2`, etc.).
 ---

### Usage Examples

#### Example 1: North Atlantic Basin Analysis
```bash
regions="NA"  
list_vars=("U850" "V850" "SLP750")  
validation_years=(2015)  
test_years=(2018)  
```

#### Example 2: Multi-Basin with Random Splitting
```bash
regions="EP NA"  
random_split=1  
val_pc=30  
test_pc=15  
```
---
## WRF (Weather Research and Forecasting) Parameters  
**Location**: Line 88 of the job script  

Configure parameters for processing idealized storm simulations from WRF outputs.  

---

### Core Parameters  


| Parameter                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `imsize_variables`         | Spatial dimensions (width height) for input variable images. Example: `64 64` = 64×64 pixels. |
| `imsize_labels`            | Spatial dimensions to extract labels. Must align with grid resolution:<br>- Example 1: 20×20 pixels (18km grid) → 180×180 pixels (2km grid)<br>- Example 2: 60×60 pixels (6km grid)<br>- Example 3: 20×20 pixels (18km grid) |
| `VAR_LEVELS_WRF`           | WRF variables to extract. **Names must match WRF outputs**, with the last 2 digits indicating vertical levels:<br>- `U01`, `V01`: Horizontal wind components (level 1)<br>- `T01`: Temperature (level 1)<br>- `QVAPOR01`: Water vapor mixing ratio (level 1)<br>- `PSFC`: Surface pressure (no level suffix). |
| `X_resolution_wrf`<br>`Y_resolution_wrf` | Domain identifiers for nested grids:<br>- `d01` = 18km parent grid<br>- `d02` = 6km nested grid<br>- `d03` = 2km nested grid<br>Defines input (X) and label (Y) grid resolutions. |
| `output_resolution`        | Resolution (km) for RMW calculation. Matches `Y_resolution_wrf` (e.g., 18 for d01). |


---
### Experiment Mapping  
#### Training/Testing Setup  

| Parameter                  | Format & Purpose                                                                 | Example                                                                 |
|----------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `train_experiment_wrf`     | **Format**: `"exp_X:exp_Y"` pairs (variables folder:labels folder).<br>**Purpose**: Defines pairs of WRF output folders. The script extracts variables from `exp_X` and labels from `exp_Y`, using grid resolutions defined by `X/Y_resolution_wrf`. Folder names are arbitrary - actual resolutions are determined by domain identifiers (`d01`=18km, `d02`=6km, `d03`=2km). | `"nested:non_nested"` with:<br>- `X_resolution_wrf='d01'` (18km parent grid for variables)<br>- `Y_resolution_wrf='d03'` (2km nested grid for labels)<br>*Result*: Uses 18km data from "nested" folder as inputs to predict 2km labels from "non_nested" folder. |
| `test_experiment_wrf`      | **Format**: `"exp_X:exp_Y"` pairs.<br>**Purpose**: Defines test datasets using same resolution mapping logic as training pairs. | `"exp_02km_m03:exp_02km_m03"` = Test on 2km data (requires `Y_resolution_wrf='d03'`). |
| `val_experiment_wrf`       | **Format**: `"exp_X:exp_Y"` pairs (optional).<br>**Purpose**: Explicit validation datasets. Overrides `val_pc` percentage splitting if provided. | `"exp_val:exp_val"` with `Y_resolution_wrf='d02'` = 6km validation data. |




**Key Clarifications**:

1.  **Resolution Control**: Folder names (e.g., "exp_02km") are  _descriptive labels only_  - actual grid resolution is defined by:
    
    -   `X_resolution_wrf`: Domain ID for variables (`d01`-`d03`)
        
    -   `Y_resolution_wrf`: Domain ID for labels
        
2.  **Cross-Resolution Example**:
    
    -   `"coarse:fine"`  +  `X_resolution_wrf='d01'`/`Y_resolution_wrf='d03'`  = 18km→2km mapping
        
3.  **Domain-Resolution Mapping**:
```bash
d01 = 18km    # Parent grid
d02 = 6km     # First nested grid
d03 = 2km     # Second nested grid
```
---

### Usage Examples  
#### 1. Basic Training Configuration  
```bash
# Train on 18km experiments m01-m10, nested grids (in this example, is characterized by the flag _02km_), test on m03  
train_experiment_wrf=("exp_02km_m01:exp_02km_m01" "exp_02km_m02:exp_02km_m02" ...)  
test_experiment_wrf=("exp_02km_m03:exp_02km_m03")  
X_resolution_wrf='d01'  
Y_resolution_wrf='d01'  
output_resolution=18  
```
#### 2. Cross-Resolution Mapping
Use 18km input (d01), non-nested grid (in our setup, _18km means non-nested grid, and _02km means nested grid), to estimate 6km labels (d02) , nested grid.
```bash
train_experiment_wrf=("exp_18km_m01:exp_02km_m01")  
X_resolution_wrf='d01'  
Y_resolution_wrf='d02'  
```
---
## CNN Model Parameters
**Location**: Line 115 of the job script  

Configure training hyperparameters and model architecture settings.  

---

### Training Parameters  

| Parameter          | Description                                                                 | Default Value | Example        |
|--------------------|-----------------------------------------------------------------------------|---------------|----------------|
| `learning_rate`    | Initial learning rate for the Adam optimizer.                               | 0.0001        | 0.00005        |
| `batch_size`       | Number of samples per training batch. Impacts memory usage and convergence. | 256           | 128 or 512     |
| `num_epochs`       | Total number of training iterations over the entire dataset.                | 300           | 500            |
| `image_size`       | Spatial dimensions of input images. Input images will be resized to fit the CNN architecture (height × width in pixels).              | 64            | 128            |

---
# How to run the workflow?
To run the entire workflow, execute the workflow using the `job_br200.sh` script, which centralizes configuration parameters. Advanced users may adjust additional parameters within individual Python scripts, though this is generally not required.  



## Execution Steps  
1. **Job Script**: `job_br200.sh` defines configurable workflow parameters.  
2. **Preprocessing Stages**:  
   - **MERRA2**: 3 sequential preprocessing steps.  
   - **WRF**: 1 preprocessing step.  
3. **Model Pipeline**:
   - Load preprocessed data.   
   - Build model.   
   - Save results.  
---
### Workflow Control  
#### Execution Sequence (Line 21 of `job_br200.sh`)  
```bash
# Example control sequence  
merra=(1 1 1)  # MERRA2 preprocessing steps (1=enabled, 0=disabled)  
wrf=1           # WRF preprocessing (1=enabled)  
build=(1 1 1)  # Model building, data loading, result saving
```
---
### Key Rules  

#### **Mutually Exclusive Execution**  
- MERRA2 and WRF preprocessing cannot run concurrently.  
- Submit separate jobs for each dataset.  

#### **WRF Default Behavior**  
- MERRA2 preprocessing is automatically disabled when the data source is set to `WRF`, and vice versa (handled at line 49).  
---

### Modifying Workflow Steps  

#### **Example: Run Only the Final MERRA2 Step**  
Edit line 23 in `job_br200.sh`:  
```bash
merra=(0 0 1)  # Skips MERRA2 Steps 1 and 2
```
---
## MERRA2 Preprocessing Scripts  
**Location**: Line 135 of the job script  
**Control Flags**: `merra=(1 1 1)` at line 23 (1=enabled, 0=disabled)  

---

### Stage 1: Storm-Centered Domain Extraction  
**Script**: `MERRA2tc_domain.py` 
**Trigger Condition**: `merra[0]=1`  

#### What It Does  
1. **IBTrACS Processing**:  
   - Reads IBTrACS CSV to extract:  
     - Tropical cyclone timing  
     - Center coordinates (latitude/longitude)  
     - Intensity metrics (`VMAX`, `PMIN`, `RMW`)  
2. **MERRA2 Domain Extraction**:  
   - Extracts spatial domains from MERRA2 data:  
     - Centered at TC locations from IBTrACS  
     - Domain size = `windowsize_x` × `windowsize_y` degrees  
   - Preserves **all** MERRA2 variables/levels at native resolution  
   - Saves as NetCDF files with TC metadata  

#### Key Parameters  
```bash
--windowsize 18 18          # Spatial domain size (degrees)
--regions "EP NA WP"        # Basins to process
--outputpath "$workdir"     # Output directory: workdir/TC_domain/
```
#### Output Structure
```
workdir/TC_domain/
└── [basin]/[year]/         # e.g., EP/2017/
    └── TC_[datetime].nc    # NetCDF files per storm snapshot
```
 
Governed by these parameters:

-   `besttrack`,  `datapath`,  `workdir`
    
-   `windowsize_x`,  `windowsize_y`,  `regions`
---
### Stage 2: Variable-Label Pair Generation

**Script**:  `TC-extract_data_TSU.py`  
**Trigger**:  `merra[1]=1`

#### What It Does

-   From Stage 1 NetCDFs, extracts user-specified  `list_vars`  (e.g., U850, SLP750) and IBTrACS labels to form pairs of variables-labels NumPy savefiles.
    
-   Organizes data into monthly/yearly TensorFlow-ready NumPy arrays.
    

#### Output Structure
```
workdir/Domain_data/[experiment_name]/data/[year]/{variables[month].npy,labels[month].npy}  
```
Governed by these parameters:

-   `list_vars`,  `force_rewrite`
    
-   `workdir`,  `windowsize_x`,  `windowsize_y`
---
### Stage 3: NaN Value Imputation

**Script**:  `TC-CA_NaN_filling.py`  
**Trigger**:  `merra[2]=1`

#### What It Does

-   Fills missing values using wind field relationships defined in  `nan_fill_map`.
    
-   Propagates fixes to dependent variables (e.g., T850 if linked to U850/V850).
    

#### Output

-   Modified arrays in  `Domain_data/[experiment_name]/[year]/[Original_variables_filename]fixed.npy`  (NaN-free).  
    Governed by these parameters:
    
-   `nan_fill_map`,  `var_num`
    
-   `workdir`,  `windowsize_x`,  `windowsize_y`

---
## WRF Preprocessing Scripts  
**Location**: Line 168 of the job script  
**Control Flag**: `wrf=1` , at line 24 (1=enabled, 0=disabled)  

---

### WRF Data Extraction  
**Script**: `wrf_data/extractor.py`

#### What It Does  
- Processes idealized storm simulations from WRF outputs into TensorFlow-ready datasets.  
- Creates paired input-output files:  
  - **Inputs (x)**: Atmospheric variables from user-defined `VAR_LEVELS_WRF`  
  - **Labels (y)**: Cyclone metrics (`VMAX`, `PMIN`, or `RMW`)  
  
```
workdir/wrf_data/  
│ ├── x_d01_64x64_exp_02km_m01.npy # Input variables (X_resolution_wrf = d01)  
│ └── y_d03_64x64_exp_02km_m01.npy # Labels (Y_resolution_wrf = d03)  
```
**File Naming Convention**:  
- `x_{X_resolution}_{imsize_variables}_{x_experiment_name}.npy`  
- `y_{Y_resolution}_{imsize_labels}_{y_experiment_name}.npy`  

#### Governed by Parameters  
- `imsize_variables`, `imsize_labels` (input/label dimensions)  
- `X_resolution_wrf`, `Y_resolution_wrf` (grid resolutions: d01=18km, d02=6km, d03=2km)  
- `train_experiment_wrf`, `test_experiment_wrf`, `val_experiment_wrf` (experiment pairs)  
- `output_resolution` (RMW calculation basis)  

---

### Key Notes  
- **Resolution Mapping**: Folder names (e.g., `exp_02km_m01`) are arbitrary - actual resolutions are controlled by `X/Y_resolution_wrf`.  
- **Nested Grid Support**: Enables cross-resolution experiments (e.g., 18km inputs → 2km labels).  
-  **Set separation**: Although the script is governed by `train_experiment_wrf`, `test_experiment_wrf`, and `val_experiment_wrf`, it does not explicitly separate data into those sets. The separation, both for MERRA2 and WRF, is carried out in the first Builder (next section) step.

---
## Building Model and Plot Scripts
**Location**: Line 186 of the job script  
**Control Flags**: `build=(1 1 1)` (1=enabled, 0=disabled)  

---

### Stage 1: Data Preparation (`TC-universal_data_reader.py`)  
**Trigger**: `build[0]=1`  

#### What It Does  
- Organizes preprocessed MERRA2/WRF data into standardized TensorFlow datasets:  
  - `train_x`, `train_y` (training inputs/labels)  
  - `test_x`, `test_y` (testing inputs/labels)  
  - `val_x`, `val_y` (validation inputs/labels)  
  - `*_z` files if `st_embed=1` (space-time embeddings)  
- Generates temporary files with unique `temp_id` (a random sequence of 10 characters) to prevent conflicts:  
```
temporary_folder/train_x_[temp_id].npy  
temporary_folder/train_y_[temp_id].npy
```

#### Governed by Parameters  
- `data_source`, `var_num`, `st_embed`  
- `validation_years`, `test_years`, `random_split`  
- `X/Y_resolution_wrf`, `imsize_variables`, `imsize_labels`  

---

### Stage 2: Model Training (`TC-build_model.py`)  
**Trigger**: `build[1]=1`  

#### What It Does  
- Constructs CNN model from `config` file specifications  
- Trains using prepared datasets with specified hyperparameters:  
- Optimizer: Adam (`learning_rate`)  
- Batch processing (`batch_size`)  
- Training duration (`num_epochs`)  
- Saves trained model with dynamic naming:  
```
workdir/model/{model_name}_{mode}_{data_source}_{st_flag}
```

#### Governed by Parameters  
- `model_name`, `learning_rate`, `batch_size`, `num_epochs`  
- `image_size` (must match preprocessing dimensions)  
- `config` (CNN architecture definition)  

---

### Stage 3: Evaluation & Reporting (`TC-test_plot.py`)  
**Trigger**: `build[2]=1`  

#### What It Does  
1. **Model Testing**:  
 - Evaluates on test set using final trained model  
 - Calculates RMSE, MAE between predictions and labels  
2. **Visualization**:  
 - Generates two plots:  
   - Box plot: Prediction vs label distributions  
   - Scatter plot: 1:1 line with MAE envelope  
 - Saves as `workdir/text_report/fig_{full_model_name}.png`  
3. **Text Report**:  
 - Text file with prediction-label pairs, RMSE and MAE:  
   ```
   workdir/text_report/{text_report_name}  
   ```  

#### Governed by Parameters  
- `text_report_name`, `plot_unit` (axis labels)  
- `model_name`, `image_size`  

---
# Acknowledgment  

This project is funded by the U.S. National Science Foundation (NSF) under the principal investigator **Chanh Kieu**. Any dissemination of information, code, or data associated with this project must comply with NSF guidelines and regulations.  

***THE END***
[^1]: [MERRA-2 FAQ](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/FAQ/)
