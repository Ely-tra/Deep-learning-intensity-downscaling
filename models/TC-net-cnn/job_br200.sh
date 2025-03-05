#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J check_VIT
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/u/kmluong/BigRed200/Deep-learning-intensity-downscaling/models/TC-net-cnn/

# ==============================================================================
# Experiment Setup and Configuration Script
#
# This script sets up parameters used to generate and run various Python scripts 
# for our deep learning experiments. These parameters are hardcoded here to ensure 
# consistency across different runs and modules.
# ==============================================================================

# ------------------------------------------------------------------------------
# Control Sequence: Toggle Execution of Different Script Sections
# ------------------------------------------------------------------------------
merra=(0 1 1)          # Array to control execution of MERRA2 data processing scripts (0 = off, 1 = on)
wrf=1                  # Flag to control execution of WRF data processing scripts (0 = off, 1 = on)
build=(1 0 0)          # Array to control execution of builder scripts (0 = off, 1 = on)

# ------------------------------------------------------------------------------
# File and Directory Paths: Define Locations for Data, Output, and Intermediate Files
# ------------------------------------------------------------------------------
workdir='/N/slate/kmluong/TC-net-cnn_workdir/'   # Working directory for saving output files
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'  # Path to hurricane best track dataset
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'  # Directory containing raw MERRA2 meteorological data
wrf_base="/N/project/Typhoon-deep-learning/data/tc-wrf/"  # Base directory for WRF simulation data
temporary_folder='/N/slate/kmluong/TC-net-cnn_workdir/'  # Temporary directory for storing intermediate processing files
config='model_core/77.json'  # Path to the JSON configuration file for the neural network model

# ------------------------------------------------------------------------------
# Experiment Naming: Unique Identifiers for Reports and Models
# ------------------------------------------------------------------------------
text_report_name='H18h18.txt'          # Filename for the text report (saved under workdir/text_report)
experiment_identification='H18h18'     # Unique identifier for the experiment
model_name='H18h18'                    # Model name; manually assigned to differentiate multiple models
plot_unit='m/s'                        # Unit for plotting results (e.g., wind speed in meters per second)

# ------------------------------------------------------------------------------
# Common Settings: Basic Configuration Options Used Across Scripts
# ------------------------------------------------------------------------------
mode='VMAX'          # Operation mode: VMAX (max sustained wind speed), PMIN (min pressure), or RMW (radius of max winds)
data_source='WRF'    # Data source selection: options include 'MERRA2' or 'WRF'
val_pc=20            # Percentage of training data reserved for validation (used if no explicit validation set is provided)

# Adjust control flags based on selected data source:
if [ "$data_source" = "MERRA2" ]; then
    wrf=0  # When using MERRA2, disable WRF-related processing
elif [ "$data_source" = "WRF" ]; then
    merra=(0 0 0)  # When using WRF, disable MERRA2-related processing
fi

# Create a unique temporary ID based on current time, process ID, random number, and UUID
temp_id=$(echo "$(date +%s%N)$$$BASHPID$RANDOM$(uuidgen)" | sha256sum | tr -dc 'A-Za-z0-9' | head -c10)

test_pc=10  # Percentage of training data reserved for testing (applies to random split scenarios)

# ------------------------------------------------------------------------------
# MERRA2 Configuration: Parameters for Processing MERRA2 Data
# ------------------------------------------------------------------------------
regions="EP NA WP"      # Basins to analyze (e.g., Eastern Pacific, North Atlantic, Western Pacific)
st_embed=0              # Toggle for space-time embedding (0 = disabled, 1 = enabled)
force_rewrite=0     # Flag to force re-creation of files, even if they already exist, use int

# List of meteorological variables to be processed from MERRA2
list_vars=("U850" "V850" "T850" "RH850" "U950" "V950" "T950" "RH950" "U750" "V750" "T750" "RH750" "SLP750")

# Spatial window dimensions (in degrees) for data extraction
windowsize_x=18  # Window size along the x-axis
windowsize_y=18  # Window size along the y-axis

# Define validation and testing periods based on years
validation_years=(2014)  # Year(s) to use for validation
test_years=(2017)        # Year(s) to use for testing

random_split=1  # If enabled, use percentages (val_pc and test_pc) instead of year-based splits

# Map for handling NaN values in the dataset (format: "referenced wind field: fields to fix;")
nan_fill_map="0,1:0,1,2,3;4,5:4,5,6,7;8,9:8,9,10,11"

# ------------------------------------------------------------------------------
# WRF (Weather Research and Forecasting) Configuration: Parameters for WRF Data
# ------------------------------------------------------------------------------
imsize_variables="64 64"  # Dimensions (width height) for variable images
imsize_labels="64 64"     # Dimensions (width height) for label images

# Define variable levels for WRF data processing
VAR_LEVELS_WRF=("U01" "U02" "U03" "V01" "V02" "V03" "T01" "T02" "T03" "QVAPOR01" "QVAPOR02" "QVAPOR03" "PSFC")

# Specify experiment names for training and testing in WRF data
train_experiment_wrf=(
    "exp_02km_m01:exp_02km_m01"                  # experiment for x: experiment for y
    "exp_02km_m02:exp_02km_m02"
    "exp_02km_m04:exp_02km_m04"
    "exp_02km_m05:exp_02km_m05"
    "exp_02km_m06:exp_02km_m06"
    "exp_02km_m07:exp_02km_m07"
    "exp_02km_m08:exp_02km_m08"
    "exp_02km_m09:exp_02km_m09"
    "exp_02km_m10:exp_02km_m10"
)

test_experiment_wrf=("exp_02km_m03:exp_02km_m03")
val_experiment_wrf=''     # Placeholder for WRF validation experiment (if needed)
X_resolution_wrf='d01'    # Horizontal resolution identifier for X-axis
Y_resolution_wrf='d01'    # Horizontal resolution identifier for Y-axis

# ------------------------------------------------------------------------------
# Model Configuration: Settings for the Neural Network Model Training
# ------------------------------------------------------------------------------
learning_rate=0.0001      # Learning rate for the optimizer
batch_size=256          # Number of samples per training batch
num_epochs=300          # Total number of training epochs
image_size=64           # Size of the input images for the model

# ------------------------------------------------------------------------------
# Final Setup: Display the Configuration Settings
# ------------------------------------------------------------------------------
if [[ "$data_source" == "WRF" ]]; then
    var_num=${#VAR_LEVELS_WRF[@]}  # Count elements in VAR_LEVELS_WRF array
elif [[ "$data_source" == "MERRA2" ]]; then
    var_num=${#list_vars[@]}  # Count elements in list_vars array
fi
echo "Data source is set to $data_source"
echo "Working directory is $workdir"

# End of setup script: The variables defined above are now available for subsequent processing steps.
# ===============================================================================================================================================
# MERRA2
# ===============================================================================================================================================
if [ "${merra[0]}" -eq 1 ]; then
    python MERRA2tc_domain.py \
        --csvdataset "$besttrack" \
        --datapath "$datapath" \
        --outputpath "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --regions "$regions" \
        --minlat -90.0 --maxlat 90.0 \
        --minlon -180.0 --maxlon 180.0 \
        --maxwind 10000 --minwind 0 \
        --maxpres 10000 --minpres 0 \
        --maxrmw 10000 --minrmw 0
fi

if [ "${merra[1]}" -eq 1 ]; then
    python TC-extract_data_TSU.py \
        --workdir "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --list_vars ${list_vars[@]} \
        --force_rewrite $force_rewrite
fi

if [ "${merra[2]}" -eq 1 ]; then
    python TC-CA_NaN_filling.py \
        --workdir "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --var_num "$var_num" \
        --channel_map $nan_fill_map
fi

# ===============================================================================================================================================
# WRF
# ===============================================================================================================================================
if [ "$wrf" -eq 1 ]; then
    python wrf_data/extractor.py \
        -ix $imsize_variables\
        -iy $imsize_labels\
        -r $workdir \
        -b $wrf_base \
        -vl "${VAR_LEVELS_WRF[@]}" \
        -xew "${train_experiment_wrf[@]}" \
        -tew "${test_experiment_wrf[@]}" \
        -vew "${val_experiment_wrf[@]}" \
        -xd $X_resolution_wrf \
        -td $Y_resolution_wrf
fi

# ===============================================================================================================================================
# Builder
# ===============================================================================================================================================
if [ "${build[0]}" -eq 1 ]; then
    python TC-universal_data_reader.py \
        --root $workdir \
        --windowsize $windowsize_x $windowsize_y \
        --var_num $var_num \
        --st_embed $st_embed\
        --validation_year_merra "${validation_years[@]}" \
        --test_year_merra "${test_years[@]}" \
        -temp "${temporary_folder}" \
        -ss ${data_source} \
        -tid "$temp_id" \
        -r_split $random_split \
        -test_pc $test_pc \
        -val_pc $val_pc \
        -wrf_ix $imsize_variables \
        -wrf_iy $imsize_labels \
        -xew "${train_experiment_wrf[@]}" \
        -tew "${test_experiment_wrf[@]}" \
        -vew "${val_experiment_wrf[@]}" \
        -xd $X_resolution_wrf \
        -td $Y_resolution_wrf
        
fi

if [ "${build[1]}" -eq 1 ]; then
    python TC-build_model.py \
        --mode $mode \
        --root $workdir \
        --windowsize $windowsize_x $windowsize_y \
        --var_num $var_num \
        --st_embed $st_embed\
        --model_name $model_name \
        --learning_rate $learning_rate \
        --batch_size $batch_size \
        --num_epochs $num_epochs \
        --image_size $image_size \
        -temp "${temporary_folder}" \
        -ss ${data_source} \
        -cfg $config \
        -tid "$temp_id"
fi

if [ "${build[2]}" -eq 1 ]; then
    python TC-test_plot.py \
        --mode $mode \
        --root $workdir \
        --image_size $image_size \
        --st_embed $st_embed \
        --model_name $model_name \
        -temp ${temporary_folder} \
        -ss ${data_source} \
        -tid "$temp_id" \
        --text_report_name $text_report_name \
        -u $plot_unit
fi

find "$workdir/temp/" -type f -name "*$temp_id*" -delete
