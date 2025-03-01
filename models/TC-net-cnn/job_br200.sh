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

#
# Set up experiment parameters, which will be used to generate Python scripts.
# These parameters are currently hardwired in each script.
#
# ===============================================================================================================================================
# CONTROL SEQUENCE
# WHICH STEPS TO RUN
# ===============================================================================================================================================
merra=(0 0 0)  # Control execution for MERRA2 related scripts
wrf=1          # Control execution for WRF related scripts
build=(1 0 0)  # Control execution for Builder related scripts
# ===============================================================================================================================================
# COMMON SETTINGS
# These settings are common across different parts of the script and provide basic configuration.
# ===============================================================================================================================================
mode='VMAX'  # Operation mode (VMAX: maximum sustained wind speed, PMIN: minimum pressure, RMW: radius of maximum winds)
workdir='/N/slate/kmluong/TC-net-cnn_workdir/'  # Directory for output files
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'  # Path to best track data
data_source='WRF'  # Data source to be used, MERRA2/WRF
val_pc=20 # Percentage of training data reserved for validation, will be used if no validation set is specified, or MERRA2 random split is enabled
if [ "$data_source" = "MERRA2" ]; then
    wrf=0  # Sets all elements in the merra control array to 0
elif [ "$data_source" = "WRF" ]; then
    merra=(0 0 0)  # Sets the wrf control variable to 0
fi
temp_id=$(echo "$(date +%s%N)$$$BASHPID$RANDOM$(uuidgen)" | sha256sum | tr -dc 'A-Za-z0-9' | head -c10)
test_pc=10 # Percentage of training data reserved for test, will be used if MERRA2 random split is enabled
# ===============================================================================================================================================
# MERRA2 CONFIGURATION
# Specific configuration for handling MERRA2 dataset.
# ===============================================================================================================================================
regions="EP NA WP"  # To select basins to conduct research on
var_num=13  # Number of variables to process (solely for dynamic data naming)
st_embed=0  # Space-time embedding toggle (0 for off)
force_rewrite=False  # Force rewrite of existing files toggle
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'  # Path to raw MERRA2 data
list_vars=("U850" "V850" "T850" "RH850" "U950" "V950" "T950" "RH950" "U750" "V750" "T750" "RH750" "SLP750")  # List of meteorological variables
windowsize_x=18  # Window size along the x-axis (degree)
windowsize_y=18  # Window size along the y-axis (degree)
validation_years=(2014)  # Years used for validation
test_years=(2017)  # Years used for testing
random_split=1 # Use val_pc and test_pc instead of year.
# ===============================================================================================================================================
# WRF (Weather Research and Forecasting) CONFIGURATION
# Configuration for WRF model data handling.
# ===============================================================================================================================================
experiment_identification='H18h18'  # Identifier for the experiment
imsize_variables="64 64"  # Image size for variables
imsize_labels="64 64"  # Image size for labels
wrf_base="/N/project/Typhoon-deep-learning/data/tc-wrf/"  # Base path for WRF data
VAR_LEVELS_WRF=("U01" "U02" "U03" "V01" "V02" "V03" "T01" "T02" "T03" "QVAPOR01" "QVAPOR02" "QVAPOR03" "PSFC")
test_exp_wrf=5
# =============================================================================================================================================
# MODEL CONFIGURATION
# Settings for the neural network model.
# ===============================================================================================================================================
temporary_folder='/N/slate/kmluong/TC-net-cnn_workdir/'  # Temporary folder for intermediate data
model_name='H18h18'  # Core name of the model, automatic naming is not supported, so to save multiple models, users need to assign model names manually
learning_rate=0.0001
batch_size=256  # Batch size for training
num_epochs=300  # Number of training epochs
image_size=64  # Size of the input images for the model
config='model_core/77.json'  # Path to the model configuration file
text_report_name='H18h18.txt'  # Filename for the text report, will be saved under workdir/text_report

# Now the variables and settings from var_control.sh are available to use in this script
echo "Data source is set to $data_source"
echo "Working directory is $workdir"
# This is a proposed workflow
#
# ===============================================================================================================================================
# MERRA2
# ===============================================================================================================================================
if [ "${merra[0]}" -eq 1 ]; then
    python MERRA2tc_domain.py \
        --csvdataset "$besttrack" \
        --datapath "$datapath" \
        --outputpath "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --regions EP NA WP \
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
        --list_vars $list_vars \
        --force_rewrite
fi

if [ "${merra[2]}" -eq 1 ]; then
    python TC-CA_NaN_filling.py \
        --workdir "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --var_num "$var_num"
fi

# ===============================================================================================================================================
# WRF
# ===============================================================================================================================================
if [ "$wrf" -eq 1 ]; then
    python wrf_data/extractor.py \
        -exp_id $experiment_identification\
        -ix $imsize_variables\
        -iy $imsize_labels\
        -r $workdir \
        -b $wrf_base \
        -vl "${VAR_LEVELS_WRF[@]}"
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
        -tew $test_exp_wrf \
        -r_split $random_split \
        -test_pc $test_pc \
        -val_pc $val_pc \
        -wrf_eid $experiment_identification \
        -wrf_ix $imsize_variables \
        -wrf_iy $imsize_labels
        
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
        --text_report_name $text_report_name
fi

#find "$workdir/temp/" -type f -name "*$temp_id*" -delete
