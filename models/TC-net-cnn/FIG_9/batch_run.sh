#!/bin/bash -l
# ==============================================================================
# Experiment Setup and Configuration Script
#
# This script sets up parameters used to generate and run various Python scripts
# for our deep learning experiments. These parameters are hardcoded here to ensure
# consistency across different runs and modules.
# ==============================================================================
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J TCNN-SENS
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
module load python/gpu/3.10.10
#set -x
declare -a skipped_channels=(
  "0 1"           # UV 850
#  "4 5"           # UV 950
#  "8 9"           # UV 750
#  "0 1 4 5 8 9"   # All wind
#  "2"             # T 850
#  "6"             # T 950
#  "10"            # T 750
#  "2 6 10"        # All T
#  "3"             # RH 850
#  "7"             # RH 850
#  "11"            # RH 850
#  "3 7 11"        # RH 850
#  "12"            # SLP
)
declare -a suffixes=(
#  "dec_apr_"   # December–April
  "may_nov_"   # May–November
)
declare -a descriptions=(
  "UV 850"     # for channels 0 1
#  "UV 950"     # for channels 4 5
#  "UV 750"     # for channels 8 9
#  "All wind"   # for channels 0 1 4 5 8 9
#  "T 850"      # for channel 2
#  "T 950"      # for channel 6
#  "T 750"      # for channel 10
#  "All T"      # for channels 2 6 10
#  "RH 850"     # for channel 3
#  "RH 950"     # for channel 7
#  "RH 750"     # for channel 11
#  "All RH"     # for channels 3 7 11
#  "SLP"        # for channel 12
)
# File/Directory for input data, output, and intermediate files
workdir='/N/slate/kmluong/TC-net-cnn_workdir'   # Working directory for saving output files
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'  # Path to TC best track 
merra2data='/N/project/Typhoon-deep-learning/data/nasa-merra2/'  # Directory containing raw MERRA2 data
wrfdata="/N/project/Typhoon-deep-learning/data/tc-wrf/"  # Base directory for WRF simulation data

# Control Sequence: Toggle execution of the whole workflow
merra=(0 0 0)               # Array to control execution of MERRA2 data processing (0 = off, 1 = on)
wrf=0                       # Flag to control execution of WRF data processing (0 = off, 1 = on)
build=(1 1 1)               # Array to control execution of builder scripts (0 = off, 1 = on)

# Common Settings: Basic Configuration Options Used Across Scripts
mode='VMAX'                  # Operation mode: VMAX/PMIN/RMW/ALL
data_source='MERRA2'        # Data source selection: options include 'MERRA2' or 'WRF'
if [ "$data_source" = "MERRA2" ]; then
    wrf=0                   # When using MERRA2, disable WRF-related processing
    list_vars=("U850" "V850" "T850" "RH850" "U950" "V950" "T950" \
                "RH950" "U750" "V750" "T750" "RH750" "SLP750")
    var_num=${#list_vars[@]}
    echo "${var_num} number of vars to process: ${list_vars[@]}"
elif [ "$data_source" = "WRF" ]; then
    merra=(0 0 0)           # When using WRF, disable MERRA2-related processing
    VAR_LEVELS_WRF=("U01" "V01" "PSFC")
    var_num=${#VAR_LEVELS_WRF[@]}
    echo "${var_num} number of vars to process: ${VAR_LEVELS_WRF[@]}"
fi
temp_id=$(echo "$(date +%s%N)$$$BASHPID$RANDOM$(uuidgen)" | sha256sum | tr -dc 'A-Za-z0-9' | head -c10)
if [ $mode = "ALL" ]; then
    config='model_core/77all.json'    # Default JSON config model with 7x7 for all metrics
else
    config='../model_core/77.json'       # Default JSON config model with 7x7 for each metric
fi

# WRF Configurations
expName='H02h02'            # Unique identifier for the experiment
plot_unit='m/s'             # Unit for plotting results (e.g., wind speed in meters per second)
imsize_variables="180 180"  # Dimensions (width height) for variable images
imsize_labels="180 180"     # Dimensions (width height) for label images
X_resolution_wrf='d03'      # Horizontal resolution identifier for X-axis
Y_resolution_wrf='d03'      # Horizontal resolution identifier for Y-axis
val_experiment_wrf=''       # Placeholder for WRF validation experiment (if needed)
train_experiment_wrf=(      # experiment for x: experiment for y
                            "exp_02km_m01:exp_02km_m01"             
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
if [ $Y_resolution_wrf == "d01"  ]; then
    dx=18
elif [ $Y_resolution_wrf == "d02"  ]; then
    dx=6
elif [ $Y_resolution_wrf == "d03"  ]; then
    dx=2
fi

# MERRA2 Configurations
regions="EP NA WP"          # Basins to analyze (e.g., Eastern Pacific, North Atlantic, Western Pacific)
st_embed=0                  # Toggle for space-time embedding (0 = disabled, 1 = enabled)
force_rewrite=0             # Flag to force re-creation of files, even if they already exist, use int
windowsize_x=18             # Domain size along the x-axis to cut MERRRA global data
windowsize_y=18             # Domain size along the y-axis to cut MERRRA global data
validation_years=(2014)     # Year(s) to use for validation
test_years=(2017)           # Year(s) to use for testing
nan_fill_map="0,1:0,1,2,3;4,5:4,5,6,7;8,9:8,9,10,11"      # Map for handling NaN values in the dataset 
                                                          #(format: "referenced wind field: fields to fix;")
# Deep learning Configuration
random_split=2              # 1: use percentages (val_pc and test_pc), 0: year-based splits
test_pc=10                  # Percent of training data for testing (applies to random split scenarios)
val_pc=5                    # Percent of training data for validation (used if no explicit validation )
learning_rate=0.0001        # Learning rate for the optimizer
batch_size=256              # Number of samples per training batch
num_epochs=500              # Total number of training epochs
image_size=64               # Size of the input images for the model
echo "Data source is set to $data_source for experiment ${expName} and mode ${mode}"
echo "Working directory is $workdir"
if [ "$data_source" = "MERRA2" ]; then
    text_report_name="${mode}TCNN_${var_num}c.txt"        # text report (saved under workdir/text_report)
    model_name="TCNN_${windowsize_x}w_${var_num}c"        # Model name assigned to each experiemnts
    expName=""
elif [ "$data_source" = "WRF" ]; then
    text_report_name="${mode}${expName}_${var_num}c.txt"  # text report (saved under workdir/text_report)
    model_name="${expName}_${var_num}c"                   # Model name assigned to each experiemnts
    st_embed=0                                            # no extra space time information for WRF data 
fi
#workdir="${workdir}/${data_source}_${expName}${windowsize_x}w"
temporary_folder="${workdir}/"

# MERRA2 preprocess
if [ "${merra[0]}" -eq 1 ]; then
    python MERRA2tc_domain.py \
        --csvdataset "$besttrack" \
        --datapath "$merra2data" \
        --outputpath "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --regions $regions \
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

# WRF preprocess
if [ "$wrf" -eq 1 ]; then
    python wrf_data/extractor.py \
        -ix $imsize_variables\
        -iy $imsize_labels\
        -r $workdir \
        -b $wrfdata \
        -vl "${VAR_LEVELS_WRF[@]}" \
        -xew "${train_experiment_wrf[@]}" \
        -tew "${test_experiment_wrf[@]}" \
        -vew "${val_experiment_wrf[@]}" \
        -xd $X_resolution_wrf \
        -td $Y_resolution_wrf \
        -or $dx
fi

# Deep-learning model builder
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
  for idx in "${!skipped_channels[@]}"; do
    channels="${skipped_channels[$idx]}"
    desc="${descriptions[$idx]}"
    for suffix in "${suffixes[@]}"; do
      echo "Cutting ${channels} for ${desc} for ${suffix}" >> output_fig9.txt 2>&1
      python TC-build_model.py \
        --mode $mode \
        --root $workdir \
        --windowsize $windowsize_x $windowsize_y \
        --var_num $var_num \
        --st_embed $st_embed \
        --model_name $model_name \
        --learning_rate $learning_rate \
        --batch_size $batch_size \
        --num_epochs $num_epochs \
        --image_size $image_size \
        -temp "${temporary_folder}" \
        -ss ${data_source} \
        -cfg $config \
        -tid "$temp_id" \
        -cs $channels

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
          -u $plot_unit \
          -cs $channels \
          -ts $suffix \
          >> output_fig9.txt 2>&1
      fi

    done
  done
fi

find "$workdir/temp/" -type f -name "*$temp_id*" -delete
