#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 09:59:00
#SBATCH -J ViT-ctl
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
set -x
module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/u/ckieu/BigRed200/model/TC-net-ViT/

# set up data input/output paths
workdir='/N/project/Typhoon-deep-learning/output/ViTC/'
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
wrfdata="/N/project/Typhoon-deep-learning/data/tc-wrf/"
merrapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'

# Set up experiment parameters, which will be used to generate Python scripts.
mode='VMAX'                    # VMAX/PMIN/RMW/ALL
data_source='MERRA2'           # WRF or MERRA2

# workflow control
merra=(1 1 1)                  # Array to control execution of MERRA2 processing (0 = off, 1 = on)
wrf=0                          # Flag to control execution of WRF processing (0 = off, 1 = on)
build=(1 1 1)                  # Array to control execution of builder scripts (0 = off, 1 = on)
random_split=0                 # 0: split by year, 1: split randomly
test_pc=10                     # fraction for test (only when random_split=1)
val_pc=5                       # fraction for val (only when random_split=1)

# MERRA2 configurations
regions="EP NA WP"             # Basins to analyze (e.g., Eastern Pacific, North Atlantic, Western Pacific)
validation_years=(2014)        # validation years to monitor/save best model, e.g. (2015 2016 ...)
test_years=(2017)              # test years to verify... this is IMPORTANT for TC intensity
xfold=10                       # NO NEED: number of folds for statistical robustness check
kfold='NO'                     # NO NEED: option to do k-fold statistical testing
list_vars=("U850" "V850" "T850" "RH850" \
           "U950" "V950" "T950" "RH950" \
           "U750" "V750" "T750" "RH750" \
           "U200" "V200" "T200" "SLP750")

# WRF configurations
expName='H02h02'
imsize_variables="189 189"     # Image size for x data
imsize_labels="189 189"        # Image size for y data
X_resolution_wrf='d03'         # Horizontal resolution identifier for X-axis
Y_resolution_wrf='d03'         # Horizontal resolution identifier for Y-axis
wrf_vars=("U01" "U02" "U03" "U10" "V01" "V02" \
          "V03" "V10" "T01" "T02" "T03" "T08" \
          "QVAPOR01" "QVAPOR02" "QVAPOR03" "PSFC")
val_experiment_wrf=''          # Placeholder for WRF validation experiment (if needed)
train_experiment_wrf=(         # experiment for x: experiment for y
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

# VIT model settings
st_embed=0                     # space-time embedding
windowsize_x=19                # winddow size in x dim around the TC center
windowsize_y=19                # winddow size in y dim around the TC center
x_size=72                      # resized of the x-dim
y_size=72                      # resized of the y-dim
seasonal='NO'                  # option to do TC intensity retrieval for each month
learning_rate=0.001            # learning rate for VIT model
weight_decay=0.0001            # learning rate decay weight
batch_size=256                 # batch size
num_epochs=500                 # number of epochs
patch_size=12                  # patch size for splitting images with VIT
image_size=72                  # re-sized dimension before feeding into VIT
projection_dim=64              # embedding dimension for VIT
num_heads=4                    # number of parallel heads
transformer_layers=8           # number of enconder layers for VIT
mlp_head_units="2048 1024"     # number of feed forward layers for each encoder block

# cross check workflow and parameters
if [ "$data_source" == "MERRA2" ]; then
    wrf=0
    var_num=${#list_vars[@]}
    expName="ViTC"
    echo "Running $var_num variables ${list_vars[@]}"
elif [ "$data_source" == "WRF" ]; then
    merra=(0 0 0)
    var_num=${#wrf_vars[@]}
    echo "Running $var_num variables ${wrf_vars[@]}"
    st_embed=0
fi
model_name="VIT_${mode}${expName}_${var_num}c_${windowsize_x}w"
report_name="${model_name}.txt"
temp_id=$(echo "$(date +%s%N)$$$BASHPID$RANDOM$(uuidgen)" | sha256sum | tr -dc 'A-Za-z0-9' | head -c10)
workdir="${workdir}/${model_name}/"

# MERRA 2 pre-processing
if [ "${merra[0]}" -eq 1 ]; then
    echo "Running step 1: MERRA2tc_domain.py ..."
    python MERRA2tc_domain.py \
        --csvdataset "$besttrack" \
        --datapath "$merrapath" \
        --outputpath "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --regions $regions \
        --minlat -90.0 --maxlat 90.0 \
        --minlon -180.0 --maxlon 180.0 \
        --maxwind 10000 --minwind 0 \
        --maxpres 10000 --minpres 0 \
        --maxrmw 10000 --minrmw 0
else
    echo "MERRA step 1: MERRA2tc_domain.py will be skipped"
fi

if [ "${merra[1]}" -eq 1 ]; then
    echo "Running step 2: TC-extract_data_TSU.py ..."
    python TC-extract_data_TSU.py \
        --workdir "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --list_vars ${list_vars[@]} \
        --force_rewrite True
else
    echo "MERRA step 2: TC-extract_data_TSU.py will be skipped"
fi

if [ "${merra[2]}" -eq 1 ]; then
    echo "Running step 3: TC-CA_NaN_filling_kfold.py ..."
    python TC-CA_NaN_filling_kfold.py \
        --workdir "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --var_num "$var_num"
else 
    echo "MERRA step 3: TC-CA_NaN_filling_kfold.py will be skipped"
fi

# WRF preprocessing
if [ "$wrf" -eq 1 ]; then
    echo "Running step 4: extractor.py ..."
    python wrf_data/extractor.py \
        -ix $imsize_variables\
        -iy $imsize_labels\
        -r $workdir \
        -b $wrfdata \
        -vl "${wrf_vars[@]}" \
        -xew "${train_experiment_wrf[@]}" \
        -tew "${test_experiment_wrf[@]}" \
        -vew "${val_experiment_wrf[@]}" \
        -xd $X_resolution_wrf \
        -td $Y_resolution_wrf \
        -or $dx
else
    echo "WRF step extractor.py will be skipped"
fi

# VIT model build
if [ "${build[0]}" -eq 1 ]; then
    echo "Running step 5: readdata ..."
    python TC-universal_data_reader.py \
        --root $workdir \
        --windowsize $windowsize_x $windowsize_y \
        --var_num $var_num \
        --st_embed $st_embed\
        --validation_year_merra "${validation_years[@]}" \
        --test_year_merra "${test_years[@]}" \
        -temp "${workdir}" \
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
else
    echo "Build step 1: TC-universal_data_reader.py will be skipped"
fi

if [ "${build[1]}" -eq 1 ]; then
    echo "Running step 6: TC-build_model.py ..."
    python TC-build_model.py \
        --mode $mode \
        --root $workdir \
        --windowsize $windowsize_x $windowsize_y \
        --var_num $var_num \
        --x_size $x_size \
        --y_size $y_size \
        --st_embed $st_embed \
        --model_name $model_name \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --num_epochs $num_epochs \
        --patch_size $patch_size \
        --image_size $image_size \
        --projection_dim $projection_dim \
        --num_heads $num_heads \
        --transformer_layers $transformer_layers \
        --mlp_head_units $mlp_head_units \
        --validation_year "${validation_years[@]}" \
        --test_year "${test_years[@]}" \
        --data_source $data_source \
        --work_folder $workdir \
        -tid "$temp_id"
else
    echo "Build step 2: TC-build_model.py will be skipped"    
fi

if [ "${build[2]}" -eq 1 ]; then
    echo "Running step 7: TC-test_plot.py ..."
    python TC-test_plot.py \
        --mode $mode \
        --root $workdir \
        --windowsize $windowsize_x $windowsize_y \
        --var_num $var_num \
        --image_size $image_size \
        --st_embed $st_embed \
        --model_name $model_name \
        --work_folder $workdir \
        --data_source $data_source \
        --text_report_name $report_name \
        -tid "$temp_id"
else
    echo "Build step 3: TC-test_plot.py will be skipped"
fi

#find "$workdir/temp/" -type f -name "*$temp_id*" -delete
