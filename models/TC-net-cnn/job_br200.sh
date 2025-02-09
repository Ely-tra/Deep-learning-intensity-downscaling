#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J check_VIT
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/slate/trihnguy/Deep-learning-intensity-downscaling/models/TC-net-ViT/
#
# Set up experiment parameters, which will be used to generate Python scripts.
# These parameters are currently hardwired in each script.
#
source var_control.sh

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
        -b $wrf_base\
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
        -ss ${data_source}
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
        --validation_year "${validation_years[@]}" \
        --test_year "${test_years[@]}" \
        -temp "${temporary_folder}" \
        -ss ${data_source}
fi

if [ "${build[2]}" -eq 1 ]; then
    python TC-test_plot.py \
        --mode $mode \
        --root $workdir \
        --windowsize $windowsize_x $windowsize_y \
        --var_num $var_num \
        --image_size $image_size \
        --st_embed $st_embed \
        --model_name $model_name \
        -temp ${temporary_folder} \
        -ss ${data_source}
fi
