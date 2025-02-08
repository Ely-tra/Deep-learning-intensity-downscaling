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

# This is a proposed workflow
#
# ===============================================================================================================================================
# MERRA2
# ===============================================================================================================================================
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

python TC-extract_data_TSU.py \
    --inputpath "$inputpath" \
    --workdir "$workdir" \
    --windowsize "$windowsize_x" "$windowsize_y" \
    --list_vars $list_vars \
    --force_rewrite

python TC-CA_NaN_filling.py \
    --workdir "$workdir" \
    --windowsize "$windowsize_x" "$windowsize_y" \
    --var_num "$var_num"



    
# ===============================================================================================================================================
# WRF
# ===============================================================================================================================================\


python wrf_data/extractor.py \
    -exp_id \
    -ix \
    -iy \
    -r $workdir \
    -b \


# ===============================================================================================================================================
# Builder
# ===============================================================================================================================================
#python TC-Split_KFold.py --workdir $workdir --windowsize $windowsize_x $windowsize_y --kfold $xfold --var_num $var_num
python TC-universal_data_reader.py \
    --root $workdir \
    --windowsize $windowsize_x $windowsize_y \
    --var_num $var_num \
    --st_embed $st_embed\
    --validation_year_merra "${validation_years[@]}" \
    --test_year_merra "${test_years[@]}" \
    -temp "${temporary_folder}" \
    -ss ${data_source}

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
