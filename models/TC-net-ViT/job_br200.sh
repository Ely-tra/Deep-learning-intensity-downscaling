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
windowsize_x=19
windowsize_y=19
var_num=13
x_size=64
y_size=64
kernel_size=7
xfold=10
seasonal='NO'
extra_info='NO'
kfold='NO'
mode='VMAX' #VMAX PMIN RMW
st_embed='--st_embed'  # Include if you want space-time embedding, otherwise leave empty
learning_rate=0.001
weight_decay=0.0001
batch_size=256
num_epochs=100
patch_size=12
image_size=72
projection_dim=64
num_heads=4
transformer_layers=8
mlp_head_units="2048 1024"
model_name='ViT_model2'
validation_years=(2014)  # Specify validation years here
test_years=(2017)        # Specify test years here
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'
workdir='/N/project/Typhoon-deep-learning/output-Tri/'
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
input_path='/N/project/Typhoon-deep-learning/output/TC_domain/'
list_vars=("U850" "V850" "T850" "RH850" "U950" "V950" "T950" "RH950" "U750" "V750" "T750" "RH750" "SLP750")
list_vars="${list_vars[@]}"
# This is a proposed workflow
#

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

python TC-CA_NaN_filling_kfold.py \
    --workdir "$workdir" \
    --windowsize "$windowsize_x" "$windowsize_y" \
    --var_num "$var_num"

#python TC-Split_KFold.py --workdir $workdir --windowsize $windowsize_x $windowsize_y --kfold $xfold --var_num $var_num

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
    --test_year "${test_years[@]}"

python TC-test_plot.py \
    --mode $mode \
    --workdir $workdir \
    --windowsize $windowsize_x $windowsize_y \
    --var_num $var_num \
    --x_size $x_size \
    --y_size $y_size \
    --st_embed $st_embed \
    --model_name $model_name \
    --validation_year "${validation_years[@]}" \
    --test_year "${test_years[@]}"

