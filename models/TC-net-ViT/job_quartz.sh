#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 36:00:00
#SBATCH -J VITtrain
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
set -x
conda deactivate
module load python/gpu/3.10.10
cd /N/u/ckieu/BigRed200/model/Deep-learning-intensity-downscaling/models/TC-net-ViT
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'
workdir='/N/project/Typhoon-deep-learning/output/'
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
#
# Set up experiment parameters, which will be used to generate Python scripts.
#
windowsize_x=11                # winddow size in x dim around the TC center 
windowsize_y=11                # winddow size in y dim around the TC center
x_size=64                      # resized of the x-dim
y_size=64                      # resized of the y-dim
kernel_size=7                  # kernel size for CNN architecture
seasonal='NO'                  # option to do TC intensity retrieval for each month
mode='VMAX'                    # mode o training for VMAX PMIN RMW
st_embed=1                     # extra information for training, otherwise leave empty
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
model_name='VIT_Quartz'        # model name
validation_years=(2014)        # validation years to monitor/save best model, e.g. (2015 2016 ...)
test_years=(2017)              # test years to verify... this is IMPORTANT for TC intensity
xfold=10                       # NO NEED: number of folds for statistical robustness check
kfold='NO'                     # NO NEED: option to do k-fold statistical testing
list_vars=("U850" "V850" "T850" "RH850" \
           "U950" "V950" "T950" "RH950" \
           "U750" "V750" "T750" "RH750" \
           "SLP750")
var_num=${#list_vars[@]}
list_vars="${list_vars[@]}"
tcDomainPath="${workdir}/TC_domain_${model_name}/"

# Running the full workflow
step1=0
step2=0
step3=0
step4=1
step5=1
if [ "$step1" == "1" ]; then
    echo "Running step 1: MERRA2tc_domain.py ..."
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
else
    echo "Step 1: MERRA2tc_domain.py will be skipped"
fi

if [ "$step2" == "1" ]; then
    echo "Running step 2: TC-extract_data_TSU.py ..."
    python TC-extract_data_TSU.py \
        --inputpath "$tcDomainPath" \
        --workdir "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --list_vars $list_vars \
        --force_rewrite
else
    echo "Step 2 TC-extract_data_TSU.py will be skipped"
fi

if [ "$step3" == "1" ]; then
    echo "Running step 3: TC-CA_NaN_filling_kfold.py ..."
    python TC-CA_NaN_filling_kfold.py \
        --workdir "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --var_num "$var_num"
else 
    echo "Step 3 TC-CA_NaN_filling_kfold.py will be skipped"
fi

#This step is to randomize data for all years... no longer needed
#because it produces a wrong sampling for TC intensity retrieval
#python TC-Split_KFold.py --workdir $workdir --windowsize \
#$windowsize_x $windowsize_y --kfold $xfold --var_num $var_num

if [ "$step4" == "1" ]; then
    echo "Running step 4: TC-build_model_new.py ..."
    python TC-build_model_new.py \
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
else
    echo "Step 4 TC-build_model.py will be skipped"    
fi

if [ "$step5" == "1" ]; then
    echo "Running step 5: TC-test_plot.py ..."
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
else
    echo "Step 5 TC-test_plot.py will be skipped"
fi
