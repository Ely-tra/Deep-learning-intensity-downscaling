#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 36:00:00
#SBATCH -J TC-ViT
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
set -x
module load PrgEnv-gnu
module load python/gpu/3.10.10
#cd /N/u/ckieu/BigRed200/model/Deep-learning-intensity-downscaling/models/TC-net-ViT
cd /N/slate/trihnguy/Deep-learning-intensity-downscaling/models/TC-net-ViT

datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'
#workdir='/N/project/Typhoon-deep-learning/output/'
workdir='/N/slate/trihnguy/output'
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
#
# Set up experiment parameters, which will be used to generate Python scripts.
#
windowsize_x=19                # winddow size in x dim around the TC center 
windowsize_y=19                # winddow size in y dim around the TC center
x_size=64                      # resized of the x-dim
y_size=64                      # resized of the y-dim
kernel_size=7                  # kernel size for CNN architecture
seasonal='NO'                  # option to do TC intensity retrieval for each month
mode='VMAX'                    # mode of training for VMAX PMIN RMW
st_embed=1                     # extra information for training, otherwise leave empty
learning_rate=0.001            # learning rate for VIT model
weight_decay=0.0001            # learning rate decay weight
batch_size=256                 # batch size
num_epochs=500                 # number of epochs
patch_size=9                  # patch size for splitting images with VIT
image_size=72                  # re-sized dimension before feeding into VIT
projection_dim=64              # embedding dimension for VIT
num_heads=4                    # number of parallel heads     
transformer_layers=8           # number of enconder layers for VIT
mlp_head_units="2048 1024"     # number of feed forward layers for each encoder block
model_name='VIT_patchsize_9'  # model name
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
data_source='MERRA2' # WRF MERRA2
work_folder='/N/slate/trihnguy/output'
test_experiment_wrf=(5)  # Define test experiments (modify as needed)
validation_experiment_wrf=() # Add section number in if needed 
validation_percentage=10
report_name='report_patchsize_9_19x19.txt'
temp_id=$(echo "$(date +%s%N)$$$BASHPID$RANDOM$(uuidgen)" | sha256sum | tr -dc 'A-Za-z0-9' | head -c10)
test_pc=10
val_pc=20
test_exp_wrf=5
random_split=1

#================
#WRF
#================
experiment_identification='H18l18'
imsize_variables="72 72"  # Image size for variables
imsize_labels="72 72"  # Image size for labels
wrf_base="/N/project/Typhoon-deep-learning/data/tc-wrf/"  # Base path for WRF data
VAR_LEVELS_WRF=("U01" "U02" "U03" "V01" "V02" "V03" "T01" "T02" "T03" "QVAPOR01" "QVAPOR02" "QVAPOR03" "PSFC")
test_exp_wrf=5
#====================

# Running the full workflow
step1=0
step2=0
step3=0
step4=0
step5=0
step6=0
step7=0

if [ "$data_source" == "MERRA2" ]; then
    step1=1
    step2=1
    step3=1
    step5=1
    step6=1
    step7=1
elif [ "$data_source" == "WRF" ]; then
    step4=1
    step5=1
    step6=1
    step7=1
fi

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
        --workdir "$workdir" \
        --windowsize "$windowsize_x" "$windowsize_y" \
        --list_vars $list_vars \
        --force_rewrite True
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
    echo "Running step 4: extractor.py ..."
    python wrf_data/extractor.py \
        -exp_id $experiment_identification\
        -ix $imsize_variables\
        -iy $imsize_labels\
        -r $workdir \
        -b $wrf_base \
        -vl "${VAR_LEVELS_WRF[@]}"
else
    echo "Step 4 extractor.py will be skipped"
fi

if [ "$step5" == "1" ]; then
    echo "Running step 4: readdata ..."
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
        -tew $test_exp_wrf \
        -r_split $random_split \
        -test_pc $test_pc \
        -val_pc $val_pc
else
    echo "Step 5 TC-universal_data_reader.py will be skipped"
fi

if [ "$step6" == "1" ]; then
    echo "Running step 4: TC-build_model.py ..."
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
        --work_folder $work_folder \
        -tid "$temp_id"
else
    echo "Step 6 TC-build_model.py will be skipped"    
fi

if [ "$step7" == "1" ]; then
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
        -tid "$temp_id"
else
    echo "Step 7 TC-test_plot.py will be skipped"
fi

#find "$workdir/temp/" -type f -name "*$temp_id*" -delete
