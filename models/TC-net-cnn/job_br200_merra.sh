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
mode='VMAX' #VMAX PMIN RMW
st_embed=0  # Include if you want space-time embedding, otherwise leave empty
learning_rate=0.001
batch_size=256
num_epochs=100
image_size=64
model_name='CNNmodel'
validation_years=(2014)  # Specify validation years here
test_years=(2017)        # Specify test years here
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'
workdir='/N/project/Typhoon-deep-learning/output-Tri/'
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
inputpath='/N/project/Typhoon-deep-learning/output-Tri/TC_domain/'
list_vars=("U850" "V850" "T850" "RH850" "U950" "V950" "T950" "RH950" "U750" "V750" "T750" "RH750" "SLP750")
list_vars="${list_vars[@]}"
temporary_folder='/N/project/Typhoon-deep-learning/output/'
text_report_name='report.txt'
data_source='MERRA2' #Not to change
val_pc=10 #Useless for now
config='model_core/test.json'

# This is a proposed workflow
#
    parser = argparse.ArgumentParser(description='Train a Vision Transformer model for TC intensity correction.')
    parser.add_argument('-r', '--root', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Working directory path')
    parser.add_argument('-ws', '--windowsize', type=int, nargs=2, default=[19, 19], help='Window size as two integers (e.g., 19 19)')
    parser.add_argument('-vno', '--var_num', type=int, default=13, help='Number of variables')
    parser.add_argument('-st', '--st_embed', type=bool, default=False, help='Including space-time embedded')
    parser.add_argument('-vym', '--validation_year_merra', nargs='+', type=int, default=[2014], help='Year(s) taken for validation (MERRA2 dataset)')
    parser.add_argument('-tym', '--test_year_merra', nargs='+', type=int, default=[2017], help='Year(s) taken for test (MERRA2 dataset)')
    parser.add_argument('-tew', '--test_experiment_wrf', nargs='+', type=int, default=[5], help='Experiment taken for test (WRF dataset)')
    parser.add_argument('-vew', '--validation_experiment_wrf', nargs='+', type=int, default=None, help='Experiment taken for validation (WRF dataset)')
    parser.add_argument('-ss', '--data_source', type=str, default='MERRA', help='Data source to conduct experiment on')
    parser.add_argument('-dxx', '--dxx', type=str, default='d01', help='Label quality for WRF experiments, d01 is the best, d03 is the worst')
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Temporary working folder')
    return parser.parse_args()
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
    parser = argparse.ArgumentParser(description='Train a Vision Transformer model for TC intensity correction.')
    parser.add_argument('-r', '--root', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Working directory path')
    parser.add_argument('-ws', '--windowsize', type=int, nargs=2, default=[19, 19], help='Window size as two integers (e.g., 19 19)')
    parser.add_argument('-vno', '--var_num', type=int, default=13, help='Number of variables')
    parser.add_argument('-st', '--st_embed', type=bool, default=False, help='Including space-time embedded')
    parser.add_argument('-vym', '--validation_year_merra', nargs='+', type=int, default=[2014], help='Year(s) taken for validation (MERRA2 dataset)')
    parser.add_argument('-tym', '--test_year_merra', nargs='+', type=int, default=[2017], help='Year(s) taken for test (MERRA2 dataset)')
    parser.add_argument('-tew', '--test_experiment_wrf', nargs='+', type=int, default=[5], help='Experiment taken for test (WRF dataset)')
    parser.add_argument('-vew', '--validation_experiment_wrf', nargs='+', type=int, default=None, help='Experiment taken for validation (WRF dataset)')
    parser.add_argument('-ss', '--data_source', type=str, default='MERRA', help='Data source to conduct experiment on')
    parser.add_argument('-dxx', '--dxx', type=str, default='d01', help='Label quality for WRF experiments, d01 is the best, d03 is the worst')
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Temporary working folder')
    return parser.parse_args()
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
