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
#windowsize='[19,19]'
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
st_embed='NO'
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'
workdir='/N/project/Typhoon-deep-learning/output-Tri/'
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
# This is a proposed workflow
#
python MERRA2tc_domain.py --datapath $datapath --workdir $workdir --besttrack $besttrack --windowsize $windowsize_x $windowsize_y --var_num $var_num
python TC-extract_data_TSU.py --workdir $workdir --windowsize $windowsize_x $windowsize_y --var_num $var_num
python TC-CA_NaN_filling_kfold.py --workdir $workdir --windowsize $windowsize_x $windowsize_y --var_num $var_num
python TC-Split_KFold.py --workdir $workdir --windowsize $windowsize_x $windowsize_y --kfold $xfold --var_num $var_num
python TC-build_model.py --mode $mode --root $workdir --windowsize $windowsize_x $windowsize_y  --var_num $var_num --kernel_size $kernel_size --x_size $x_size --y_size $y_size --xfold $xfold --st_embed $st_embed
#python TC-build_model.py PMIN $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-build_model.py RMW  $workdir $windowsize $num_var $kernel_size $x_size $y_size
python TC-test_plot.py --mode $mode --workdir $workdir --windowsize $windowsize_x $windowsize_y --var_num $var_num --kernel_size $kernel_size --x_size $x_size --y_size $y_size --xfold $xfold --st_embed $st_embed
#python TC-plot-test.py PMIN $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-plot-test.py RMW  $workdir $windowsize $num_var $kernel_size $x_size $y_size
