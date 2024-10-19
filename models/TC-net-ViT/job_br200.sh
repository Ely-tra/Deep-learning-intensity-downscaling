#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J check_VIT
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/u/ckieu/BigRed200/model/Deep-learning-intensity-downscaling/models/TC-net-ViT/
#
# Set up experiment parameters, which will be used to generate Python scripts.
# These parameters are currently hardwired in each script.
#
windowsize='[19,19]'
num_var=13
x_size=64
y_size=64
kernel_size=7
nfold=10
seasonal='NO
extra_info='NO'
kfold='NO'
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'
workdir='/N/project/Typhoon-deep-learning/output/'
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
#
# This is an old workflow
#
python MERRA2tc_domain.py
python TC-extract_data_TSU.py
python TC-CA_NaN_filling_kfold.py
python TC-Split_KFold.py
python TC-build_model_VMAX.py
python TC-test_plot_VMAX.py
#
# This is a proposed workflow
#
#python MERRA2tc_domain.py $datapath $workdir $besttrack $windowsize $num_var
#python TC-extract_data_TSU.py $workdir $windowsize $num_var
#python TC-CA_NaN_filling_kfold.py $workdir $windowsize $num_var
#python TC-Split-KFold.py $workdir $windowsize $num_var
#python TC-build_model.py VMAX $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-build_model.py PMIN $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-build_model.py RMW  $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-plot-test.py VMAX $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-plot-test.py PMIN $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-plot-test.py RMW  $workdir $windowsize $num_var $kernel_size $x_size $y_size
