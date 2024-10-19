#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J check_CNN
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/u/ckieu/BigRed200/model/Deep-learning-intensity-downscaling/models/TC-net-cnn/
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
# runing the entire workflow
#
#python MERRA2tc_domain.py $datapath $workdir $besttrack $windowsize $num_var
#python TC-extract_data.py $workdir $windowsize $num_var
#python TC-CA_NaN_filling.py $workdir $windowsize $num_var
#python TC-Split.py $workdir $windowsize $num_var
#python TC-CNN_model.py VMAX $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-CNN_model.py PMIN $workdir $windowsize $num_var $kernel_size $x_size $y_size 
#python TC-CNN_model.py RMW  $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-plot-testdata.py VMAX $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-plot-testdata.py PMIN $workdir $windowsize $num_var $kernel_size $x_size $y_size
#python TC-plot-testdata.py RMW  $workdir $windowsize $num_var $kernel_size $x_size $y_size

