#!/bin/bash -l

#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J check-CNN
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G

module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/u/kmluong/BigRed200/TC-net-cnn/kfold/

# List of values to replace 'a'
values=(1 2 3 4 5 6 7 8 9 10)

# Loop through the values and run scripts with each replacement
for val in "${values[@]}"
do
    echo "Running scripts with value: $val"
    #python TC-Split_seasonal.py $val
    python retrieval_model_vmax_seasonal.py $val
    python TC-KFold.py $val > output_${val}.txt
done
