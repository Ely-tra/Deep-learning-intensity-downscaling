#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J nwp_convlstm
#SBATCH -p gpu --gpus-per-node v100:1
#SBATCH -A r00043 
#SBATCH --mem=128G
#module load PrgEnv-gnu
module load python/gpu
cd /N/slate/ckieu/deep-learning/CNN-augmentation/
#
# setting up the ML experiments
#
basin_train="WP"
basin_test="EP"
domain_size="30x30"
dataroot="/N/project/hurricane-deep-learning/data/ncep_extracted_binary_${domain_size}"
#
# loop over all lead times
#
rm -rf *conv*layer*dense*
for leadtime in 00 12 24 36; do
    echo "Running Stage 1, lead time ${leadtime}, training basin ${basin_train}, testing basin ${basin_test}, domain size ${domain_size}"
    python tcg_CNN_p1.py ${dataroot}/${basin_train}/${leadtime}/

    echo "Running Stage 2 ..."
    python tcg_CNN_p2.py ${leadtime}

    echo "Running Stage 3 ..."
    python tcg_CNN_p3.py ${leadtime} ${dataroot}/${basin_test}/${leadtime}/ >& out_${basin_train}2${basin_test}.txt
    tail -n 8 out_${basin_train}2${basin_test}.txt > report_${leadtime}_${basin_train}2${basin_test}_${domain_size}.txt
done

