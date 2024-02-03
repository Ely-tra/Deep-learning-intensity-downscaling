#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J tcg-CNN
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/u/ckieu/BigRed200/model/deep-learning/CNN-augmentation
#
# setting up the ML experiments
#
basin_train="EP"
basin_test="NA"
domain_size="18x18"
dataroot="/N/project/hurricane-deep-learning/data/ncep_extracted_binary_${domain_size}"
#
# loop over all lead times
#
rm -rf *conv*layer*dense*
for leadtime in 00 12 24 36 48 60 72; do
    echo "Running Stage 1, lead time ${leadtime}, training basin ${basin_train}, testing basin ${basin_test}, domain size ${domain_size}"
    python tcg_CNN_p1.py ${dataroot}/${basin_train}/${leadtime}/
   
    echo "Running Stage 2 ..."
    python tcg_CNN_p2.py ${leadtime}
 
    echo "Running Stage 3 ..."
    python tcg_CNN_p3.py ${leadtime} ${dataroot}/${basin_test}/${leadtime}/ >& out_${basin_train}2${basin_test}.txt
    tail -n 8 out_${basin_train}2${basin_test}.txt > report_${leadtime}_${basin_train}2${basin_test}_${domain_size}.txt
done

