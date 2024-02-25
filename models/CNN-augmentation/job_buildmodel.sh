#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J build_models
#SBATCH -p general
#SBATCH -A r00043
#SBATCH --mem=15
#SBATCH -o logw.txt
#SBATCH -e logerror.txt
module load python/gpu/3.10.10
cd /N/u/kmluong/BigRed200/Deep-learning-intensity-downscaling/models/CNN-augmentation
python CNN_build_models.py

