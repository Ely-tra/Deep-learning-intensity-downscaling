#!/bin/bash

#SBATCH -J KMLWINDOWS
#SBATCH -p general
#SBATCH -o logwindows.txt
#SBATCH -e logerrorwindows.err
#SBATCH --time=10:00:00

module load python/gpu/3.10.10
cd /N/u/kmluong/BigRed200/Deep-learning-intensity-downscaling/preprocess/
python MERRA2tc_domain.py
