#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J tcg_VIT
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=50G
module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/u/kmluong/BigRed200/Deep-learning-intensity-downscaling/models/TC-net-ViT
python MERRA2tc_domain.py --outputpath /N/slate/kmluong/TC-net-ViT_workdir/
