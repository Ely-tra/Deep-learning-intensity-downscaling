#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J check-CNN
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G
module load PrgEnv-gnu
module load python/gpu/3.10.10
cd /N/u/kmluong/BigRed200/Deep-learning-intensity-downscaling/models/TC-net-cnn
#python MERRA2tc_domain.py
#python TC-extract_data.py
#python TC-CA_NaN_filling.py
#python TC-Split.py
python retrieval_model_vmax.py
#python retrieval_model_pmin.py
#python retrieval_model_rmw.py
python test_plot_vmax.py
#python test_plot_pmin.py
#python test_plot_rmw.py

