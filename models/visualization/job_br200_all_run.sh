#!/bin/bash -l
# ==============================================================================
# TCNN – Final plotting / report only
# This script assumes:
#   - All TCNN & LINmodel runs have finished.
#   - text_report/ and temp/ already exist under the derived workdir.
#   - temp_id matches the one used when generating ref_x_*.npy, etc.
# ==============================================================================

#SBATCH -N 1
#SBATCH -t 23:59:00
#SBATCH -J TCNN-plot-only
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G

module load python/gpu/3.10.10
cd /N/slate/kmluong/TC-net-cnn/
set -x

# ---------------- FIG2 / model index controls ----------------
i=5                        # starting index: TCNN5,6,7,8
FIG2_name_prefix="YEAR"    # goes into figure filename

# ---------------- Base paths & experiment config ----------------
# Base working directory (parent of MERRA2_00018w)
base_workdir="/N/slate/kmluong/TC-net-cnn_workdir"

# Data source + window/var info (must match training runs)
data_source="MERRA2"       # only MERRA2 layout is supported by TC_plot_and_report.py
windowsize_x=18
var_num=13                 # U850...SLP750 → 13 vars

# Experiment name part used in the subfolder, e.g. MERRA2_00018w
expName="000"

# Final workdir used by TC_plot_and_report.py:
#   /N/slate/kmluong/TC-net-cnn_workdir/MERRA2_00018w
workdir="${base_workdir}/${data_source}_${expName}${windowsize_x}w"

# temp_id must match the one used by TC-universal_data_reader + LINmodel test scripts
temp_id="sssssss"

echo "=========================================="
echo "Data source : $data_source"
echo "Workdir     : $workdir"
echo "windowsize_x: $windowsize_x"
echo "var_num     : $var_num"
echo "temp_id     : $temp_id"
echo "i (TCNN idx): $i"
echo "FIG2 prefix : $FIG2_name_prefix"
echo "=========================================="

# ---------------- Final plot & report step ONLY ----------------
python TC_plot_and_report.py \
    --workdir "$workdir" \
    --data_source "$data_source" \
    --windowsize_x "$windowsize_x" \
    --var_num "$var_num" \
    --temp_id "$temp_id" \
    --i "$i" \
    --name "$FIG2_name_prefix"
