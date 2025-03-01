#!/bin/bash

# List of configurations.
# Each element: experiment_identification im_var1 im_var2 im_lab1 im_lab2
configs=(
    "H18h18 64 64 64 64"
    "H18l18 64 64 64 64"
    "H18h02 25 25 200 200"
    "H02h02 200 200 200 200"
    "L18l18 64 64 64 64"
    "L18h18 64 64 64 64"
    "L18h02 25 25 200 200"
    "H02l18 200 200 25 25"
    "H02h18 200 200 25 25"
)

# Loop through the three modes.
for mode in VMAX PMIN RMW; do
    # Loop through each configuration in the list.
    for config in "${configs[@]}"; do
        # Parse the configuration.
        read -r exp_id im_var1 im_var2 im_lab1 im_lab2 <<< "$config"
        im_vars="$im_var1 $im_var2"
        im_labels="$im_lab1 $im_lab2"
        
        # Create a temporary file by copying the original job script.
        temp_file="temp_job.sh"
        cp job_br200.sh "$temp_file"
        
        # For modes other than VMAX, change line 21 to wrf=0.
        if [[ "$mode" != "VMAX" ]]; then
            sed -i "21s/.*/wrf=0          # Control execution for WRF related scripts/" "$temp_file"
        fi
        
        # Update line 27 with the current mode.
        sed -i "27s/.*/mode='$mode'  # Operation mode (VMAX: maximum sustained wind speed, PMIN: minimum pressure, RMW: radius of maximum winds)/" "$temp_file"
        
        # Update line 58 with the experiment identifier.
        sed -i "58s/.*/experiment_identification='$exp_id'  # Identifier for the experiment/" "$temp_file"
        
        # Update line 59 with the image size for variables.
        sed -i "59s/.*/imsize_variables=\"$im_vars\"  # Image size for variables/" "$temp_file"
        
        # Update line 60 with the image size for labels.
        sed -i "60s/.*/imsize_labels=\"$im_labels\"  # Image size for labels/" "$temp_file"
        
        # Optionally, print the changes for review.
        echo "Submitting job with mode: $mode, experiment_identification: $exp_id"
        diff job_br200.sh "$temp_file"
        
        # Submit the job.
        sbatch "$temp_file"
        
        # Delete the temporary file.
        rm "$temp_file"
        
        echo "Job submitted and temporary file removed."
        echo "--------------------------------------------------"
    done
done
