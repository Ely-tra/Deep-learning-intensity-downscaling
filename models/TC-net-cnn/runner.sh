###################################################################################
#                                                                                 #
#                                                                                 #
#   __     __     ______     ______     __   __     __     __   __     ______     #
#  /\ \  _ \ \   /\  __ \   /\  == \   /\ "-.\ \   /\ \   /\ "-.\ \   /\  ___\    #
#  \ \ \/ ".\ \  \ \  __ \  \ \  __<   \ \ \-.  \  \ \ \  \ \ \-.  \  \ \ \__ \   #
#   \ \__/".~\_\  \ \_\ \_\  \ \_\ \_\  \ \_\" \_\  \ \_\  \ \_\" \_\  \ \_____\  #
#    \/_/   \/_/   \/_/\/_/   \/_/ /_/   \/_/ \/_/   \/_/   \/_/ \/_/   \/_____/  #
#                                                                                 #
# This is designed for a specific purpose, not for general use                    #
###################################################################################
#!/bin/bash
# run_experiments.sh
# This script loops through a set of configurations, modifies job_br200.sh accordingly,
# submits the job via sbatch, and logs the changes.

# Configurations: Each entry is: ID imsize_var_w imsize_var_h imsize_lab_w imsize_lab_h
configs=(
    "H18h18 64 64 64 64"
    "H18l18 64 64 64 64"
    "H18h02 25 25 180 180"
    "H02h02 180 180 180 180"
    "L18l18 64 64 64 64"
    "L18h18 64 64 64 64"
    "L18h02 25 25 200 200"
    "H02l18 200 200 25 25"
    "H02h18 200 200 25 25"
)

# Mode variants: each entry is "mode_value plot_unit_value"
modes=(
  "VMAX m/s"
  "PMIN Pa"
  "RMW km"
)

# Template file (should be the original job_br200.sh)
template="job_br200.sh"

# Log file to record what was done
logfile="log.txt"

# Function to run a sed substitution with debugging prints
run_sed_debug() {
  local pattern="$1"
  local replacement="$2"
  local file="$3"
  local description="$4"
  
  echo "----- Debug: $description -----"
  echo "Before:"
  grep -E "$pattern" "$file"
  sed -i "s/$pattern/$replacement/" "$file"
  echo "After:"
  grep -E "$replacement" "$file"
  echo "---------------------------------"
}

# Loop over each configuration
for config in "${configs[@]}"; do
    # Split the config string into variables
    read -r id imsize_var1 imsize_var2 imsize_lab1 imsize_lab2 <<< "$config"
    
    # The id is in the form Axxbyy, e.g. "H18h18"
    A="${id:0:1}"      # First character (e.g., H or L) for left side
    xx="${id:1:2}"     # Characters 2-3 (e.g., "18")
    b="${id:3:1}"      # Fourth character (e.g., h or l) for right side
    yy="${id:4:2}"     # Characters 5-6 (e.g., "18")
    
    # Determine experiment prefix based on the letters:
    # For left side: if A is L or l, use "exp_18km_", otherwise "exp_02km_"
    if [[ "$A" =~ [Ll] ]]; then
      left_prefix="exp_18km_"
    else
      left_prefix="exp_02km_"
    fi
    
    # For right side: if the 4th character is L or l, use "exp_18km_", else "exp_02km_"
    if [[ "$b" =~ [Ll] ]]; then
      right_prefix="exp_18km_"
    else
      right_prefix="exp_02km_"
    fi
    
    # For output_resolution (line 113) we take the last two digits as a number (remove leading zeros if any)
    output_res=$(echo "$yy" | sed 's/^0*//')
    
    # Loop over the three mode variants
    for mode_variant in "${modes[@]}"; do
      # Split the variant into mode and unit
      read -r mode_val unit_val <<< "$mode_variant"
      
      # Create a temporary job file from the template
      tmpfile="job_br200_${id}_${mode_val}.sh"
      cp "$template" "$tmpfile"
      
      echo "====== Processing config: ${id}, mode: ${mode_val} ======"
      
      # 1. Update text_report_name (line 40)
      run_sed_debug "^text_report_name=" "text_report_name='${id}.txt'" "$tmpfile" "Update text_report_name"
      
      # 2. Update experiment_identification (line 41)
      run_sed_debug "^experiment_identification=" "experiment_identification='${id}'" "$tmpfile" "Update experiment_identification"
      
      # 3. Update model_name (line 42)
      run_sed_debug "^model_name=" "model_name='${id}'" "$tmpfile" "Update model_name"
      
      # 4. Update mode (line 48 or similar)
      run_sed_debug "^mode=" "mode='${mode_val}'" "$tmpfile" "Update mode"
      
      # 5. Update plot_unit
      run_sed_debug "^plot_unit=" "plot_unit='${unit_val}'" "$tmpfile" "Update plot_unit"
      
      # 6. Update imsize_variables
      run_sed_debug "^imsize_variables=" "imsize_variables=\"${imsize_var1} ${imsize_var2}\"" "$tmpfile" "Update imsize_variables"
      
      # 7. Update imsize_labels
      run_sed_debug "^imsize_labels=" "imsize_labels=\"${imsize_lab1} ${imsize_lab2}\"" "$tmpfile" "Update imsize_labels"
      
      # 8. Determine X and Y resolutions based on xx and yy
      if [ "$xx" == "18" ]; then
        x_res="d01"
      elif [ "$xx" == "06" ]; then
        x_res="d02"
      elif [ "$xx" == "02" ]; then
        x_res="d03"
      else
        x_res="d01"
      fi
      
      if [ "$yy" == "18" ]; then
        y_res="d01"
      elif [ "$yy" == "06" ]; then
        y_res="d02"
      elif [ "$yy" == "02" ]; then
        y_res="d03"
      else
        y_res="d01"
      fi
      
      # 9. Update X_resolution_wrf (line 90)
      run_sed_debug "^X_resolution_wrf=" "X_resolution_wrf='${x_res}'" "$tmpfile" "Update X_resolution_wrf"
      
      # 10. Update Y_resolution_wrf (line 91)
      run_sed_debug "^Y_resolution_wrf=" "Y_resolution_wrf='${y_res}'" "$tmpfile" "Update Y_resolution_wrf"
      
      # 11. Update output_resolution (line 113)
      run_sed_debug "^output_resolution=" "output_resolution=${output_res}" "$tmpfile" "Update output_resolution"
      
      # 12. Update train_experiment_wrf and test_experiment_wrf arrays (lines 97-109 etc.)
      run_sed_debug "\"exp_02km_\\(m[0-9][0-9]\\):exp_02km_\\(m[0-9][0-9]\\)\"" "\"${left_prefix}\\1:${right_prefix}\\2\"" "$tmpfile" "Update experiment arrays"
      
      # 13. Change the WRF flag from 1 to 0 (line 24)
      run_sed_debug "^wrf=1" "wrf=0" "$tmpfile" "Update WRF flag"
      
      # Log what was done for this configuration and mode variant
      echo "Processed config: ${id}, imsize_variables: ${imsize_var1} ${imsize_var2}, imsize_labels: ${imsize_lab1} ${imsize_lab2}, X_resolution: ${x_res}, Y_resolution: ${y_res}, output_resolution: ${output_res}, train_experiment: [prefixes: left=${left_prefix} , right=${right_prefix}], mode: ${mode_val}, plot_unit: ${unit_val}" >> "$logfile"
      
      # Submit the job
      echo "Submitting job: $tmpfile"
      #sbatch "$tmpfile"
      
      # Remove the temporary job file
      rm "$tmpfile"
      
      echo "====== Finished processing ${id} for mode ${mode_val} ======"
      echo ""
    done
done
