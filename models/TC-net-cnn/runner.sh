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
# run_configs.sh
# This script loops over a set of configurations. For each config it:
#  1. Creates a temporary copy of job_br200.sh.
#  2. Updates the identifier lines (lines 40–42) with the first token.
#  3. Replaces line 90 and 91 using the “next 2” and “last 2” numeric tokens.
#  4. Parses the id (in the form Axxbyy) and, based on xx and yy, modifies:
#       - Lines 111 and 112: the experiment strings (using different km prefixes if A or b are H/h vs L/l).
#       - Line 113: output_resolution (set to the numeric value of yy).
#  5. For the “VMAX” run (default) no extra changes are needed.
#  6. Then, after processing the first loop, line 24 (wrf flag) is changed from 1 to 0.
#  7. The loop is repeated twice more: first setting mode to PMIN and plot_unit to Pa, then mode RMW and plot_unit to km.
#  8. Each job file is submitted (sbatch), logged (appended to log.txt), and then deleted.
#
# IMPORTANT:
# - The sed commands below assume the original job_br200.sh file’s line numbers match those given.
# - You may need to adjust the sed line numbers if your file differs.
# - The mapping from id to “m_code” is as follows:
#     if xx=="18" then m01, if xx=="06" then m02, if xx=="02" then m03;
#     similarly for yy.
# - For the experiment string (train_experiment_wrf), the prefix is chosen by:
#       * for the x-part, if the first letter A is H then "exp_02km_" else (if L) "exp_18km_"
#       * for the y-part, if the fourth character (b) is h (or H) then "exp_02km_" else "exp_18km_"
#
# Set up the list of configs.
configs=(
  "H18h18 64 64 64 64"
  "H18l18 64 64 64 64"
  "H18h02 25 25 200 200"
  "H02h02 180 180 180 180"
  "L18l18 64 64 64 64"
  "L18h18 64 64 64 64"
  "L18h02 25 25 180 180"
  "H02l18 180 180 25 25"
  "H02h18 180 180 25 25"
)

logfile="log.txt"
> "$logfile"  # reset logfile

# Function to process one configuration.
# Parameters:
#   $1: the configuration string (e.g. "H18h18 64 64 64 64")
#   $2: mode value (VMAX, PMIN, or RMW)
#   $3: unit value (m/s, Pa, or km)
process_config() {
  local config_line="$1"
  local mode_value="$2"
  local unit_value="$3"

  # Parse tokens
  local id token2 token3 token4 token5
  read -r id token2 token3 token4 token5 <<< "$config_line"

  # Lines 40-42: update identifier-based values.
  # text_report_name gets id+".txt", experiment_identification and model_name get id.
  # (Assumes these are exactly on lines 40, 41, and 42.)
  
  # Parse the id: expected format Axxbyy (e.g. H18h18)
  local A="${id:0:1}"
  local xx="${id:1:2}"
  local b="${id:3:1}"
  local yy="${id:4:2}"

  # Map xx and yy to m-codes.
  local m_code_x m_code_y
  if [[ "$xx" == "18" ]]; then
    m_code_x="m01"
  elif [[ "$xx" == "06" ]]; then
    m_code_x="m02"
  elif [[ "$xx" == "02" ]]; then
    m_code_x="m03"
  else
    m_code_x="m00"
  fi

  if [[ "$yy" == "18" ]]; then
    m_code_y="m01"
  elif [[ "$yy" == "06" ]]; then
    m_code_y="m02"
  elif [[ "$yy" == "02" ]]; then
    m_code_y="m03"
  else
    m_code_y="m00"
  fi

  # Determine experiment prefixes.
  local prefix_x prefix_y
  if [[ "$A" == "H" ]]; then
    prefix_x="exp_02km_"
  else
    prefix_x="exp_18km_"
  fi

  # For the y-part, decide based on letter b (assume lowercase for l means L-type).
  if [[ "$b" == "h" || "$b" == "H" ]]; then
    prefix_y="exp_02km_"
  else
    prefix_y="exp_18km_"
  fi

  local exp_x="${prefix_x}${m_code_x}"
  local exp_y="${prefix_y}${m_code_y}"

  # For output_resolution, use the numeric value of yy (strip leading zero if any)
  local output_res
  output_res=$(echo "$yy" | sed 's/^0*//')

  # Create a temporary file from the original job file.
  local tmp_file="job_${id}_${mode_value}.sh"
  cp job_br200.sh "$tmp_file"

  # Replace lines based on our configuration.
  # Lines 40-42: update id values.
  sed -i "40s/.*/text_report_name='${id}.txt'/" "$tmp_file"
  sed -i "41s/.*/experiment_identification='${id}'/" "$tmp_file"
  sed -i "42s/.*/model_name='${id}'/" "$tmp_file"

  # Lines 90 and 91: update using the “next 2” and “last 2” numbers.
  # (Assuming here that the two numbers for line 90 come from tokens 2 and 3,
  # and for line 91 from tokens 4 and 5. Since the original format is like 'd01',
  # we use just token2 and token4.)
  sed -i "90s/.*/X_resolution_wrf='d${token2}'/" "$tmp_file"
  sed -i "91s/.*/Y_resolution_wrf='d${token4}'/" "$tmp_file"

  # Lines 111-113: update the train_experiment_wrf array and output_resolution.
  # Here we assume that line 111 is for the x-part, 112 for the y-part, and 113 for output_resolution.
  sed -i "111s/.*/train_experiment_wrf_x='${exp_x}'/" "$tmp_file"
  sed -i "112s/.*/train_experiment_wrf_y='${exp_y}'/" "$tmp_file"
  sed -i "113s/.*/output_resolution=${output_res}/" "$tmp_file"

  # For the VMAX run, we assume the file already has mode='VMAX' and plot_unit='m/s'.
  # For the PMIN and RMW runs we change:
  if [[ "$mode_value" != "VMAX" ]]; then
    # Change the mode on line 48 (assumed to be where mode is defined)
    sed -i "48s/.*/mode='${mode_value}'/" "$tmp_file"
    # Change the plot unit on line 43.
    sed -i "43s/.*/plot_unit='${unit_value}'/" "$tmp_file"
  fi

  # Submit the job and log the submission.
  sbatch "$tmp_file"
  echo "Submitted job for config: [$config_line] with id: ${id}, mode: ${mode_value}, unit: ${unit_value}, X_resolution: d${token2}, Y_resolution: d${token4}, exp_x: ${exp_x}, exp_y: ${exp_y}, output_resolution: ${output_res}" >> "$logfile"

  # Delete the temporary file.
  rm "$tmp_file"
}

# --- First loop: VMAX run (default mode, unit m/s, with wrf flag still = 1) ---
for config in "${configs[@]}"; do
  process_config "$config" "VMAX" "m/s"
done

# --- Turn off WRF processing ---
# Change line 24 in the original file from wrf=1 to wrf=0.
sed -i "24s/.*/wrf=0/" job_br200.sh
echo "Changed wrf flag to 0 in job_br200.sh" >> "$logfile"

# --- Second loop: PMIN run (mode PMIN and plot unit Pa) ---
for config in "${configs[@]}"; do
  process_config "$config" "PMIN" "Pa"
done

# --- Third loop: RMW run (mode RMW and plot unit km) ---
for config in "${configs[@]}"; do
  process_config "$config" "RMW" "km"
done

echo "All jobs submitted. See $logfile for details."
