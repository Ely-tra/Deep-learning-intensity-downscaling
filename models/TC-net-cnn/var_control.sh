# var_control.sh
# ===============================================================================================================================================
# CONTROL SEQUENCE
# WHICH STEPS TO RUN
# ===============================================================================================================================================
merra=(1 1 1)  # Control execution for MERRA2 related scripts
wrf=1          # Control execution for WRF related scripts
build=(1 1 1)  # Control execution for Builder related scripts
# ===============================================================================================================================================
# COMMON SETTINGS
# These settings are common across different parts of the script and provide basic configuration.
# ===============================================================================================================================================
mode='VMAX'  # Operation mode (VMAX: maximum sustained wind speed, PMIN: minimum pressure, RMW: radius of maximum winds)
workdir='/N/project/Typhoon-deep-learning/output/'  # Directory for output files
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'  # Path to best track data
data_source='MERRA2'  # Data source to be used, MERRA2/WRF
val_pc=10  # Percentage of training data reserved for validation, will be used if no validation set is specified
if [ "$data_source" = "MERRA2" ]; then
    merra=(0 0 0)  # Sets all elements in the merra control array to 0
elif [ "$data_source" = "WRF" ]; then
    wrf=0  # Sets the wrf control variable to 0
fi
# ===============================================================================================================================================
# MERRA2 CONFIGURATION
# Specific configuration for handling MERRA2 dataset.
# ===============================================================================================================================================
regions="EP NA WP"  # To select basins to conduct research on
var_num=13  # Number of variables to process (solely for dynamic data naming)
st_embed=0  # Space-time embedding toggle (0 for off)
force_rewrite=False  # Force rewrite of existing files toggle
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'  # Path to raw MERRA2 data
list_vars=("U850" "V850" "T850" "RH850" "U950" "V950" "T950" "RH950" "U750" "V750" "T750" "RH750" "SLP750")  # List of meteorological variables
windowsize_x=19  # Window size along the x-axis (degree)
windowsize_y=19  # Window size along the y-axis (degree)
validation_years=(2014)  # Years used for validation
test_years=(2017)  # Years used for testing

# ===============================================================================================================================================
# WRF (Weather Research and Forecasting) CONFIGURATION
# Configuration for WRF model data handling.
# ===============================================================================================================================================
experiment_identification='H18h18'  # Identifier for the experiment
imsize_variables="64 64"  # Image size for variables
imsize_labels="64 64"  # Image size for labels
wrf_base="/N/project/Typhoon-deep-learning/data/tc-wrf/"  # Base path for WRF data

# ===============================================================================================================================================
# MODEL CONFIGURATION
# Settings for the neural network model.
# ===============================================================================================================================================
temporary_folder='/N/project/Typhoon-deep-learning/output/'  # Temporary folder for intermediate data
model_name='CNNmodel'  # Core name of the model, automatic naming is not supported, so to save multiple models, users need to assign model names manually
batch_size=256  # Batch size for training
num_epochs=100  # Number of training epochs
image_size=64  # Size of the input images for the model
config='model_core/test.json'  # Path to the model configuration file
text_report_name='report.txt'  # Filename for the text report, will be saved under workdir/text_report
