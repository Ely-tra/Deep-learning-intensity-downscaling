# var_control.sh

# ===============================================================================================================================================
# MERRA2
# ===============================================================================================================================================
windowsize_x=19
windowsize_y=19
var_num=13
mode='VMAX' # VMAX, PMIN, RMW
st_embed=0  # Include if you want space-time embedding, otherwise leave empty
learning_rate=0.001
batch_size=256
num_epochs=100
image_size=64
model_name='CNNmodel'
validation_years=(2014)  
test_years=(2017)        
datapath='/N/project/Typhoon-deep-learning/data/nasa-merra2/'
workdir='/N/project/Typhoon-deep-learning/output-Tri/'
besttrack='/N/project/hurricane-deep-learning/data/tc/ibtracs.ALL.list.v04r00.csv'
inputpath='/N/project/Typhoon-deep-learning/output-Tri/TC_domain/'
list_vars=("U850" "V850" "T850" "RH850" "U950" "V950" "T950" "RH950" "U750" "V750" "T750" "RH750" "SLP750")
temporary_folder='/N/project/Typhoon-deep-learning/output/'
text_report_name='report.txt'
data_source='MERRA2' 
val_pc=10
config='model_core/test.json'
