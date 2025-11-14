#
# DESCRIPTION:
#	This script plots test data for a trained model, performs preprocessing/norm,
#	then uses the model to predict outcomes. The results are then computed RMSE
#	and MAE metrics, and visualized through box/scatter plots to compare predicted
#	values against true values.
#
# HIST: - Jan 26, 2024: created by Khanh Luong for CNN
#       - Oct 02, 2024: adapted for VIT by Tri Nguyen
#       - Oct 19, 2024: cross-checked and cleaned up by CK
#       - Oct 30, 2024: added arguments input by TN
#====================================================================================
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
#
# Define parameters and data path. Note that x_size is the input data size. By default
# is (64x64) after resized for windowsize < 26x26. For a larger wind:wown size, set it
# to 128.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Test and Plot Model Predictions for TC Intensity")
    parser.add_argument("--mode", default="VMAX", type=str, help="Mode of operation (e.g., VMAX, PMIN, RMW)")
    parser.add_argument('-r', "--root", default="/N/project/Typhoon-deep-learning/output/", type=str, 
                        help="Directory to save output data")
    parser.add_argument('--st_embed', type=int, default=0, help='Including space-time embedded')
    parser.add_argument("--model_name", default='LINmodel', type=str, help="Base of the model name")
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', 
                        help='Temporary working folder')
    parser.add_argument('-ss', '--data_source', type=str, default='MERRA2', help='Data source')
    parser.add_argument('-tid', '--temp_id', type=str)
    parser.add_argument('-u', '--unit', type=str, default='Knots', help = 'Displayed unit')

    return parser.parse_args()

args = parse_args()
mode = args.mode
workdir = args.root
st_embed = args.st_embed
model_name = args.model_name
data_source=args.data_source
work_folder=args.work_folder
temp_id=args.temp_id
unit=args.unit
model_name = f'{model_name}_{data_source}_{mode}{"_st" if st_embed else ""}'
report_directory = os.path.join(workdir, 'text_report')
os.makedirs(report_directory, exist_ok=True)
model_dir = workdir + '/model/' + model_name
temp_dir = os.path.join(work_folder, 'temp')

######################################################################################
# All fucntions below
######################################################################################
def mode_switch(mode):
    switcher = {
        'VMAX': 0,
        'PMIN': 1,
        'RMW': 2
    }
    # Return the corresponding value if mode is found, otherwise return None as default
    return switcher.get(mode, None)

def load_data(temp_dir, temp_id=temp_id):
    global test_x, test_y, test_z

    # Load mandatory test data files
    test_x = np.load(os.path.join(temp_dir, f'test_x_{temp_id}.npy'))
    test_y = np.load(os.path.join(temp_dir, f'test_y_{temp_id}.npy'))

    # Optionally load test_z if it exists
    if f'test_z_{temp_id}.npy' in os.listdir(temp_dir):
        test_z = np.load(os.path.join(temp_dir, f'test_z_{temp_id}.npy'))
    else:
        test_z = None  # Ensure test_z is defined even if it does not exist

def root_mean_squared_error(y_true, y_pred):
    """Calculate root mean squared error."""
    m = tf.keras.metrics.RootMeanSquaredError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def MAE(y_true, y_pred):
    """Calculate mean absolute error."""
    m = tf.keras.metrics.MeanAbsoluteError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def mae_for_output(index):
    """Metric function to return MAE for a specific output index."""
    def mae(y_true, y_pred):
        return tf.keras.metrics.mean_absolute_error(y_true[:, index], y_pred[:, index])
    mae.__name__ = f'mae_{index+1}'
    return mae

def rmse_for_output(index):
    """Metric function to return RMSE for a specific output index."""
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.keras.metrics.mean_squared_error(y_true[:, index], y_pred[:, index]))
    rmse.__name__ = f'rmse_{index+1}'
    return rmse

def normalize_channels(X, y):
    """Normalize the channel data for all samples in the dataset."""
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i, :, :, var].flat[np.abs(X[i, :, :, var]).argmax()]
            X[i, :, :, var] = X[i, :, :, var] / abs(maxvalue)
    print("Finish normalization...")
    return X, y
#
# MAIN CALL: Initialize dictionary to store results
#
datadict = {}
def normalize_Z(Z):
    Z[:,2] = (Z[:,2]+90) / 180
    Z[:,3] = (Z[:,3]+180) / 360
    return Z


def plotPrediction(datadict,predict,truth,pc,mode,name,unit,report_directory):
    if mode == "ALL":
        test_y = truth[:,pc]
        if pc == 0:
            myUnit = "knot"
            myMode = "VMAX"
        elif pc == 1:
            myUnit = "hPa"
            myMode = "PMIN"
        elif pc == 2:
            myUnit = "nm"
            myMode = "RMW"
    else:
        test_y = truth
        myMode = mode
        myUnit = unit

    # Calculate metrics and store results
    datadict[name + 'rmse'] = root_mean_squared_error(predict[:,pc], test_y)
    datadict[name + 'MAE'] = MAE(predict[:,pc], test_y)
    datadict[name] = predict[:,pc]

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    axs[0].boxplot([datadict[name].reshape(-1), test_y])
    axs[0].grid(True)
    axs[0].set_ylabel(myUnit, fontsize=20)
    axs[0].text(0.95, 0.05, '(a)', transform=axs[0].transAxes, fontsize=20, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].set_xticklabels(['Predicted', 'Truth'], fontsize=20)

    # Second subplot
    axs[1].scatter(test_y, datadict[name].reshape(-1))
    axs[1].grid()
    axs[1].set_xlabel('Truth', fontsize=20)
    axs[1].set_ylabel('Prediction', fontsize=20)
    axs[1].text(0.95, 0.05, '(b)', transform=axs[1].transAxes, fontsize=20, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    axs[1].plot(np.arange(min(test_y), max(test_y)), np.arange(min(test_y), max(test_y)), 'r-', alpha=0.8)
    mae = datadict[name+'MAE']
    rmse = datadict[name+'rmse']
    #axs[1].fill_between(np.arange(min(test_y), max(test_y)), 
    #                    np.arange(min(test_y), max(test_y)) + mae, 
    #                    np.arange(min(test_y), max(test_y)) - mae, 
    #                    color='red', alpha=0.3)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    
    # Legends with RMSE and MAE without markers
    custom_lines = [
                    Line2D([0], [0], color='none', marker='', label=f'RMSE: {rmse:.2f}'),
                    Line2D([0], [0], color='none', marker='', label=f'MAE: {mae:.2f}')]

    #axs[1].legend(custom_lines, [f'RMSE: {rmse:.2f}', f'MAE: {mae:.2f}'], fontsize=12, handlelength=0)

    figPath = f"{report_directory}/fig_{myMode}{name}.png" 
    textPath = f"{report_directory}/{myMode}{name}.txt" 
    plt.savefig(figPath)
    print(f"Saving result to: {figPath}")
    print('RMSE = ' + str("{:.2f}".format(datadict[name + 'rmse'])) + ' and MAE = ' + str("{:.2f}".format(datadict[name + 'MAE'])))
    output_str = 'RMSE = ' + str("{:.2f}".format(datadict[name + 'rmse'])) + ' and MAE = ' + str("{:.2f}".format(datadict[name + 'MAE']))
    if not os.path.exists(report_directory):
        os.makedirs(report_directory)
    with open(textPath, 'w') as file:
        file.write(f"Saving result to: {figPath}\n")
        file.write(output_str + '\n')
        file.write('Predictions vs Actual Values:\n')
        for i in range(len(predict)):
            file.write(f"{predict[i][pc]}, {test_y[i]} \n")

#==============================================================================================
# Main call
#==============================================================================================

b=mode_switch(mode)
load_data(temp_dir)

# Normalize the data before encoding
test_x=np.transpose(test_x, (0, 2, 3, 1))
if mode == "ALL":
    test_x, test_y = normalize_channels(test_x, test_y[:,0:3])
else:
    test_x, test_y = normalize_channels(test_x, test_y[:,b])
if st_embed:
    test_z = normalize_Z(test_z)

# Load  model and perform predictions
lin_model_path = model_dir + "_linear_model.npz"
print(f"Loading linear model from: {lin_model_path}")
lin = np.load(lin_model_path)

B = lin["B"]                        # (F+1, T)
n_features = int(lin["n_features"][0])

# Build test features: spatial average over H,W, optionally concat Z
X = test_x
N, H, W, C = X.shape
X_red = X.mean(axis=(1, 2))      # (N, C)
X_feat = X_red                   # base features

if st_embed and test_z is not None:
    if test_z.shape[0] != N:
        raise ValueError(
            f"test_z and test_x have inconsistent sample sizes: "
            f"{test_z.shape[0]} vs {N}"
        )
    X_feat = np.concatenate([X_feat, test_z], axis=1)

if X_feat.shape[1] != n_features:
    raise ValueError(
        f"Feature size mismatch: test has {X_feat.shape[1]} features, "
        f"but linear model was trained with {n_features}."
    )

# Add bias and predict: Y_hat = [1, X_feat] @ B
X_aug = np.concatenate(
    [np.ones((N, 1), dtype=X_feat.dtype), X_feat],
    axis=1
)   # (N, F+1)

predict = X_aug @ B              # (N, T) where T=1 or 3 depending on mode
print(f"Prediction output shape is {predict.shape}")

name = model_name
if mode == "ALL":
    for pc in range(3):
        plotPrediction(datadict, predict, test_y, pc, mode, name, unit, report_directory)
else:
    plotPrediction(datadict, predict, test_y, 0, mode, name, unit, report_directory)
