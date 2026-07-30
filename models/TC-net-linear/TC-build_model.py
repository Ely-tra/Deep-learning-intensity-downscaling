# DESCRIPTION: This script utilizes TensorFlow to implement a (CNN) designed for correcting 
#       TC intensity/structure from grided climate data, using the workflow inherited from the
#       previous TC formation project (https://github.com/kieucq/tcg_deep_learning). The model 
#       consists of several layers with varying functionalities including convolutional layers 
#       for TC feature extraction and dense layers for regression. Special attention is given 
#       to preprocessing steps like normalization and resizing, and the model is tuned to adapt 
#       its learning rate over epochs.
#
#       Note that one can re-design the model by looking at the model's layer configurations 
#       and loss functions (see line ???), set the dataset paths (see line ???), and run the 
#       script. The model architecture (see line ???) can be adjusted by modifying the parameters 
#       for convolutional and dense layers.   
#
# MODEL LAYERS:
#       - Layer 1 (Conv2D): 128 filters, 15x15 kernel, 'relu' activation, input shape=input data
#       - Layer 2 (MaxPooling2D): Pool size of 2, reduces spatial dimensions by half.
#       - Layer 3 (Conv2D): 64 filters, 15x15 kernel, uses 'relu' activation.
#       - Layer 4 (MaxPooling2D): Pool size of 2, further reduces spatial dimensions.
#       - Layer 5 (Conv2D): 256 filters, 9x9 kernel, uses 'relu' activation.
#       - Layer 6 (MaxPooling2D): Pool size of 2, reduces spatial dimensions.
#       - Layer 7 (Conv2D): Configurable number of filters, 5x5 kernel, uses 'relu' activation 
#       - Layer 8 (Conv2D): Same as previous, but with 'valid' padding to adjust output size.
#       - Flatten and Dense layers: Transform convolutional output to 1D .
#
# FUNCTIONS:
#       - mae_for_output: Custom mean absolute error function for specific outputs. Interchangable 
#         with TF MAE metric
#       - rmse_for_output: Custom root mean squared error function for specific outputs. 
#         Interchangable with TF RMSE metric.
#       - main: Orchestrates model construction, compilation, and training using specified 
#         parameters and datasets.
#       - normalize_channels: Normalizes data channels within the input array.
#
# USAGE: Users need to modify the main call with proper paths and parameters before running 
#
# HIST: - May 14, 2024: created by Khanh Luong
#       - May 18, 2024: cross-checked and cleaned up by CK
#
# AUTH: Minh Khanh Luong
#==============================================================================================
import numpy as np
import argparse
import os
#
# Edit the parameters properly before running this script
#
def parse_args():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer model for TC intensity correction.')
    parser.add_argument('-m', '--mode', type=str, default = 'VMAX', help='Mode of operation (e.g., VMAX, PMIN, RMW)')
    parser.add_argument('-mname', '--model_name', type=str, default = 'LINmodel', help='Core name of the model')
    parser.add_argument('-r', '--root', type=str, default = '/N/project/Typhoon-deep-learning/output/', help='Working directory path')
    parser.add_argument('-vno', '--var_num', type=int, default = 13, help='Number of variables')
    parser.add_argument('-st', '--st_embed', type=int, default = 0, help='Including space-time embedded')
    parser.add_argument('-ss', '--data_source', type=str, default = 'MERRA2')
    parser.add_argument('-temp', '--work_folder', type=str, default='/N/project/Typhoon-deep-learning/output/', help='Temporary working folder')
    parser.add_argument('-tid', '--temp_id', type=str)
    return parser.parse_args()
args = parse_args() 				
mode = args.mode
root = args.root
var_num = args.var_num
st_embed = args.st_embed
data_source=args.data_source
work_folder=args.work_folder
temp_id=args.temp_id
model_dir = os.path.join(root, 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_name = args.model_name
model_name = f'{model_name}_{data_source}_{mode}{"_st" if st_embed else ""}'
temp_dir = os.path.join(work_folder, 'temp')
#####################################################################################
# DO NOT EDIT BELOW UNLESS YOU WANT TO MODIFY THE SCRIPT
#####################################################################################
def mode_switch(mode):
    switcher = {
        'VMAX': 0,
        'PMIN': 1,
        'RMW': 2
    }
    # Return the corresponding value if mode is found, otherwise return None or a default value
    return switcher.get(mode, None)
def load_data(temp_dir, temp_id=temp_id):
    global train_x, train_y, train_z, val_x, val_y, val_z

    # Check for training data files and load them if they exist
    if f'train_x_{temp_id}.npy' in os.listdir(temp_dir):
        train_x = np.load(os.path.join(temp_dir, f'train_x_{temp_id}.npy'))
    if f'train_y_{temp_id}.npy' in os.listdir(temp_dir):
        train_y = np.load(os.path.join(temp_dir, f'train_y_{temp_id}.npy'))
    if f'train_z_{temp_id}.npy' in os.listdir(temp_dir):
        train_z = np.load(os.path.join(temp_dir, f'train_z_{temp_id}.npy'))

    # Check for validation data files and load them if they exist
    if f'val_x_{temp_id}.npy' in os.listdir(temp_dir):
        val_x = np.load(os.path.join(temp_dir, f'val_x_{temp_id}.npy'))
    if f'val_y_{temp_id}.npy' in os.listdir(temp_dir):
        val_y = np.load(os.path.join(temp_dir, f'val_y_{temp_id}.npy'))
    if f'val_z_{temp_id}.npy' in os.listdir(temp_dir):
        val_z = np.load(os.path.join(temp_dir, f'val_z_{temp_id}.npy'))


#==============================================================================================
# Normalize bands value.
# NOTE: normalize only features, not labels (althought requires labels as input)
# NOTE: normalize by sample, not batch normalization.
#==============================================================================================


def normalize_channels(X,y):
    """
    Normalizes each channel in each sample individually.

    Parameters:
    - X: Input array of shape (nsample, height, width, number_channels).
    - y: Corresponding labels.

    Returns:
    - Normalized X and y arrays.
    """
    nsample = X.shape[0]
    number_channels = X.shape[3]
    for i in range(nsample):
        for var in range(number_channels):
            maxvalue = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
            X[i,:,:,var] = X[i,:,:,var] / abs(maxvalue)
    print("Finish normalization...")
    return X,y

#==============================================================================================
# Defining custom learning rate
#==============================================================================================
def normalize_Z(Z):
    Z[:,2] = (Z[:,2]+90) / 180
    Z[:,3] = (Z[:,3]+180) / 360
    return Z
    
def _ensure_2d_y(y):
    """
    Ensure target array is 2D: (N,) -> (N,1); (N,k) stays (N,k).
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


def _fit_linear_model(X_feat, Y):
    """
    Fit linear regression Y = X_feat @ B (with bias term).
    
    X_feat: (N, F)  features (no bias column yet)
    Y     : (N, T)  targets (2D)
    Returns:
        B          : (F+1, T) coefficient matrix
        Y_pred_tr : (N, T) training predictions
    """
    X_feat = np.asarray(X_feat)
    Y = _ensure_2d_y(Y)

    N = X_feat.shape[0]
    X_aug = np.concatenate(
        [np.ones((N, 1), dtype=X_feat.dtype), X_feat],
        axis=1
    )  # (N, F+1)

    B, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
    Y_pred_tr = X_aug @ B
    return B, Y_pred_tr


def _compute_mae_rmse(y_true, y_pred):
    """
    Compute MAE and RMSE per target column.
    Returns:
        mae  : (T,)
        rmse : (T,)
    """
    y_true = _ensure_2d_y(y_true)
    y_pred = _ensure_2d_y(y_pred)

    diff = y_pred - y_true
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    return mae, rmse

#==============================================================================================
# Model
#==============================================================================================
def main(X, Y, loss='huber', NAME='best_model', st_embed=False, var_num=13):
    """
    Instead of building a CNN, fit a linear regression on spatially averaged channels
    (and optionally Z features when st_embed=True), using train as training and val as test.
    
    X: np.ndarray, shape (N, H, W, C)
    Y: np.ndarray, shape (N,) or (N, T)
    """
    # ---- 1. Build training features (average over H,W; optionally concat Z) ----
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 4:
        raise ValueError(f"Expected X with 4 dims (N,H,W,C), got {X.shape}")

    N, H, W, C = X.shape
    # spatial average over height and width
    X_red = X.mean(axis=(1, 2))     # (N, C)

    # Start with averaged channels as base features
    X_feat_tr = X_red               # (N, C)

    # If space-time embedding is enabled, append Z features
    if st_embed and 'train_z' in globals() and train_z is not None:
        Z_tr = np.asarray(train_z)
        if Z_tr.shape[0] != N:
            raise ValueError(
                f"train_z and X have inconsistent sample sizes: "
                f"{Z_tr.shape[0]} vs {N}"
            )
        # Z_tr is already normalized above via normalize_Z
        X_feat_tr = np.concatenate([X_feat_tr, Z_tr], axis=1)  # (N, C+Z)

    # Fit linear model
    B, Y_pred_tr = _fit_linear_model(X_feat_tr, Y)
    mae_tr, rmse_tr = _compute_mae_rmse(Y, Y_pred_tr)

    # ---- 2. Save the linear model (coefficients + some metadata) ----
    # B has shape (F+1, T): first row is bias
    model_path = NAME + "_linear_model.npz"
    np.savez(
        model_path,
        B=B,
        var_num=np.array([var_num]),
        st_embed=np.array([int(st_embed)]),
        mode=np.array([mode]),
        data_source=np.array([data_source]),
        n_features=np.array([X_feat_tr.shape[1]]),
    )
    print(f"Saved linear model to: {model_path}")

    # For compatibility with old code that expects a 'history', we return a dict
    history = {
        "B": B,
        "train_mae": mae_tr,
        "train_rmse": rmse_tr
    }
    print("Training MAE per target:", mae_tr)
    print("Training RMSE per target:", rmse_tr)

    return history



#==============================================================================================
# MAIN CALL:
#==============================================================================================

b=mode_switch(mode)
load_data(temp_dir)

# Transpose train_x as it is always present
# Normalize train data, which is always present
train_x = np.transpose(train_x, (0, 2, 3, 1))
if mode == "ALL":
    train_x, train_y = normalize_channels(train_x, train_y[:,0:3])
else:
    train_x, train_y = normalize_channels(train_x, train_y[:,b])

# Check if validation data exists before normalization and transposition
if 'val_x' in globals() and 'val_y' in globals():
    val_x = np.transpose(val_x, (0, 2, 3, 1))
    if mode == "ALL":
        val_x, val_y = normalize_channels(val_x, val_y[:, 0:3])
    else:
        val_x, val_y = normalize_channels(val_x, val_y[:, b])

# Normalize val_z if it exists and st_embed is true
if 'val_z' in globals() and val_z is not None and st_embed:
    val_z = normalize_Z(val_z)

# ---- Merge train and val: treat val as part of train ----
if 'val_x' in globals() and val_x is not None and 'val_y' in globals() and val_y is not None:
    print("Merging train and validation sets for final linear model fit...")
    train_x = np.concatenate([train_x, val_x], axis=0)
    train_y = np.concatenate([train_y, val_y], axis=0)

    if st_embed and 'train_z' in globals() and train_z is not None and \
       'val_z' in globals() and val_z is not None:
        train_z = np.concatenate([train_z, val_z], axis=0)

# Assuming train_x is defined and checking the number of channels
number_channels = train_x.shape[3]
print('Input shape of the X features data (train+val): ', train_x.shape)
print('Input shape of the y label data (train+val): ', train_y.shape)
print('Number of input channel extracted from X is: ', number_channels)

history = main(
    X=train_x,
    Y=train_y,
    NAME=os.path.join(model_dir, model_name),
    st_embed=st_embed,
    var_num=var_num
)