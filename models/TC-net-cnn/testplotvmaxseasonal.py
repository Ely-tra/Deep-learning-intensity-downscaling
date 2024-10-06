import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import re  # Import regex module

# Assuming the files are in a directory called 'd'
directory = '/Users/elytra/Documents/JupyterNB/Processing_Center/monthly/'

# Function to calculate MAE and RMSE
def calculate_errors(true_values, predictions):
    mae = tf.keras.losses.MeanAbsoluteError()
    rmse = tf.keras.metrics.RootMeanSquaredError()

    mae_result = mae(true_values, predictions).numpy()
    rmse_result = rmse(true_values, predictions).numpy()

    return mae_result, rmse_result

# Lists to store results
months = []
mae_values = []
rmse_values = []
sample_counts = []

# Regex pattern to find month number in filenames
month_pattern = re.compile(r'\d+')

# Get all filenames, sort them to ensure chronological order
filenames = sorted(os.listdir(directory), key=lambda x: int(month_pattern.search(x).group(0)) if month_pattern.search(x) else 0)

# Load files and compute errors
for filename in filenames:
    if filename.startswith('resultVMAX'):
        month_number = int(month_pattern.search(filename).group(0))
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month_number - 1]
        vmax_file = os.path.join(directory, filename)
        test_file = os.path.join(directory, f'test18x18{month_number:02d}fixed.npy_y.npy')

        if os.path.exists(vmax_file) and os.path.exists(test_file):
            vmax_data = np.load(vmax_file)
            test_data = np.load(test_file)

            # Check shapes and compatibility
            if test_data.shape[0] == vmax_data.shape[0] and test_data.shape[1] == 3:
                # Extract first channel data
                first_channel_data = test_data[:, 0]

                # Calculate MAE and RMSE
                mae_result, rmse_result = calculate_errors(first_channel_data, vmax_data)

                # Store results
                months.append(month_name)
                mae_values.append(mae_result)
                rmse_values.append(rmse_result)
                sample_counts.append(test_data.shape[0])  # Store the number of samples

            else:
                print(test_data.shape, vmax_data.shape)
                print(f'Data mismatch in month: {month_name}')
        else:
            print(vmax_file, test_file)
            print(f'Missing files for month: {month_name}')


# Plotting the results
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# Plotting MAE with sample counts as xticks
bars1 = axs[0].bar(range(len(sample_counts)), rmse_values, color='blue', width=0.5)
axs[0].set_title('Maximum wind speed retrieval accuracy by month', fontsize=20)
axs[0].set_ylabel('RMSE', fontsize=20)
axs[0].set_ylim(3.9, max(rmse_values) + 1)
axs[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
axs[0].set_axisbelow(True)  # Ensure grid is behind the bars

# Move xticks to the top of the subplot, just below the top border
axs[0].xaxis.set_label_position('top')  # Position xlabel at the top
axs[0].xaxis.tick_top()  # Move the ticks to the top
axs[0].set_xticks(range(len(sample_counts)))

# Create white boxes for xticks and labels without borders
axs[0].set_xticklabels(sample_counts, fontsize=14, 
                       bbox=dict(facecolor='white', boxstyle='round,pad=0.3', edgecolor='none'))

# Adjust tick positions and label
axs[0].tick_params(axis='x', which='both', pad=-25)  # Adjust position closer to the border
axs[0].tick_params(axis='y', which='major', labelsize=14)
axs[0].set_xlabel('Number of Samples', fontsize=16, labelpad=-45,
                  bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0'))


# Plotting RMSE with months as xticks
bars2 = axs[1].bar(range(len(months)), mae_values, color='red', width=0.5)
axs[1].set_ylabel('MAE', fontsize=20)
axs[1].set_ylim(2.7, max(mae_values) + 1)
axs[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
axs[1].set_axisbelow(True)  # Ensure grid is behind the bars

# Set months as xticks for the second subplot with enlarged font size
axs[1].set_xticks(range(len(months)))
axs[1].set_xticklabels(months, fontsize=14)  # Increase font size
axs[1].set_xlabel('Month', fontsize=16)
axs[1].tick_params(axis='y', which='major', labelsize=14)

# Adjust layout to ensure everything fits well
plt.tight_layout()
plt.savefig('/Users/elytra/Documents/JupyterNB/Processing_Center/monthly/20%.png')

