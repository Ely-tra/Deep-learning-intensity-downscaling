import re
import matplotlib.pyplot as plt

# 1) channel → name mapping
def convert(ch):
    key = {
        '0 1':   '850 mb wind',
        '4 5':   '950 mb wind',
        '8 9':   '750 mb wind',
        '0 1 4 5 8 9': '3-level wind',
        '2':     '850 mb temperature',
        '6':     '950 mb temperature',
        '10':    '750 mb temperature',
        '2 6 10':'3-level temperature',
        '3':     '850 mb relative humidity',
        '7':     '950 mb relative humidity',
        '11':    '750 mb relative humidity',
        '3 7 11':'3-level relative humidity',
        '12':    'Sea level pressure',
    }
    return key.get(ch, ch)

# 2) containers for each season
dec_apr_ch, dec_apr_rmse, dec_apr_mae = [], [], []
may_nov_ch, may_nov_rmse, may_nov_mae = [], [], []

mode = None
logfile = 'output_fig9.txt'  # replace with your actual log file path

# 3) parse log
with open(logfile) as f:
    for line in f:
        line = line.strip()
        if line.startswith('Cutting'):
            parts = line.split()
            # grab everything between "Cutting" and the first "for"
            first_for = parts.index('for')
            channels  = ' '.join(parts[1:first_for])
            if 'apr_' in line:
                mode = 'apr'
                dec_apr_ch.append(convert(channels))
            elif 'nov_' in line:
                mode = 'nov'
                may_nov_ch.append(convert(channels))
        elif line.startswith('RMSE'):
            m = re.search(r'RMSE\s*=\s*([\d.]+)\s*and\s*MAE\s*=\s*([\d.]+)', line)
            if not m:
                continue
            rmse_val = float(m.group(1))
            mae_val  = float(m.group(2))
            if mode == 'apr':
                dec_apr_rmse.append(rmse_val)
                dec_apr_mae.append(mae_val)
            elif mode == 'nov':
                may_nov_rmse.append(rmse_val)
                may_nov_mae.append(mae_val)

# 4) sort both experiments by May‐Nov RMSE (descending)
may_indices = sorted(range(len(may_nov_rmse)), key=lambda i: may_nov_rmse[i], reverse=True)
# apply that same ordering to Dec-Apr
dec_indices = may_indices

dec_ch_sorted   = [dec_apr_ch[i]   for i in dec_indices]
dec_rmse_sorted = [dec_apr_rmse[i] for i in dec_indices]
dec_mae_sorted  = [dec_apr_mae[i]  for i in dec_indices]

may_ch_sorted   = [may_nov_ch[i]   for i in may_indices]
may_rmse_sorted = [may_nov_rmse[i] for i in may_indices]
may_mae_sorted  = [may_nov_mae[i]  for i in may_indices]
# add Control at the end
control_label = 'Control'
control_rmse  = 4.55
control_mae   = 3.33
may_nov_rmse  = 5.89
may_nov_mae   = 3.76
# Dec-Apr
dec_ch_sorted  .append(control_label)
dec_rmse_sorted.append(control_rmse)
dec_mae_sorted .append(control_mae)
# May-Nov
may_ch_sorted  .append(control_label)
may_rmse_sorted.append(may_nov_rmse)
may_mae_sorted .append(may_nov_mae)

# 5) plotting
plt.rcParams.update({'font.size': 14})

# Dec-Apr plot
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
colors_rmse = ['b'] * (len(dec_ch_sorted)-1) + ['#800080']  # purple last
colors_mae  = ['r'] * (len(dec_ch_sorted)-1) + ['#800080']

axs[0].bar(dec_ch_sorted, dec_rmse_sorted, color=colors_rmse, zorder=3, width=0.5)
axs[0].set_title('Dec-Apr Feature Importance', fontsize=18)
axs[0].set_ylabel('RMSE', fontsize=16)
axs[0].axhline(y=control_rmse, linestyle='--', color='black', linewidth=1.5)
axs[0].set_ylim([4,7.4])
axs[0].grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)

axs[1].bar(dec_ch_sorted, dec_mae_sorted, color=colors_mae, zorder=3, width=0.5)
axs[1].set_ylabel('MAE', fontsize=16)
axs[1].set_xlabel('Removed Feature', fontsize=16)
axs[1].set_xticklabels(dec_ch_sorted, rotation=60, ha='right')
axs[1].set_ylim([2.7,4.6])
axs[1].axhline(y=control_mae, linestyle='--', color='black', linewidth=1.5)
axs[1].grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)

plt.tight_layout()
plt.savefig('fig_dec_apr_feature_importance.png', dpi=400)

# May-Nov plot
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# example for Dec-Apr RMSE subplot
colors_rmse = ['b'] * (len(dec_ch_sorted)-1) + ['#800080']  # purple last
colors_mae  = ['r'] * (len(dec_ch_sorted)-1) + ['#800080']


axs[0].bar(may_ch_sorted, may_rmse_sorted, color=colors_rmse, zorder=3, width=0.5)
axs[0].set_title('May-Nov Feature Importance', fontsize=18)
axs[0].set_ylabel('RMSE', fontsize=16)
axs[0].set_ylim([4,7.4])
axs[0].axhline(y=may_nov_rmse, linestyle='--', color='black', linewidth=1.5)
axs[0].grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)

axs[1].bar(may_ch_sorted, may_mae_sorted, color=colors_mae, zorder=3, width=0.5)
axs[1].set_ylabel('MAE', fontsize=16)
axs[1].set_xlabel('Removed Feature', fontsize=16)
axs[1].set_xticklabels(may_ch_sorted, rotation=60, ha='right')
axs[1].set_ylim([2.7,4.6])
axs[1].axhline(y=may_nov_mae, linestyle='--', color='black', linewidth=1.5)
axs[1].grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)

plt.tight_layout()
plt.savefig('fig_may_nov_feature_importance.png', dpi=400)

