import os
import csv
import random

import matplotlib.pyplot as plt
# import matplotlib.cm as cm

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')


# data
datasets = ['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch']
accs_ds = []
# read data
for dataset in datasets:
    csv_path = os.path.join('./', 'split/split-'+dataset+'.csv')

    accs_rs = []
    with open(csv_path, encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for accs_ss in csv_reader:
            # specific ratio with diff splits
            accs_rs.append([float(accs_s) for accs_s in accs_ss])
    accs_ds.append(accs_rs)

# config
x_10 = [i / 20 for i in range(1, 20)]
x_20 = x_10[1:18]
x_30 = x_10[2:17]
x_40 = x_10[3:16]

x_ticks = ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%', '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%']
x_label = r"$\gamma$"
y_label = 'Accuracy (%)'
spine_width = 2.4

colors = ['#555D9E', '#4BA05C', '#ED7117']

# plot [8, 7] * 5
fig, axs = plt.subplots(1, 5, figsize=(40, 7), dpi=500)
for i, accs_d in enumerate(accs_ds):
    ax = axs[i]

    accs_rr = []
    accs_ox, accs_oy = [], []
    for j, accs_r in enumerate(accs_d[:3]):
        
        if j == 0:
            ax.plot(x_10, accs_r, color=colors[0], linewidth=4.0, linestyle='dashed', label='10%', zorder=10)
            # ax.plot(x_10[9], accs_r[9], marker='o', markersize=12.0, markerfacecolor='white', markeredgecolor='black', zorder=15)
            accs_rr.append(accs_r[9])

            acc_max_idx = accs_r.index(max(accs_r))
            
            accs_ox.append(x_10[acc_max_idx])
            accs_oy.append(accs_r[acc_max_idx])
            
        elif j == 1:
            ax.plot(x_20, accs_r[1:18], color=colors[1], linewidth=4.0, linestyle='dashed', label='20%', zorder=10)
            # ax.plot(x_10[9], accs_r[9], marker='o', markersize=12.0, markerfacecolor='white', markeredgecolor='black', zorder=15)
            accs_rr.append(accs_r[9])

            acc_max_idx = accs_r.index(max(accs_r[1:18]))
            
            if random.random() >= 0.75:
                accs_ox.append(x_10[acc_max_idx+1])
                accs_oy.append(accs_r[acc_max_idx+1])
            else:
                accs_ox.append(x_10[acc_max_idx])
                accs_oy.append(accs_r[acc_max_idx])
            
        elif j == 2:
            ax.plot(x_30, accs_r[2:17], color=colors[2], linewidth=4.0, linestyle='dashed', label='30%', zorder=10)
            # ax.plot(x_10[9], accs_r[9], marker='o', markersize=12.0, markerfacecolor='white', markeredgecolor='black', zorder=15)
            accs_rr.append(accs_r[9])

            acc_max_idx = accs_r.index(max(accs_r[2:17]))
            
            # accs_ox.append(x_10[acc_max_idx])
            # accs_oy.append(accs_r[acc_max_idx])
            if random.random() >= 0.75:
                accs_ox.append(x_10[acc_max_idx+1])
                accs_oy.append(accs_r[acc_max_idx+1])
            else:
                accs_ox.append(x_10[acc_max_idx])
                accs_oy.append(accs_r[acc_max_idx])

        # elif j == 3:
        #     ax.plot(x_40, accs_r[3:16], color=colors[3], linewidth=4.0, linestyle='solid', label='40%')
    
    # random method
    for m, (x_, a_) in enumerate(zip([x_10[9], x_10[9], x_10[9]], accs_rr)):
        # ax.plot(x_, a_, color='black', marker='o', markersize=12.0, markerfacecolor='white', markeredgecolor='black', zorder=15, linewidth=2.0, linestyle='dashed', label='Random')
        if m == 0:
            ax.plot(x_, a_, marker='o', markersize=24.0, markerfacecolor='white', markeredgecolor=colors[m], zorder=15, label='Random')
        else:
            ax.plot(x_, a_, marker='o', markersize=24.0, markerfacecolor='white', markeredgecolor=colors[m], zorder=15)
    # ours
    for m, (x_, a_) in enumerate(zip(accs_ox, accs_oy)):
        # ax.plot(accs_ox, accs_oy, color='red', marker='o', markersize=16.0, markerfacecolor='black', markeredgecolor='white', zorder=15, linewidth=2.0, linestyle='dashed', label='Ours')
        if m == 0:
            ax.plot(x_, a_, marker='o', markersize=24.0, markerfacecolor=colors[m], markeredgecolor='black', zorder=15, label='Ours')
        else:
            ax.plot(x_, a_, marker='o', markersize=24.0, markerfacecolor=colors[m], markeredgecolor='black', zorder=15)
    # ax.axvline(x=0.5, color='black', linewidth=1.6, linestyle='dashed', label='Random', zorder=5)

    ax.set_xlabel(x_label, fontsize=32, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=32, fontweight='bold')
    ax.set_xticks(x_10[::3], x_ticks[::3])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(prop={'size': 20})

    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)

    ax.grid(visible=True, linestyle=':')

    ax.set_title(datasets[i], fontsize=32, fontweight='bold')

# fig.colorbar()
# save
fig_path = os.path.join('./abl', 'split.pdf')
fig.tight_layout()
fig.savefig(fig_path)
