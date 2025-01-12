import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['legend.loc'] = 'lower right'

# ---------- Random ----------
random_accs_mean = {
    'CXRB10': [22.45,26,29.73,30.94,32.21,33.56,34.39,35.38,36.32],
    'DeepWeeds': [84.37,89.64,91.8,93.49,94.43,94.98,95.19,95.57,95.82],
    'DTD': [50.39,59.96,65.47,67.95,71.16,71.78,72.87,74.05,74.61],
    'FGVCAircraft': [18.6,31.25,41.57,48.83,55.42,59.76,62.92,66.56,68.61],
    'Sketch': [36.8,57.27,63.79,69.78,72.49,74.56,76.27,77.24,78.34],
    'Average': [42.39,52.67,58.32,62.04,64.99,66.76,68.19,69.59,70.59]
}

# ---------- Ours ----------
ours_accs_mean = {
    'CXRB10': [24.89,29.08,31.41,33.13,34.53,35.42,36.04,37.02,37.37],
    'DeepWeeds': [85.95,90.84,92.88,94.05,94.86,95.37,95.72,95.96,96.15],
    'DTD': [52.95,61.82,66.49,69.51,71.78,72.81,73.84,74.35,74.82],
    'FGVCAircraft': [20.31,33.8,43.16,50.05,56.79,60.68,63.67,66.89,69.05],
    'Sketch': [40.64,57.58,67.21,70.72,73.35,75.15,76.58,77.95,78.7],
    'Average': [44.95,54.62,60.23,63.42,66.06,67.75,69.05,70.35,71.16],
}

ours_accs_std = {
    'CXRB10': [0.28,0.55,0.47,0.27,0.06,0.34,0.19,0.34,0.39],
    'DeepWeeds': [0.15,0.32,0.17,0,0.05,0.09,0.13,0.12,0.02],
    'DTD': [0.24,0.48,0.12,0.11,0.09,0.13,0.17,0.08,0.15],
    'FGVCAircraft': [0.41,0.25,0.3,0.15,0.26,0.12,0.07,0.07,0.08],
    'Sketch': [0.38,0.16,0.04,0.08,0.08,0.11,0.1,0.08,0.09],
    'Average': [0.1,0.2,0.14,0.06,0.06,0.05,0.09,0.04,0.1]
}

# ---------- Forgetting ----------
forgetting_accs_mean = {
    'CXRB10': [23.34,28.3,29.62,31.21,33.69,34.08,35.19,35.61,36.48],
    'DeepWeeds': [80.81,88.8,91.34,91.81,92.53,94.03,94.92,95.55,95.89],
    'DTD': [47.97,61.76,66.99,69.41,71.83,72.57,73.79,74.24,74.54],
    'FGVCAircraft': [19.69,33.11,42.8,50.26,56.5,60.72,63.51,66.86,68.8],
    'Sketch': [39.47,58.15,64.65,69.31,71.73,73.98,75.85,77.1,78.36],
    'Average': [42.09,53.82,58.9,62.22,65.08,66.98,68.66,69.79,70.73]
}

forgetting_accs_std = {
    'CXRB10': [0.57,0.3,0.91,0.76,0,0.47,0.88,0.23,0.75],
    'DeepWeeds': [0.62,0.31,0.09,0.29,0.76,0.33,0.02,0.03,0.08],
    'DTD': [0.03,1.42,0.99,1.9,1.42,1.69,1.78,1.21,1.07],
    'FGVCAircraft': [1.26,0.62,0.13,0.01,0.69,0.1,0.08,0.06,0.15],
    'Sketch': [0.87,0.2,0.08,0.27,0.26,0.01,0.15,0.15,0.05],
    'Average': [0.09,0.16,0.18,0.08,0.38,0.29,0.32,0.04,0.15]
}

# ---------- Entropy ----------
entropy_accs_mean = {
    'CXRB10': [24.09,28.15,29.08,32.17,33.08,34.04,34.31,35.23,36.43],
    'DeepWeeds': [77.52,84.07,87,89.4,91.02,92.42,93.62,94.71,95.58],
    'DTD': [53.87,60.66,64.1,66.86,68.7,70.42,71.51,72.85,74.32],
    'FGVCAircraft': [21.21,32.5,41.66,47.99,54.02,57.51,61.48,65.2,68.2],
    'Sketch': [43.85,59.89,65.5,69.12,71.4,73.35,75.32,76.77,78.26],
    'Average': [43.96,52.9,57.32,60.94,63.48,65.4,67.11,68.8,70.4]
}

entropy_accs_std = {
    'CXRB10': [1,0.27,0.03,0.72,0.04,0.11,0.49,0.15,0.5],
    'DeepWeeds': [0.64,0.08,0.15,0.08,0.06,0.07,0.06,0.06,0.12],
    'DTD': [0.55,0.81,1.21,1.34,0.97,0.85,0.98,0.8,1.27],
    'FGVCAircraft': [0.07,0.83,0.08,0.55,0.05,0.37,0.37,0.09,0.15],
    'Sketch': [0.88,0.26,0.16,0.19,0.04,0.09,0.02,0.03,0.02],
    'Average': [0.14,0.1,0.04,0.33,0.03,0.04,0.19,0.05,0.08]
}

# ---------- kCG ----------
kcg_accs_mean = {
    'CXRB10': [20.61,25.31,27.89,30.09,30.89,33.49,34.9,35.59,36.73],
    'DeepWeeds': [84.02,89.57,92.14,93.76,94.61,95.2,95.54,95.66,95.91],
    'DTD': [45.87,58.76,65.17,68.31,70.45,72.57,73.44,74.52,75.14],
    'FGVCAircraft': [17.03,29.09,39.33,47.13,54.25,57.89,62.81,65.53,68.7],
    'Sketch': [34.19,55.41,64.34,69.63,72.67,74.64,76.36,77.53,78.59],
    'Average': [40.04,51.39,57.58,61.62,64.43,66.59,68.47,69.61,70.87]
}

kcg_accs_std = {
    'CXRB10': [0.15,0.24,0.7,0.13,0.66,0.72,0.05,0.16,0.72],
    'DeepWeeds': [0.63,0.04,0.08,0.14,0.05,0.03,0.14,0.13,0.09],
    'DTD': [2.19,1.07,2,0.79,0.54,1.3,0.79,1.31,1.16],
    'FGVCAircraft': [0.49,0.31,0.95,0.37,0.02,1.49,0.15,0.71,0.42],
    'Sketch': [0.82,0.84,0.07,0.15,0.14,0.29,0.09,0.08,0.15],
    'Average': [0.16,0.31,0.46,0.09,0.07,0.23,0.02,0.22,0.13]
}

# ---------- cd ----------
cd_accs_mean = {
    'CXRB10': [17.32,22.33,25.51,28.89,30.89,33.39,33.78,35.5,36.17],
    'DeepWeeds': [78.09,88.96,93.14,94.43,95.3,95.63,95.85,96.08,96.16],
    'DTD': [42.52,55.57,62.93,67.2,70.51,72.57,74.05,74.64,75.02],
    'FGVCAircraft': [17.39,30.14,40.61,48.06,55.01,59.91,63.73,66.8,69.05],
    'Sketch': [29.06,52.75,62.23,68.77,72.54,72.25,76.85,77.96,78.84],
    'Average': [36.7,49.8,56.72,61.32,64.7,66.6,68.71,70.06,70.9]
}

cd_accs_std = {
    'CXRB10': [0.08,1.16,0.53,0.22,0.16,0.09,0.65,0.12,0.82],
    'DeepWeeds': [4.58,0.76,0.04,0.18,0.09,0.03,0.15,0.02,0.04],
    'DTD': [2.1,0.87,2.68,1.6,1.15,1.04,1.04,1.44,1.46],
    'FGVCAircraft': [1.06,0.37,0.13,0.07,0.36,0.19,0.21,0.01,0.42],
    'Sketch': [0.91,0.83,0.19,0.13,0.28,3.83,0.1,0,0.1],
    'Average': [0.37,0.05,0.14,0.09,0.09,0.82,0.18,0.11,0.2]
}


# fig config
x = [i / 10 for i in range(1, 10)]
x_ticks = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
x_label = 'Pruning Ratio'
y_label = 'Accuracy Gap (%)'
spine_width = 2.4

for dataset in ['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch', 'Average']:
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=500)

    # CD
    # cd_delta = np.array(cd_accs_mean[args.dataset])[::-1] - np.array(random_accs_mean[args.dataset])[::-1]
    # ax.plot(x, cd_delta, color='red', marker='o', linestyle=':', linewidth=2.0, label='CD')
    # ax.fill_between(x, cd_delta-np.array(cd_accs_std[args.dataset])[::-1], cd_delta+np.array(cd_accs_std[args.dataset])[::-1], facecolor='red', alpha=0.15)

    # kCG
    kcg_delta = np.array(kcg_accs_mean[dataset])[::-1] - np.array(random_accs_mean[dataset])[::-1]
    ax.plot(x, kcg_delta, color='#DA422A', marker='.', linestyle=':', linewidth=2.0, label='kCG', markersize=20)
    ax.fill_between(x, kcg_delta-np.array(kcg_accs_std[dataset])[::-1], kcg_delta+np.array(kcg_accs_std[dataset])[::-1], facecolor='red', alpha=0.15)

    # Entropy
    entropy_delta = np.array(entropy_accs_mean[dataset])[::-1] - np.array(random_accs_mean[dataset])[::-1]
    ax.plot(x, entropy_delta, color='#555D9E', marker='.', linestyle='--', linewidth=2.0, label='Entropy', markersize=20)
    ax.fill_between(x, entropy_delta-np.array(entropy_accs_std[dataset])[::-1], entropy_delta+np.array(entropy_accs_std[dataset])[::-1], facecolor='blue', alpha=0.15)

    # Forgetting
    forgetting_delta = np.array(forgetting_accs_mean[dataset])[::-1] - np.array(random_accs_mean[dataset])[::-1]
    ax.plot(x, forgetting_delta, color='#4BA05C', marker='.', linestyle='-.', linewidth=2.0, label='Forgetting', markersize=20)
    ax.fill_between(x, forgetting_delta-np.array(forgetting_accs_std[dataset])[::-1], forgetting_delta+np.array(forgetting_accs_std[dataset])[::-1], facecolor='green', alpha=0.15)

    # Ours
    our_delta = np.array(ours_accs_mean[dataset])[::-1] - np.array(random_accs_mean[dataset])[::-1]
    ax.plot(x, our_delta, color='#ED7117', marker='.', linestyle='-', linewidth=2.0, label='Ours', markersize=20)
    ax.fill_between(x, our_delta-np.array(ours_accs_std[dataset])[::-1], our_delta+np.array(ours_accs_std[dataset])[::-1], facecolor='orange', alpha=0.15)

    # decoration
    ax.set_xlabel(x_label, fontsize=24, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=24, fontweight='bold')
    ax.set_xticks(x, x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(prop={'size': 16})

    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)

    ax.grid(visible=True, linestyle=':')

    ax.set_title(dataset, fontsize=24, fontweight='bold')
    # save
    fig_path = os.path.join('./figs/main/gap', dataset+'.pdf')
    fig.tight_layout()
    fig.savefig(fig_path)