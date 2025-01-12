'''
Box plot
Ablation on the network pruning strategy: Random v.s. L1
'''
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')

# data
sccs = [
    [0.473, 0.47, 0.498, 0.48, 0.491, 0.49, 0.486, 0.505, 0.51],
    [0.46, 0.41, 0.44, 0.45, 0.42, 0.43, 0.46, 0.46, 0.47],
]

accs = [
    [57.10, 57.43, 57.44, 57.36, 57.39, 57.57, 57.25, 57.20, 57.39],
    [56.97, 56.54, 56.21, 56.40, 56.50, 56.80, 56.43, 56.93, 57.04],
]

# config
spine_width = 2.4
label_size = 24
# title_size = 32
tick_size = 20
legend_size = 16
# colors = ['#26B170', '#5B84C4'] # #FB9B50
colors = ['#4BA05C', '#555D9E']
labels = ['L1', 'Rand']

fig, axs = plt.subplots(2, 1, figsize=(8, 5), dpi=500)

bplot = axs[0].boxplot(sccs, whis=(0, 100), patch_artist=True, labels=labels, vert=False, widths=0.5)
axs[0].set_xlabel('Spearman Correlation', fontsize=label_size, fontweight='bold')
# axs[0].set_title('Spearman Correlation', fontsize=label_size, fontweight='bold')
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
axs[0].set_yticks([1, 2], labels, rotation=90, ha='right')
axs[0].tick_params(axis='y', labelsize=16)
for label in (axs[0].get_yticklabels()):
    # label.set_fontsize(24)
    label.set_fontweight('bold')
axs[0].tick_params(axis='x', labelsize=tick_size)

bplot = axs[1].boxplot(accs, whis=(0, 100), patch_artist=True, labels=labels, vert=False, widths=0.5)
axs[1].set_xlabel('Accuracy (%)', fontsize=label_size, fontweight='bold')
# axs[1].set_title('Accuracy (%)', fontsize=label_size, fontweight='bold')
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
axs[1].set_yticks([1, 2], labels, rotation=90, ha='right')
axs[1].tick_params(axis='y', labelsize=16)
for label in (axs[1].get_yticklabels()):
    # label.set_fontsize(24)
    label.set_fontweight('bold')
axs[1].tick_params(axis='x', labelsize=tick_size)

## grid & spines
for ax in axs:
    ax.grid(visible=False)
    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)

# fig.suptitle('(b)', fontsize=title_size, fontweight='bold')

fig_path = os.path.join('./figs/abl', 'pruning.pdf')
fig.tight_layout()
fig.savefig(fig_path)