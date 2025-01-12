'''
Bar plot
Visualize the ablation of under-sampling with flexibility and randomness (FlexRand)
'''
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')

import numpy as np

# data
x_label = 'Learning Complexity'
y_label = 'Accuracy (%)'
x_ticks = ['Masking', 'Optimization']
legends = ['T-H', 'T-E', 'T-F', 'FlexRand']
# colors = ['#c3272b', '#5B84C4', '#26B170']
colors = ['#DA422A', '#555D9E', '#4BA05C', '#ED7117']
x = np.arange(len(x_ticks))
spacing = 0.4
width = (1 - spacing) / len(legends)

accs = [
    [48.85, 49.60], # hard
    [52.50, 55.38], # easy
    [54.26, 55.85], # flexible
    [57.31, 57.50]  # FlexRand
]

fig, ax = plt.subplots(figsize=(8, 5), dpi=500)
for i, (accs_, legend) in enumerate(zip(accs, legends)):
    ax.bar(x - spacing/2 + i*width, accs_, width, label=legend, linewidth=1.6, edgecolor='black', color=colors[i], zorder=10)
ax.axhline(y=55.92, color='black', linewidth=1.6, linestyle='dashed', label='Random', zorder=5)

# Decoration
## config
spine_width = 2.4
label_size = 24
# title_size = 32
tick_size = 20
legend_size = 16

## limit
ax.set_ylim(48.5, 58.5)
## labels
ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')

## ticks
ax.set_xticks(x, x_ticks)
ax.tick_params(axis='both', which='major', labelsize=tick_size)

## legends
ax.legend(prop={'size': legend_size}, loc=2)

## spines
for loc in ['bottom', 'left', 'top', 'right']:
    ax.spines[loc].set_linewidth(spine_width)
    ax.spines[loc].set_zorder(12)

## grid
ax.grid(axis='y', which='major', visible=True, linestyle=':', zorder=0)

# fig.suptitle('(c)', fontsize=title_size, fontweight='bold')

# Saving
fig_path = os.path.join('./figs/abl', 'flexrand.pdf')
fig.tight_layout()
fig.savefig(fig_path)