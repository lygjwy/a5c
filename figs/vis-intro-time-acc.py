'''
Bar plots comparing time and acc of different dataset pruning methods
'''

import os
# PAR_DIR=$(pwd); export PYTHONPATH="${PAR_DIR}:$PYTHONPATH"

import matplotlib.ticker
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')

times = [1035, 1035.2, 1033, 1033, 1033, 1033, 1035.3, 1033, 4051, 29.5]
accs = [54.31, 58.34, 59.15, 59.06, 59.14, 59.03, 57.48, 56.88, 56.43, 60.45]
acc_rand = 59.21
time_ft = 1000

def main():

    # plotting
    x_ticks = ['Herding', 'kCG', 'Forget', 'LC', 'Entropy', 'Margin', 'CD', 'GraNd', 'EL2N', 'Ours']
    x_pos = np.arange(len(x_ticks))
    x_label = 'Method'
    y_label_time = 'Time (s)'
    y_label_acc = 'Accuracy (%)'
    
    colors = ['#565e9f', '#3c69a8', '#0a73ad', '#007cac', '#0084a1', '#008b98', '#00928d', '#00987d', '#4ba05c', '#ED7117']

    ## config
    spine_width = 2.4
    label_size = 24
    # title_size = 32
    tick_size = 11
    legend_size = 16

    # ----------------------- unmasking -------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=500)
    
    bar1 = ax.bar(x_pos, np.log10(np.array(times)), align='center', linewidth=1.6, edgecolor='black', color=colors, zorder=10)
    for i, rect in enumerate(bar1):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{times[i]:.0f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 3, 4], ['10', '100', '1000', '10000'])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(log_to_linear))

    # ax.axhline(y=time_ft, color='black', linewidth=1.6, linestyle='dotted', label='Fine-tuning', zorder=5)

    # ax.set_ylim(0, 4100)
    ax.set_xticks(x_pos, x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
    ax.set_ylabel(y_label_time, fontsize=label_size, fontweight='bold')
    
    # ax.legend(prop={'size': legend_size}, loc=2)
    ## spines
    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)
        ax.spines[loc].set_zorder(12)

    ## grid
    ax.grid(axis='y', which='major', visible=True, linestyle=':', zorder=0)

    # Saving
    fig_path = os.path.join('./figs/intro', 'time.pdf')
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=500)
    ## right part
    bar2 = ax.bar(x_pos, accs, align='center', linewidth=1.6, edgecolor='black', color=colors, zorder=10)
    for rect in bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height+0.1, f'{height:.2f}', ha='center', va='bottom', fontsize=12)

    ax.axhline(y=acc_rand, color='black', linewidth=1.6, linestyle='dashed', label='Random', zorder=5)
    ax.set_ylim(54, 61)
    ax.set_xticks(x_pos, x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
    ax.set_ylabel(y_label_acc, fontsize=label_size, fontweight='bold')
    
    ## legends
    ax.legend(prop={'size': legend_size}, loc=2)

    ## spines
    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)
        ax.spines[loc].set_zorder(12)

    ## grid
    ax.grid(axis='y', which='major', visible=True, linestyle=':', zorder=0)

    # Saving
    fig_path = os.path.join('./figs/intro', 'accs.pdf')
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.clf()


if __name__ == '__main__':

    main()