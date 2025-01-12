'''
Triple y-axis line chart
Visualize the ablation results of frequency in the Learning Complexity (unmasking)
'''
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')

# plt.rcParams['text.usetex'] = True


# data
accs = [56.82, 57.16, 57.31, 57.28, 57.30, 57.29, 57.33, 57.30]
sccs = [0.41, 0.45, 0.49, 0.49, 0.50, 0.51, 0.51, 0.51]
time = [20, 60, 100, 200, 400, 600, 800, 1000]

# config
spine_width = 2.4
label_size = 24
# title_size = 32
tick_size = 20
legend_size = 20

x = [1/50, 3/50, 5/50, 10/50, 20/50, 30/50, 40/50, 50/50]
x_ticks = ['1', '3', '5', '10', '20', '30', '40', '50']
x_label = 'Learning Path Size'

fig, ax = plt.subplots(figsize=(8, 5), dpi=500)

twin1 = ax.twinx()

p1, = ax.plot(x, accs, color='#555D9E', linestyle='solid', linewidth=2.0, marker='.', markersize=15) # , label='Accuracy (%)'
p2, = twin1.plot(x, sccs, color='#ED7117', linestyle='dashdot', linewidth=2.0, marker='.', markersize=15) # , label='Spearman Correlation'
# p3, = twin2.plot(x, time, color='#DA422A', linestyle='dotted', linewidth=2.0, marker='v') # , label='Time (s)'

## label
# ax.set(xlabel=, ylabel='Accuracy (%)', )
# twin1.set(ylabel='Spearman Correlation')
# twin2.set(ylabel='Time (s)')
ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=label_size, fontweight='bold')
twin1.set_ylabel('Spearman Correlation', fontsize=label_size, fontweight='bold')
# twin2.set_ylabel('Time (s)', fontsize=label_size, fontweight='bold')
ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
# twin2.yaxis.label.set_color(p3.get_color())

## ticks
ax.set_xticks(x, x_ticks)
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.tick_params(axis='y', colors=p1.get_color())
twin1.tick_params(axis='both', which='major', labelsize=tick_size)
twin1.tick_params(axis='y', colors=p2.get_color())

# twin2.tick_params(axis='y', colors=p3.get_color())

## spines
for loc in ['bottom', 'left', 'top', 'right']:
    ax.spines[loc].set_linewidth(spine_width)
# twin1.spines['right'].set_linewidth(spine_width)
# twin2.spines['right'].set_linewidth(spine_width)

ax.spines['left'].set_color(p1.get_color())
ax.spines['right'].set_color(p2.get_color())
twin1.spines['left'].set_color(p1.get_color())
twin1.spines['right'].set_color(p2.get_color())
# twin2.spines['right'].set_color(p3.get_color())

## legend
# ax.legend(handles=[p1, p2, p3], prop={'size': legend_size}, loc=1)

## grid
ax.grid(visible=True, linestyle=':')

# fig.suptitle('(a)', fontsize=title_size, fontweight='bold')

fig_path = os.path.join('./figs/abl', 'size.pdf')
fig.tight_layout()
fig.savefig(fig_path)