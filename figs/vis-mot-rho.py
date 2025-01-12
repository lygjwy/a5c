'''
Visualize the Motivation Spearman Correlation Coefficient between the optimization-based LC and disclosing-based LC
'''

import os
# os.system('PAR_DIR=$(pwd); export PYTHONPATH="${PAR_DIR}:$PYTHONPATH"')

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')

from scipy import stats

import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from datasets import get_dataset


def extract_cidxs(dataset, cidx):

    mu, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # IN-1K

    data_trfms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mu, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mu, std)
        ])
    }

    datasets = {mode: get_dataset('/mnt/sharedata/hdd/jwy/datasets', dataset, mode, data_trfms[mode]) for mode in ['train', 'val']}
    train_sets = ConcatDataset([datasets['train'], datasets['val']])

    length = len(train_sets)

    selected_idxcs = []
    for i in range(length):
        _, y = train_sets[i]
        if y == cidx:
            selected_idxcs.append(i)
    
    return selected_idxcs


def min_max_norm(lists):
        max_, min_ = np.max(lists), np.min(lists)
        
        return (lists - min_) / (max_ - min_)


def main():

    datasets = ['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch']
    class_names = ['Nodule', 'Parthenium', 'Sprinkled', 'DHC-6', 'Revolver']
    cidxs = [9, 0, 43, 7, 8]
    colors = ['blue', 'purple', 'orange', 'red', 'green']

    sccs, ft_ls_crs, masking_ls_crs = [], [], []

    # load data
    for i, cidx in enumerate(cidxs):
        dataset = datasets[i]
        c_idxs = extract_cidxs(dataset, cidx)

        ft_ls_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, 'cle/pt_fully-arch_resnet18/ls.npy')
        ft_ls_c = np.load(ft_ls_path)[c_idxs]
    
        masking_ls_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, 'lwot-l1-s95-i20_f50/pt_fully-arch_resnet18/ls.npy')
        masking_ls_c = np.load(masking_ls_path)[c_idxs]
    
        res = stats.spearmanr(ft_ls_c, masking_ls_c)
        rho = res.statistic
        sccs.append('%.2f' % rho)

        idxs = np.random.choice(len(c_idxs), 64, replace=False)
        ft_ls_crs.append(min_max_norm(ft_ls_c[idxs]))
        masking_ls_crs.append(min_max_norm(masking_ls_c[idxs]))

    # plot
    ##  fig config
    x_label = 'Loss @ Masking'
    y_label = 'Loss @ Optimization'
    spine_width = 2.4
    label_size = 20
    # title_size = 32
    tick_size = 20
    # legend_size = 24

    for i, dataset in enumerate(['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch']):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=500)
        class_name = class_names[i]
            
        ax.scatter(masking_ls_crs[i], ft_ls_crs[i], c=colors[i], alpha=0.8, zorder=10)

        # decoration
        ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')
        ax.set_ylim([0.0, 1.0])

        ax.tick_params(axis='both', which='major', labelsize=tick_size)

        for loc in ['bottom', 'left', 'top', 'right']:
            ax.spines[loc].set_linewidth(spine_width)

        ax.grid(visible=True, linestyle=':', zorder=0)
        # title = '(c)'
        # + datasets[i] + ' (' + class_name + ')\n' + r"$\rho = $" + str(sccs[i])
        # ax.set_title(title, fontsize=title_size, fontweight='bold')
        print(class_name, str(sccs[i]))
        
        # save
        fig_path = os.path.join('./figs/mot', 'rho-%s.pdf' % dataset)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.clf()

if __name__ == '__main__':

    main()
