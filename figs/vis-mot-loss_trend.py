'''
Visualize the loss trend over the epoch and capacity
'''
import os
# PAR_DIR=$(pwd); export PYTHONPATH="${PAR_DIR}:$PYTHONPATH"
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')

import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from datasets import get_dataset

def extract_cidxs(dataset):

    c_idxs = []
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
    num_classes = datasets['train'].num_classes
    train_sets = ConcatDataset([datasets['train'], datasets['val']])
    for i in range(num_classes):
        c_idxs.append([])

    length = len(train_sets)
    for i in range(length):
        _, y = train_sets[i]
        c_idxs[y].append(i)
    
    return c_idxs


def min_max_norm(min_, max_, array_):
    return (array_ - min_) / (max_ - min_)


def extract_e_h_lss(ft_lss, masking_lss, idxs_c):
    ft_lss_c = ft_lss[:, idxs_c]
    masking_lss_c = masking_lss[:, idxs_c]

    ft_ls_c = np.average(ft_lss_c, axis=0)
    e_idx = np.argmin(ft_ls_c)
    h_idx = np.argmax(ft_ls_c)

    return ft_lss_c[:, e_idx], ft_lss_c[:, h_idx], masking_lss_c[:, e_idx], masking_lss_c[:, h_idx]


def main(args):

    datasets = ['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch', 'Average']
    arch = args.arch

    # collecting the data
    # sccs_all = []
    ft_lss_ds_e, ft_lss_ds_h, ft_lss_ds_m, masking_lss_ds_e, masking_lss_ds_h, masking_lss_ds_m = [], [], [], [], [], []

    for i, dataset in enumerate(datasets[:5]):
        cidxs = extract_cidxs(dataset)

        # calculate the spearman correlation coefficient
        # ft_ls_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, 'cle/pt_fully-arch_%s/ls.npy' % arch)
        # masking_ls_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, 'lwot-l1-s95-i20_f50/pt_fully-arch_%s/ls.npy' % arch)
        # sccs = []
        # for c_idx in cidxs:

        #     ft_ls_c = np.load(ft_ls_path)[c_idx]
        #     masking_ls_c = np.load(masking_ls_path)[c_idx]
        
        #     res = stats.spearmanr(ft_ls_c, masking_ls_c)
        #     rho = res.statistic
        #     sccs.append(rho)
        # sccs_all.append(float('%.2f' % np.mean(sccs)))

        # /mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/dataset/cle/pt_fully-arch_resnet18/lss.npy
        ft_lss_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, 'cle/pt_fully-arch_%s/lss.npy' % arch)
        ft_lss = np.load(ft_lss_path)
        # /mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/dataset/lwot-l1-s95-i20_f50/pt_fully-arch_resnet18/lss.npy
        masking_lss_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, 'lwot-l1-s95-i20_f50/pt_fully-arch_%s/lss.npy' % arch)
        masking_lss = np.load(masking_lss_path)

        ft_lss_es, ft_lss_hs, masking_lss_es, masking_lss_hs = [], [], [], []
        for idxs_c in cidxs:
            ft_lss_c_e, ft_lss_c_h, masking_lss_c_e, masking_lss_c_h = extract_e_h_lss(ft_lss, masking_lss, idxs_c)
            ft_lss_es.append(ft_lss_c_e)
            ft_lss_hs.append(ft_lss_c_h)
            masking_lss_es.append(masking_lss_c_e)
            masking_lss_hs.append(masking_lss_c_h)

        # average over classes
        ft_lss_e = np.average(np.array(ft_lss_es), axis=0)
        ft_lss_h = np.average(np.array(ft_lss_hs), axis=0)
        ft_lss_m = np.average(ft_lss, axis=1)

        masking_lss_e = np.average(np.array(masking_lss_es), axis=0)
        masking_lss_h = np.average(np.array(masking_lss_hs), axis=0)
        masking_lss_m = np.average(masking_lss, axis=1)

        # normalization
        min_, max_ = np.min(np.concatenate([ft_lss_e, ft_lss_h, ft_lss_m])), np.max(np.concatenate([ft_lss_e, ft_lss_h, ft_lss_m]))
        ft_lss_ds_e.append(min_max_norm(min_, max_, ft_lss_e))
        ft_lss_ds_h.append(min_max_norm(min_, max_, ft_lss_h))
        ft_lss_ds_m.append(min_max_norm(min_, max_, ft_lss_m))

        min_, max_ = np.min(np.concatenate([masking_lss_e, masking_lss_h, masking_lss_m])), np.max(np.concatenate([masking_lss_e, masking_lss_h, masking_lss_m]))
        masking_lss_ds_e.append(min_max_norm(min_, max_, masking_lss_e))
        masking_lss_ds_h.append(min_max_norm(min_, max_, masking_lss_h))
        masking_lss_ds_m.append(min_max_norm(min_, max_, masking_lss_m))

    ## average over different datasets
    ft_lss_ds_e.append(np.average(np.stack(ft_lss_ds_e, axis=0), axis=0))
    ft_lss_ds_h.append(np.average(np.stack(ft_lss_ds_h, axis=0), axis=0))
    ft_lss_ds_m.append(np.average(np.stack(ft_lss_ds_m, axis=0), axis=0))
    masking_lss_ds_e.append(np.average(np.stack(masking_lss_ds_e, axis=0), axis=0))
    masking_lss_ds_h.append(np.average(np.stack(masking_lss_ds_h, axis=0), axis=0))
    masking_lss_ds_m.append(np.average(np.stack(masking_lss_ds_m, axis=0), axis=0))

    # sccs_all.append(np.mean(sccs_all))

    # plot
    # fig config
    x = [i/50 for i in range(1, 51)]
    x_ = [1/50,  10/50, 25/50, 40/50, 50/50]
    x_ticks_ft = ['1', '10', '25', '40', '50']
    x_ticks_um = ['2%', '20%', '50%', '80%', '100%']

    x_label_ft = 'Epoch'
    x_label_um = 'Capacity'

    y_label = 'Loss'
    spine_width = 2.4
    label_size = 24
    # title_size = 32
    tick_size = 20
    legend_size = 20
    
    for i, dataset in enumerate(['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch', 'Average']):

        ft_lss_e, ft_lss_h, ft_lss_m = ft_lss_ds_e[i], ft_lss_ds_h[i], ft_lss_ds_m[i]
        masking_lss_e, masking_lss_h, masking_lss_m = masking_lss_ds_e[i], masking_lss_ds_h[i], masking_lss_ds_m[i]
        
        # loss trend for fine-tuning learning complexity
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), dpi=500)
        ax.plot(x, ft_lss_h, color='#DA422A', linestyle='-', linewidth=2.0, label='Hard')
        ax.fill_between(x, 0, ft_lss_h, facecolor='red', alpha=0.15)
        ax.plot(x, ft_lss_e, color='#555D9E', linestyle='-', linewidth=2.0, label='Easy')
        ax.fill_between(x, 0, ft_lss_e, facecolor='blue', alpha=0.2)
        ax.plot(x, ft_lss_m, color='black', linestyle='dashed', linewidth=2.0, label='Test')

        ax.set_xlabel(x_label_ft, fontsize=label_size, fontweight='bold')
        ax.set_xlim(1/50, 1.0)
        ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')
        ax.set_ylim(0.0, 1.0)

        ax.set_xticks(x_, x_ticks_ft)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(prop={'size': legend_size}, loc=2)

        for loc in ['bottom', 'left', 'top', 'right']:
            ax.spines[loc].set_linewidth(spine_width)

        ax.grid(visible=False)
        # ax.set_title('(b)', fontsize=24, fontweight='bold')
        # fig.suptitle('(b)', fontsize=title_size, fontweight='bold')

        fig_path = os.path.join('./figs/mot', 'loss_trend-optimization-%s.pdf' % dataset)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.clf()

        # split line
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), dpi=500)
        ax.plot(x, masking_lss_h[::-1], color='#DA422A', linestyle='-', linewidth=2.0, label='Hard')
        ax.fill_between(x, 0, masking_lss_h[::-1], facecolor='red', alpha=0.15)
        ax.plot(x, masking_lss_e[::-1], color='#555D9E', linestyle='-', linewidth=2.0, label='Easy')
        ax.fill_between(x, 0, masking_lss_e[::-1], facecolor='blue', alpha=0.2)
        ax.plot(x, masking_lss_m[::-1], color='black', linestyle='dashed', linewidth=2.0, label='Test')

        ax.set_xlabel(x_label_um, fontsize=label_size, fontweight='bold')
        ax.set_xlim(1/50, 1.0)
        ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')
        ax.set_ylim(0.0, 1.0)

        ax.set_xticks(x_, x_ticks_um)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(prop={'size': legend_size}, loc=2)

        for loc in ['bottom', 'left', 'top', 'right']:
            ax.spines[loc].set_linewidth(spine_width)

        ax.grid(visible=False)
        # ax.set_title('(a)', fontsize=24, fontweight='bold')
        # fig.suptitle('(a)', fontsize=title_size, fontweight='bold')

        fig_path = os.path.join('./figs/mot', 'loss_trend-unmasking-%s.pdf' % dataset)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50', 'vit_small', 'vit_base'], default='resnet18')
    args = parser.parse_args()

    main(args)