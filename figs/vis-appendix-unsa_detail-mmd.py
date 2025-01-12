'''
collect mmd of different sampling
'''

import os
# PAR_DIR=$(pwd); export PYTHONPATH="${PAR_DIR}:$PYTHONPATH"

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset


from datasets import get_dataset
from models import get_cclf


def extract_datasets(dataset):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # IN-1K
    
    data_trfms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    datasets = {mode: get_dataset('/mnt/sharedata/hdd/jwy/datasets', dataset, mode, data_trfms[mode]) for mode in ['train', 'val', 'test']}

    return datasets

def extract_idxcs(dataset, num_classes):
    
    c_idxs = []
    for i in range(num_classes):
        c_idxs.append([])
    
    length = len(dataset)
    for i in range(length):
        _, y = dataset[i]
        c_idxs[y].append(i)

    return c_idxs


def extract_zs(data_loader, num_classes, model):

    zs = []
    m = nn.AdaptiveAvgPool2d(1)

    arch, ckpt = model
    cclf = get_cclf(arch, num_classes, ckpt)
    if torch.cuda.is_available():
        cclf.cuda()
    cudnn.benchmark = True

    cclf.eval()
    for data in data_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

        with torch.no_grad():
            z = m(cclf.forward_features(inputs)).squeeze()
            zs.append(z.cpu().numpy())
    
    zs = np.concatenate(zs, axis=0)

    return zs


def cal_mmd(zs_p, zs_q):
    K_pp = rbf_kernel(zs_p)
    K_qq = rbf_kernel(zs_q)
    K_pq = rbf_kernel(zs_p, zs_q)

    n = zs_p.shape[0]
    m = zs_q.shape[0]

    mmd = (K_pp.sum() / (n*n) + K_qq.sum() / (m*m) - 2 * K_pq.sum() / (n*m))

    return np.sqrt(mmd)


def main():

    datasets = ['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch', 'Average']
    mmds_afd, mmds_aud = [], []
    # collect data
    for i, dataset in enumerate(datasets[:5]):
    
        data_sets = extract_datasets(dataset)
        num_classes = data_sets['train'].num_classes
        train_set = ConcatDataset([data_sets['train'], data_sets['val']])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=8)

        ## /mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/Sketch-weakly-resnet18/all
        last_model_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/', '-'.join([dataset, 'fully', 'resnet18']), 'all', '50.pth')
        model = ('resnet18', last_model_path)
        zs = extract_zs(train_loader, num_classes, model)

        # mmd for different scores
        ## mmd for finetuning
        mmds_afs = []
        for principle in ['random', 'cle', 'clh', 'cl-se', 'cl-sh']:
            mmd_s = []
            for j in range(1, 10):
                # /mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/Sketch/cl-s95/pt_fully-arch_resnet18/0.1.npy
                if 'se' in principle:
                    split = str(int(j * 10 / 2))
                    principle_ = 'cl-s%s' % split
                    idxs_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, principle_, 'pt_fully-arch_resnet18', str(j/10)+'.npy')
                elif 'sh' in principle:
                    split = str(int(100 - j * 10 / 2))
                    principle_ = 'cl-s%s' % split
                    idxs_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, principle_, 'pt_fully-arch_resnet18', str(j/10)+'.npy')
                else:
                    idxs_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, principle, 'pt_fully-arch_resnet18', str(j/10)+'.npy')
                idxs = np.load(idxs_path)
                
                zs_ = zs[idxs]
                mmd_ = cal_mmd(zs_, zs)
                mmd_s.append(mmd_)
            mmds_afs.append(mmd_s)

        mmds_afd.append(np.array(mmds_afs))
    
        ## mmd for unmasking
        mmds_aus = []
        for principle in ['random', 'lwot-l1-e-i20_f50', 'lwot-l1-h-i20_f50', 'lwot-l1-se-i20_f50', 'lwot-l1-sh-i20_f50']:
            mmd_s = []
            for j in range(1, 10):
                # /mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/Sketch/lwot-l1-s95-i20_f50/pt_fully-arch_resnet18/0.1.npy
                if 'se' in principle:
                    split = str(int(j * 10 / 2))
                    principle_ = 'lwot-l1-s%s-i20_f50' % split
                    idxs_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, principle_, 'pt_fully-arch_resnet18', str(j/10)+'.npy')
                elif 'sh' in principle:
                    split = str(int(100 - j * 10 / 2))
                    principle_ = 'lwot-l1-s%s-i20_f50' % split
                    idxs_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, principle_, 'pt_fully-arch_resnet18', str(j/10)+'.npy')
                else:
                    idxs_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, principle, 'pt_fully-arch_resnet18', str(j/10)+'.npy')
                idxs = np.load(idxs_path)
                
                zs_ = zs[idxs]
                mmd_ = cal_mmd(zs_, zs)
                mmd_s.append(mmd_)
            mmds_aus.append(mmd_s)

        mmds_aud.append(np.array(mmds_aus))

    # calculate the average
    mmds_afd.append(np.average(np.stack(mmds_afd, axis=0), axis=0))
    mmds_aud.append(np.average(np.stack(mmds_aud, axis=0), axis=0))

    # plot
    ## config
    x = [i / 10 for i in range(1, 10)]
    x_ticks = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
    x_label = 'Pruning Ratio'
    y_label = 'MMD'

    spine_width = 2.4
    label_size = 24
    # title_size = 24
    tick_size = 16
    legend_size = 12

    for i, dataset in enumerate(datasets):
                
        mmds_fr, mmds_fe, mmds_fh, mmds_fre, mmds_frh = mmds_afd[i]
        mmds_ur, mmds_ue, mmds_uh, mmds_ure, mmds_urh = mmds_aud[i]

        ## mmd for fine-tuning score
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=500, layout='constrained')

        ax.plot(x, mmds_fr[::-1], color='gray', marker='v', linestyle='dotted', linewidth=2.0, label='R')
        ax.plot(x, mmds_fe[::-1], color='blue', marker='v', linestyle='dotted', linewidth=2.0, label='T(E)')
        ax.plot(x, mmds_fre[::-1], color='blue', marker='v', linestyle='-', linewidth=2.0, label='R + T(E)')
        ax.plot(x, mmds_fh[::-1], color='red', marker='v', linestyle='dotted', linewidth=2.0, label='T(H)')
        ax.plot(x, mmds_frh[::-1], color='red', marker='v', linestyle='-', linewidth=2.0, label='R + T(H)')

        ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')
        ax.set_xticks(x, x_ticks)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(prop={'size': legend_size}, loc=2)

        for loc in ['bottom', 'left', 'top', 'right']:
            ax.spines[loc].set_linewidth(spine_width)
        ax.grid(visible=True, linestyle=':')
    
        fig_path = os.path.join('./figs', 'ms-ft-%s.pdf' % dataset)
        fig.savefig(fig_path)
        plt.clf()

        ## mmd for un-masking score
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=500, layout='constrained')

        ax.plot(x, mmds_ur[::-1], color='gray', marker='v', linestyle='dotted', linewidth=2.0, label='R')
        ax.plot(x, mmds_ue[::-1], color='blue', marker='v', linestyle='dotted', linewidth=2.0, label='T(E)')
        ax.plot(x, mmds_ure[::-1], color='blue', marker='v', linestyle='-', linewidth=2.0, label='R + T(E)')
        ax.plot(x, mmds_uh[::-1], color='red', marker='v', linestyle='dotted', linewidth=2.0, label='T(H)')
        ax.plot(x, mmds_urh[::-1], color='red', marker='v', linestyle='-', linewidth=2.0, label='R + T(H)')

        ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')
        ax.set_xticks(x, x_ticks)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(prop={'size': legend_size}, loc=2)

        for loc in ['bottom', 'left', 'top', 'right']:
            ax.spines[loc].set_linewidth(spine_width)
        ax.grid(visible=True, linestyle=':')
    
        fig_path = os.path.join('./figs', 'ms-um-%s.pdf' % dataset)
        fig.savefig(fig_path)
        plt.clf()


if __name__ == '__main__':

    main()