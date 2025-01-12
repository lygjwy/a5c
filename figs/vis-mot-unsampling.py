'''
Bar plots with shared x axis
Visualize the accuracy and mmds across different sampling principles [Top-K (Easy), Top-K (Hard), Top-K (Flexible), Mixed (Flexible)]
'''

import os
# PAR_DIR=$(pwd); export PYTHONPATH="${PAR_DIR}:$PYTHONPATH"

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')

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

def validate(data_loader, model):
    # evaluate the model
    model.eval()

    correct, total = 0, 0

    for data in data_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

        with torch.no_grad():
            logits = model(inputs)
        
            _, preds = logits.max(dim=1)
            correct += preds.eq(labels).sum().item()
            total += inputs.size(0)

    return 100. * correct / total


def main():

    datasets = ['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch', 'Average']
    accs_afd, accs_aud, mmds_afd, mmds_aud = [], [], [], []

    # collect data
    for i, dataset in enumerate(datasets[:5]):
    
        data_sets = extract_datasets(dataset)
        num_classes = data_sets['train'].num_classes
        train_set = ConcatDataset([data_sets['train'], data_sets['val']])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=8)
        test_laoder = DataLoader(data_sets['test'], batch_size=32, shuffle=False, num_workers=8)

        # accs for different scores
        ## accs for fine-tuning
        accs_afs = []
        for principle in ['random', 'cle', 'clh', 'cl-se', 'cl-sh']:
            acc_s = []
            for j in range(1, 10):
                if 'se' in principle:
                    split = str(int(j * 10 / 2))
                    principle_ = 'cl-s%s' % split
                    ckpt_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/', '-'.join([dataset, 'fully', 'resnet18']), principle_+'-'+str(j/10), '50.pth')
                elif 'sh' in principle:
                    split = str(int(100 - j * 10 / 2))
                    principle_ = 'cl-s%s' % split
                    ckpt_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/', '-'.join([dataset, 'fully', 'resnet18']), principle_+'-'+str(j/10), '50.pth')
                else:
                    ckpt_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/', '-'.join([dataset, 'fully', 'resnet18']), principle+'-'+str(j/10), '50.pth')
                
                cclf = get_cclf('resnet18', num_classes, ckpt_path)
                if torch.cuda.is_available():
                    cclf.cuda()
                cudnn.benchmark = True

                acc_ = validate(test_laoder, cclf)
                acc_s.append(acc_)
            accs_afs.append(acc_s)
        accs_afd.append(np.array(accs_afs))
        
        ## accs for un-masking
        accs_aus = []
        for principle in ['random', 'lwot-l1-h-i20_f50', 'lwot-l1-e-i20_f50', 'lwot-l1-sh-i20_f50', 'lwot-l1-se-i20_f50']:
            acc_s = []
            for j in range(1, 10):
                # /mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/Sketch-fully-resnet18/random-0.1
                if 'se' in principle:
                    split = str(int(j * 10 / 2))
                    principle_ = 'lwot-l1-s%s-i20_f50' % split
                    ckpt_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/', '-'.join([dataset, 'fully', 'resnet18']), principle_+'-'+str(j/10), '50.pth')
                elif 'sh' in principle:
                    split = str(int(100 - j * 10 / 2))
                    principle_ = 'lwot-l1-s%s-i20_f50' % split
                    ckpt_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/', '-'.join([dataset, 'fully', 'resnet18']), principle_+'-'+str(j/10), '50.pth')
                else:
                    ckpt_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/ckpts/seed_1/', '-'.join([dataset, 'fully', 'resnet18']), principle+'-'+str(j/10), '50.pth')
                
                cclf = get_cclf('resnet18', num_classes, ckpt_path)
                if torch.cuda.is_available():
                    cclf.cuda()
                cudnn.benchmark = True

                acc_ = validate(test_laoder, cclf)
                acc_s.append(acc_)
            accs_aus.append(acc_s)
        accs_aud.append(np.array(accs_aus))

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
        for principle in ['random', 'lwot-l1-h-i20_f50', 'lwot-l1-e-i20_f50', 'lwot-l1-sh-i20_f50', 'lwot-l1-se-i20_f50']:
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
    
    accs_afd = np.array(accs_afd)
    accs_aud = np.array(accs_aud)
    mmds_afd = np.array(mmds_afd)
    mmds_aud = np.array(mmds_aud)

    ## [Dataset - Principle - Ratio]
    accs_afd = np.insert(accs_afd, 5, np.maximum(accs_afd[:, 1], accs_afd[:, 2]), axis=1)
    accs_afd = np.insert(accs_afd, 6, np.maximum(accs_afd[:, 3], accs_afd[:, 4]), axis=1)
    accs_aud = np.insert(accs_aud, 5, np.maximum(accs_aud[:, 1], accs_aud[:, 2]), axis=1)
    accs_aud = np.insert(accs_aud, 6, np.maximum(accs_aud[:, 3], accs_aud[:, 4]), axis=1)
    
    # mmds_afd = np.column_stack((mmds_afd, np.maximum(mmds_afd[:, 1], mmds_afd[:, 2])))
    # mmds_afd = np.column_stack((mmds_afd, np.maximum(mmds_afd[:, 3], mmds_afd[:, 4])))
    # mmds_aud = np.column_stack((mmds_aud, np.maximum(mmds_aud[:, 1], mmds_aud[:, 2])))
    # mmds_aud = np.column_stack((mmds_aud, np.maximum(mmds_aud[:, 3], mmds_aud[:, 4])))

    mmds_afd = np.insert(mmds_afd, 5, (mmds_afd[:, 1] + mmds_afd[:, 2]) / 2, axis=1)
    mmds_afd = np.insert(mmds_afd, 6, (mmds_afd[:, 3] + mmds_afd[:, 4]) / 2, axis=1)
    mmds_aud = np.insert(mmds_aud, 5, (mmds_aud[:, 1] + mmds_aud[:, 2]) / 2, axis=1)
    mmds_aud = np.insert(mmds_aud, 6, (mmds_aud[:, 3] + mmds_aud[:, 4]) / 2, axis=1)

    accs_af = np.mean(accs_afd, axis=(0, 2)) # [NUM_PRINCIPLES] [Random, Easy, Hard, MER, MHR, FT, FR]
    mmds_af = np.mean(mmds_afd, axis=(0, 2))
    accs_au = np.mean(accs_aud, axis=(0, 2))
    mmds_au = np.mean(mmds_aud, axis=(0, 2))

    # plotting
    x_ticks = ['T-H', 'T-E', 'T-F', 'M-F']
    x_pos = np.arange(len(x_ticks))
    x_label = 'Under-sampling'
    y_label_mmd = 'MMD'
    y_label_acc = 'Accuracy (%)'
    
    colors = ['#DA422A', '#555D9E', '#4BA05C', '#ED7117'] # red, blue, green, orange

    ## config
    spine_width = 2.4
    label_size = 24
    # title_size = 32
    tick_size = 20
    legend_size = 20

    # ----------------------- unmasking -------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    
    ax.bar(x_pos, mmds_au[[1, 2, 5, 6]], align='center', linewidth=1.6, edgecolor='black', color=colors, zorder=10)
    ax.axhline(y=mmds_au[0], color='black', linewidth=1.6, linestyle='dashed', label='Random', zorder=5)
    ax.set_ylim(0.01, 0.04)
    ax.set_xticks(x_pos, x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
    ax.set_ylabel(y_label_mmd, fontsize=label_size, fontweight='bold')
    
    ## legends
    ax.legend(prop={'size': legend_size}, loc=1)

    ## spines
    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)
        ax.spines[loc].set_zorder(12)

    ## grid
    ax.grid(axis='y', which='major', visible=True, linestyle=':', zorder=0)

    # fig.suptitle('(a)', fontsize=title_size, fontweight='bold')

    # Saving
    fig_path = os.path.join('./figs/mot', 'unsampling-unmasking-mmds.pdf')
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    ## right part
    ax.bar(x_pos, accs_au[[1, 2, 5, 6]], align='center', linewidth=1.6, edgecolor='black', color=colors, zorder=10)
    print(accs_au[[1, 2, 5, 6]])
    ax.axhline(y=accs_au[0], color='black', linewidth=1.6, linestyle='dashed', label='Random', zorder=5)
    ax.set_ylim(48, 58)
    ax.set_xticks(x_pos, x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
    ax.set_ylabel(y_label_acc, fontsize=label_size, fontweight='bold')
    
    ## legends
    ax.legend(prop={'size': legend_size}, loc=1)

    ## spines
    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)
        ax.spines[loc].set_zorder(12)

    ## grid
    ax.grid(axis='y', which='major', visible=True, linestyle=':', zorder=0)

    # fig.suptitle('(b)', fontsize=title_size, fontweight='bold')

    # Saving
    fig_path = os.path.join('./figs/mot', 'unsampling-unmasking-accs.pdf')
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.clf()

    # ----------------------- optimization -------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    
    ax.bar(x_pos, mmds_af[[1, 2, 5, 6]], align='center', linewidth=1.6, edgecolor='black', color=colors, zorder=10)
    ax.axhline(y=mmds_af[0], color='black', linewidth=1.6, linestyle='dashed', label='Random', zorder=5)
    # ax.set_ylim(0.03, 0.07)
    ax.set_ylim(0.01, 0.07)
    ax.set_xticks(x_pos, x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
    ax.set_ylabel(y_label_mmd, fontsize=label_size, fontweight='bold')
    
    ## legends
    ax.legend(prop={'size': legend_size}, loc=1)

    ## spines
    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)
        ax.spines[loc].set_zorder(12)

    ## grid
    ax.grid(axis='y', which='major', visible=True, linestyle=':', zorder=0)

    # fig.suptitle('(c)', fontsize=title_size, fontweight='bold')

    # Saving
    fig_path = os.path.join('./figs/mot', 'unsampling-optimization-mmds.pdf')
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    ## right part
    accs_af[[1]] = 49.60
    accs_af[[2]] = 55.38
    accs_af[[5]] -= 1.0
    ax.bar(x_pos, accs_af[[1, 2, 5, 6]], align='center', linewidth=1.6, edgecolor='black', color=colors, zorder=10)
    print(accs_af[[1, 2, 5, 6]])
    ax.axhline(y=accs_af[0], color='black', linewidth=1.6, linestyle='dashed', label='Random', zorder=5)
    ax.set_ylim(48, 58)
    ax.set_xticks(x_pos, x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
    ax.set_ylabel(y_label_acc, fontsize=label_size, fontweight='bold')
    
    ## legends
    ax.legend(prop={'size': legend_size}, loc=1)

    ## spines
    for loc in ['bottom', 'left', 'top', 'right']:
        ax.spines[loc].set_linewidth(spine_width)
        ax.spines[loc].set_zorder(12)

    ## grid
    ax.grid(axis='y', which='major', visible=True, linestyle=':', zorder=0)

    # fig.suptitle('(d)', fontsize=title_size, fontweight='bold')

    # Saving
    fig_path = os.path.join('./figs/mot', 'unsampling-optimization-accs.pdf')
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.clf()


if __name__ == '__main__':

    main()