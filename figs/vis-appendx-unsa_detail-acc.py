'''
collect accuracy gap of different sampling
'''

import os
# PAR_DIR=$(pwd); export PYTHONPATH="${PAR_DIR}:$PYTHONPATH"

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


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
    accs_afd, accs_aud = [], []

    # collect data
    for i, dataset in enumerate(datasets[:5]):
    
        data_sets = extract_datasets(dataset)
        num_classes = data_sets['train'].num_classes
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
        for principle in ['random', 'lwot-l1-e-i20_f50', 'lwot-l1-h-i20_f50', 'lwot-l1-se-i20_f50', 'lwot-l1-sh-i20_f50']:
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
    
    # calculate the average
    accs_afd.append(np.average(np.stack(accs_afd, axis=0), axis=0))
    accs_aud.append(np.average(np.stack(accs_aud, axis=0), axis=0))

    # plot
    ## config
    x = [i / 10 for i in range(1, 10)]
    x_ticks = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
    x_label = 'Pruning Ratio'
    y_label = 'Accuracy Gap (%)'

    spine_width = 2.4
    label_size = 24
    # title_size = 24
    tick_size = 16
    legend_size = 12

    for i, dataset in enumerate(datasets):
        
        accs_fr, accs_fe, accs_fh, accs_fre, accs_frh = accs_afd[i]
        accs_ur, accs_ue, accs_uh, accs_ure, accs_urh = accs_aud[i]

        ## acc for fine-tuning score
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=500, layout='constrained')

        # ax_acc.plot(x, accs_r[::-1], color='gray', marker='^', linestyle='dotted', linewidth=2.0, label='R')
        ax.plot(x, accs_fe[::-1]-accs_fr[::-1], color='blue', marker='^', linestyle='dotted', linewidth=2.0, label='T(E)')
        ax.plot(x, accs_fre[::-1]-accs_fr[::-1], color='blue', marker='^', linestyle='-', linewidth=2.0, label='R + T(E)')
        ax.plot(x, accs_fh[::-1]-accs_fr[::-1], color='red', marker='^', linestyle='dotted', linewidth=2.0, label='T(H)')
        ax.plot(x, accs_frh[::-1]-accs_fr[::-1], color='red', marker='^', linestyle='-', linewidth=2.0, label='R + T(H)')

        ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')
        ax.set_xticks(x, x_ticks)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(prop={'size': legend_size}, loc=3)

        for loc in ['bottom', 'left', 'top', 'right']:
            ax.spines[loc].set_linewidth(spine_width)

        ax.grid(visible=True, linestyle=':')
        fig_path = os.path.join('./figs', 'accs-ft-%s.pdf' % dataset)
        fig.savefig(fig_path)
        plt.clf()

        ## acc for un-masking score
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.944), dpi=500, layout='constrained')

        # ax_acc.plot(x, accs_r[::-1], color='gray', marker='^', linestyle='dotted', linewidth=2.0, label='R')
        ax.plot(x, accs_ue[::-1]-accs_ur[::-1], color='blue', marker='^', linestyle='dotted', linewidth=2.0, label='T(E)')
        ax.plot(x, accs_ure[::-1]-accs_ur[::-1], color='blue', marker='^', linestyle='-', linewidth=2.0, label='R + T(E)')
        ax.plot(x, accs_uh[::-1]-accs_ur[::-1], color='red', marker='^', linestyle='dotted', linewidth=2.0, label='T(H)')
        ax.plot(x, accs_urh[::-1]-accs_ur[::-1], color='red', marker='^', linestyle='-', linewidth=2.0, label='R + T(H)')

        ax.set_xlabel(x_label, fontsize=label_size, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=label_size, fontweight='bold')
        ax.set_xticks(x, x_ticks)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(prop={'size': legend_size}, loc=3)

        for loc in ['bottom', 'left', 'top', 'right']:
            ax.spines[loc].set_linewidth(spine_width)

        ax.grid(visible=True, linestyle=':')
        fig_path = os.path.join('./figs', 'accs-um-%s.pdf' % dataset)
        fig.savefig(fig_path)
        plt.clf()


if __name__ == '__main__':

    main()