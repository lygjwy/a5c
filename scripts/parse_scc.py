'''
parse the spearman correlation coefficient (rho) between the lc-optimization and lc-unmasking
'''

import os
import argparse

import numpy as np
import scipy.stats as stats

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


def main(args):

    for dataset in args.datasets:
        # print('-------------------- %s --------------------' % dataset)
        for arch in args.archs:
            # print('--------------- %s ---------------' % arch)
            for pretrain in args.pretrains:
                cidxs = extract_cidxs(dataset)

                # print('---------- %s ----------' % pretrain)
                root = os.path.join(args.root, 'seed_'+str(args.seed), dataset)
                ls_op_path = os.path.join(root, 'cle', '-'.join(['pt_'+pretrain, 'arch_'+arch]), 'ls.npy')
                ls_op = np.load(ls_op_path)

                for principle in args.principles:
                    sccs = []
                    ls_um_path = os.path.join(root, principle, '-'.join(['pt_'+pretrain, 'arch_'+arch]), 'ls.npy')
                    ls_um = np.load(ls_um_path)
                    
                    for c_idx in cidxs:
                        ls_op_c = ls_op[c_idx]
                        ls_um_c = ls_um[c_idx]
                    
                        res = stats.spearmanr(ls_op_c, ls_um_c)
                        rho = res.statistic
                        sccs.append(rho)

                    print(float('%.2f' % np.mean(sccs)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./idxs')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pretrains', nargs='+', default=['weakly', 'fully'])
    parser.add_argument('--datasets', nargs='+', default=['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch'])

    parser.add_argument('--archs', nargs='+', default=['resnet18', 'resnet50', 'vit_small', 'vit_base'])
    parser.add_argument('--principles', nargs='+', default=['random', 'herding', 'kcentergreedy', 'leastconfidence', 'entropy', 'margin', 'forgetting', 'grand', 'el2n', 'contextualdiversity'])

    args = parser.parse_args()
    main(args)