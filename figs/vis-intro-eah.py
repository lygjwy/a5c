'''
Visualize easy and hasy samples based on cumulative fine-tuning loss
'''

import os
import random
import numpy as np

import torch
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms

from datasets import get_dataset


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_ds_cs(dataset):
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

    length = len(train_sets)

    cs = []
    for i in range(num_classes):
        cs.append([])
    
    for i in range(length):
        _, y = dataset[i]
        cs[y].append(i)
    
    return train_sets, cs


def main(args):

    init_seed(args.seed)

    # saving dir for hardest and easiest imgs
    demo_dir = os.path.join('./eah', args.dataset, args.arch)
    os.makedirs(demo_dir, exist_ok=True)

    datasets = ['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'Sketch', 'Average']
    for dataset in datasets:
        train_set, cs = extract_ds_cs(dataset)

        ft_ls_path = os.path.join('/mnt/sharedata/hdd/jwy/A3C/idxs/seed_1/', dataset, 'cle/pt_fully-arch_resnet18/ls.npy')
        ft_ls = np.load(ft_ls_path)
        for i, cidxs in enumerate(cs):
            ft_ls_c = ft_ls[cidxs]
            
            h_idx = cidxs[np.argmax(ft_ls_c)]
            e_idx = cidxs[np.argmin(ft_ls_c)]

            h_img, _ = train_set[h_idx]
            e_img, _ = train_set[e_idx]

            h_name = '-'.join([str(i), 'h', str(h_idx)]) + '.png'
            h_img.save(os.path.join(demo_dir, h_name))

            e_name = '-'.join([str(i), 'e', str(e_idx)]) + '.png'
            e_img.save(os.path.join(demo_dir, e_name))
    

if __name__ == '__main__':

    main()