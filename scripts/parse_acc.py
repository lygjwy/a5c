import os
import re
import argparse


def main(args):

    for dataset in args.datasets:
        # print('-------------------- %s --------------------' % dataset)
        for arch in args.archs:
            # print('--------------- %s ---------------' % arch)
            for pretrain in args.pretrains:
                # print('---------- %s ----------' % pretrain)
                # /data/home/jwy/opendatasel/outputs/seed_1/CXRB10-fully-vit_base/hpscore-i1_f10-0.1/outputs.log
                root = os.path.join(args.root, 'seed_'+str(args.seed), '-'.join([dataset, pretrain, arch]))

                for principle in args.principles:
                    accs = []
                    for i in range(1, 10):
                    # for i in range(5, 10):
                        log_dir = '-'.join([principle, str(i/10)])
                        log_path = os.path.join(root, log_dir, 'outputs.log')

                        if os.path.isfile(log_path):
                            with open(log_path, 'r') as f:
                                logs_data = f.readlines()
                                accs.extend([str(float(re.findall("\d+\.\d+", log_data)[0])) for log_data in logs_data if log_data.startswith('---> Acc:')])

                    print(','.join(accs))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./ckpts')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pretrains', nargs='+', default=['weakly', 'fully'])
    parser.add_argument('--datasets', nargs='+', default=['CXRB10', 'DeepWeeds', 'DTD', 'FGVCAircraft', 'RESISC45', 'Sketch'])

    parser.add_argument('--archs', nargs='+', default=['resnet18', 'resnet50', 'vit_small', 'vit_base'])
    parser.add_argument('--principles', nargs='+', default=['random', 'herding', 'kcentergreedy', 'leastconfidence', 'entropy', 'margin', 'forgetting', 'grand', 'el2n', 'contextualdiversity'])

    args = parser.parse_args()
    main(args)