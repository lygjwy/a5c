SEED=1

PRETRAIN=fully

ARCH=resnet18

DATASET=$1
if [[ -z $ARCH ]]; then
    echo 'PLEASE INPUT DATASET'
    exit
fi

echo ${SEED} - ${PRETRAIN} - ${ARCH} - ${DATASET}

GPUs=$2
if [[ -z $GPUs ]]; then
    echo 'PLEASE INPUT GPUs'
    exit
fi

export CUDA_VISIBLE_DEVICES=$GPUs

DATASETS_DIR='/mnt/sharedata/hdd/jwy/datasets'
CKPTS_DIR='/mnt/sharedata/hdd/jwy/A3C/ckpts'
IDXS_DIR='/mnt/sharedata/hdd/jwy/A3C/idxs'

# masking the pre-trained model with different steps
# python3 score-wot.py --seed ${SEED} --datasets_dir $DATASETS_DIR --ckpts_dir $CKPTS_DIR --idxs_dir $IDXS_DIR --dataset ${DATASET} --pretrain ${PRETRAIN} --arch ${ARCH} --interval 1.0 --frequency 1
# python3 score-wot.py --seed ${SEED} --datasets_dir $DATASETS_DIR --ckpts_dir $CKPTS_DIR --idxs_dir $IDXS_DIR --dataset ${DATASET} --pretrain ${PRETRAIN} --arch ${ARCH} --interval 0.1 --frequency 10
# python3 score-wot.py --seed ${SEED} --datasets_dir $DATASETS_DIR --ckpts_dir $CKPTS_DIR --idxs_dir $IDXS_DIR --dataset ${DATASET} --pretrain ${PRETRAIN} --arch ${ARCH} --interval 0.05 --frequency 20
# python3 score-wot.py --seed ${SEED} --datasets_dir $DATASETS_DIR --ckpts_dir $CKPTS_DIR --idxs_dir $IDXS_DIR --dataset ${DATASET} --pretrain ${PRETRAIN} --arch ${ARCH} --interval 0.033 --frequency 30
# python3 score-wot.py --seed ${SEED} --datasets_dir $DATASETS_DIR --ckpts_dir $CKPTS_DIR --idxs_dir $IDXS_DIR --dataset ${DATASET} --pretrain ${PRETRAIN} --arch ${ARCH} --interval 0.025 --frequency 40

# masking the pre-trained model with random strategy
for SEED in 1 2 3 4 5 6 7 8 9 ;
do
    for m in random l1 ;
    do
        python3 score-wot.py --seed ${SEED} --datasets_dir $DATASETS_DIR --ckpts_dir $CKPTS_DIR --idxs_dir $IDXS_DIR --dataset ${DATASET} --pretrain ${PRETRAIN} --arch ${ARCH} --masking $m --interval 0.02 --frequency 50
    done
done