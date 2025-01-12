DATASET=$1
if [[ -z $DATASET ]]; then
    echo 'PLEASE INPUT DATASET'
    exit
fi

RATIO=$2
if [[ -z $RATIO ]]; then
    echo 'PLEASE INPUT RATIO'
    exit
fi

SPLIT=$3
if [[ -z $SPLIT ]]; then
    echo 'PLEASE INPUT SPLIT'
    exit
fi

GPUs=$4
if [[ -z $GPUs ]]; then
    echo 'PLEASE INPUT GPUs'
    exit
fi

export CUDA_VISIBLE_DEVICES=$GPUs

SEED=1
PRETRAIN=fully
ARCH=resnet18

# for p in lwot-l1-s${SPLIT}-i1000_f1 lwot-l1-s${SPLIT}-i100_f10 lwot-l1-s${SPLIT}-i50_f20 lwot-l1-s${SPLIT}-i33_f30 lwot-l1-s${SPLIT}-i25_f40 cl-s${SPLIT} ;
# do
#     echo ${SEED} - ${PRETRAIN} - ${ARCH} - ${DATASET} - ${RATIO} - ${p}
#     python3 finetune.py --seed ${SEED} \
#     --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
#     --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
#     --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
#     --dataset $DATASET --pruning --principle $p --ratio $RATIO --pretrain $PRETRAIN --arch $ARCH
# done

# p=lwot-random-s${SPLIT}-i20_f50
for SEED in 1 2 3 4 5 6 7 8 9 ;
do
    for p in lwot-random-s${SPLIT}-i20_f50 lwot-l1-s${SPLIT}-i20_f50 ;
    do
        echo ${SEED} - ${PRETRAIN} - ${ARCH} - ${DATASET} - ${RATIO} - ${p}
        python3 finetune.py --seed ${SEED} \
        --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
        --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
        --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
        --dataset $DATASET --pruning --principle $p --ratio $RATIO --pretrain $PRETRAIN --arch $ARCH
    done
done
