SEED=$1
if [[ -z $SEED ]]; then
    echo 'PLEASE INPUT SEED'
    exit
fi

DATASET=$2
if [[ -z $DATASET ]]; then
    echo 'PLEASE INPUT DATASET'
    exit
fi

GPUs=$3
if [[ -z $GPUs ]]; then
    echo 'PLEASE INPUT GPUs'
    exit
fi

export CUDA_VISIBLE_DEVICES=$GPUs

echo ${SEED} - ${DATASET}
for ratio in 50 60 ;
do
    echo "------ RATIO ${ratio} ------"
    for split in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 ;
    do
        echo lwot-l1-s${split}-i20_f50
        python3 finetune.py --seed ${SEED} \
        --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
        --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
        --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
        --dataset $DATASET --pruning --principle lwot-l1-s${split}-i20_f50 --ratio $ratio --pretrain fully --arch resnet18
    done
done
