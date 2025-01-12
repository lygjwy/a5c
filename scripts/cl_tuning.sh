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

GPUs=$3
if [[ -z $GPUs ]]; then
    echo 'PLEASE INPUT GPUs'
    exit
fi

export CUDA_VISIBLE_DEVICES=$GPUs

echo ${DATASET} - ${RATIO}


for principle in cle clh ;
do
    echo ${principle}
    python3 finetune.py --seed 1 \
    --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
    --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
    --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
    --dataset $DATASET --pruning --principle ${principle} --ratio $RATIO --pretrain fully --arch resnet18
done

for split in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 ;
do
    # specify the range
    lower_bound=$((${RATIO}/2))
    upper_bound=$((100-${RATIO}/2))
    
    if [ $split -ge $lower_bound ] && [ $split -le $upper_bound ]; then
        principle=cl-s${split}
        echo ${principle}
        python3 finetune.py --seed 1 \
        --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
        --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
        --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
        --dataset $DATASET --pruning --principle ${principle} --ratio $RATIO --pretrain fully --arch resnet18
    fi
done
