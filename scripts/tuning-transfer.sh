ARCH=$1
if [[ -z $ARCH ]]; then
    echo 'PLEASE INPUT ARCH'
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

echo ${SEED} - ${PRETRAIN} - ${ARCH} - ${DATASET}
for ratio in 30 60 90 ;
do
    echo "------ RATIO ${ratio} ------"

    for split in 10 20 30 40 50 60 70 80 90 ;
    do
        # specify the range
        lower_bound=$((${ratio}/2))
        upper_bound=$((100-${ratio}/2))
        
        if [ $split -ge $lower_bound ] && [ $split -le $upper_bound ]; then
            # for principle in loss lwot ; do
            for principle in lwot ; do
                echo ${principle}-l1-s${split}-i20_f50
                python3 finetune-transfer.py --seed 1 \
                --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts-transfer \
                --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
                --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
                --dataset $DATASET --pruning --principle ${principle}-l1-s${split}-i20_f50 --ratio $ratio --pretrain fully --arch $ARCH
            done
        fi
    done
done
