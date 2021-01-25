RUN_NUMBER=1
GPU=1

LR=0.1
OPTIMIZER='mas'
P=0.125
WD='2.5e-2'
MODEL='resnet'

mkdir -p "logs/${MODEL}"

CUDA_VISIBLE_DEVICES="${GPU}" python3 run_cnn_test_cifar10.py  --method="${OPTIMIZER}" --net="${MODEL}" \
        --lr="${LR}" --partial="${P}" --wd="${WD}" \
        > "logs/${MODEL}/${OPTIMIZER}_lr_${LR}_p_=${P}_wd_=${WD}_${RUN_NUMBER}.log"
