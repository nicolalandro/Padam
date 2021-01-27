RUN_NUMBER=1
GPU=0

LR=0.01
OPTIMIZER='mps'
P=0.125
WD='2.5e-2'
MODEL='resnet'
MOMENTUM=0

mkdir -p "logs/${MODEL}"

CUDA_VISIBLE_DEVICES="${GPU}" python3 run_cnn_test_cifar10.py  --method="${OPTIMIZER}" --net="${MODEL}" \
        --lr="${LR}" --partial="${P}" --wd="${WD}" --momentum="${MOMENTUM}"\
        > "logs/${MODEL}/${OPTIMIZER}_lr_${LR}_p_${P}_wd_${WD}_mom_${MOMENTUM}_${RUN_NUMBER}.log"
