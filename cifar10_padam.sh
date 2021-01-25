LR=0.1
OPTIMIZER='padam'
P=0.125
WD='2.5e-2'
MODEL='resnet'

mkdir -p "logs/${MODEL}"

python run_cnn_test_cifar10.py  --method="${OPTIMIZER}" --net="${LR}" \
        --lr="${LR}" --partial="${LR}" --wd="${LR}" \
        > "logs/${MODEL}/${OPTIMIZER}_lr_${LR}_p_=${P}_wd_=${WD}_${RUN_NUMBER}.log"