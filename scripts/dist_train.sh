#!/bin/bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-6699}

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
    exit
fi
TORCH_DISTRIBUTED_DEBUG=DETAIL \
PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
HF_ENDPOINT=https://hf-mirror.com \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3} 