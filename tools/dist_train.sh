#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

base_path=`echo ${CONFIG%.*}`
base_path=`echo ${base_path#*/}`
CONFIG=configs/${base_path}.py
work_path=work_dirs/${base_path}
ckpt_file=${work_path}/latest.pth
map_file=${work_path}/map.txt
flops_file=${work_path}/model_flops.txt

mkdir -p ${work_path}

## get_flops
python tools/get_flops.py ${CONFIG} | tee ${flops_file}

## train
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

## publish
python tools/publish_model.py ${ckpt_file} ${ckpt_file}