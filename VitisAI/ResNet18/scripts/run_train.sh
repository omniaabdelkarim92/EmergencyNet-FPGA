#!/bin/bash

# Copyright © 2025 Omnia Abdelkariem 
# Egypt Japan University for Science and Technology 

echo " "

echo "Running Training..."
echo " "

DATA_DIR=./build/data/
WEIGHTS=./build/float
DATASET=AIDER
GPU_ID=0
export PYTHONPATH=${PWD}:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=${GPU_ID} python code/resnet18.py  --batch-size 512    --epochs 30 --backbone resnet18 --save-model --data_root ${DATA_DIR}${DATASET} --save_dir=${WEIGHTS}

# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/train.py   --batch-size 50 --test-batch-size 5 --epochs 10 --backbone resnet18 --save-model --data_root ${DATA_DIR}${DATASET} --save_dir=${WEIGHTS}
# mv ./build/float ./build/float_1.5Mpix
# mkdir ./build/float
