#!/bin/sh

# Copyright © 2025 Omnia Abdelkariem 
# Egypt Japan University for Science and Technology 

DATA_DIR=./build/data
WEIGHTS=./build/float
DATASET=AIDER
GPU_ID=0

export PYTHONPATH=${PWD}:${PYTHONPATH}


echo " "
echo "Conducting testing with floating point CNN"
echo " "
# float test
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py --backbone resnet18 --resume ${WEIGHTS}/AIDER_resnet18_best.pt --data_root ${DATA_DIR}/${DATASET}

