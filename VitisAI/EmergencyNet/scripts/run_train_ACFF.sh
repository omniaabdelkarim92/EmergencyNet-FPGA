#!/bin/sh

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

# use the main.py to train EmergencyNet  on Vitis AI Platform to generate 32fp weights 
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/main.py 

# mkdir ./build/float

