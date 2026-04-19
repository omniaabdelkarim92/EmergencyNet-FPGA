#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

DATA_DIR=./build/data
WEIGHTS=./build/float
DATASET=vcor
GPU_ID=0

export PYTHONPATH=${PWD}:${PYTHONPATH}


echo " "
echo "Conducting testing with floating point CNN"
echo " "
# float test
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test_E.py --backbone emernet --resume ${WEIGHTS}/emergencyNetv12_best.pt --data_root ${DATA_DIR}/${DATASET}

