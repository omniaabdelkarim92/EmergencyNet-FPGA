#!/bin/sh

# Copyright © 2025 Omnia Abdelkariem 
# Egypt Japan University for Science and Technology 

echo "Activate environment..."


DATA_DIR=./build/data/
WEIGHTS=./build/float
DATASET=AIDER
GPU_ID=0
#QUANT_DIR=${QUANT_DIR:-quantized}
QUANT_DIR=./build/quantized
export PYTHONPATH=${PWD}:${PYTHONPATH}


echo "Conducting Quantization"
# fix calib
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test_E.py --backbone emernet --resume ${WEIGHTS}/emergencyNetv12_best.pt --data_root ${DATA_DIR}/${DATASET} --quant_mode calib --quant_dir=${QUANT_DIR}

# fix test
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test_E.py --backbone emernet --resume ${WEIGHTS}/emergencyNetv12_best.pt --data_root ${DATA_DIR}/${DATASET} --quant_mode test  --quant_dir=${QUANT_DIR}

# deploy
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test_E.py --backbone emernet --resume ${WEIGHTS}/emergencyNetv12_best.pt --data_root ${DATA_DIR}/${DATASET} --quant_mode test  --quant_dir=${QUANT_DIR} --deploy --device=cpu
