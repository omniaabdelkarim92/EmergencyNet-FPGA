#!/bin/sh

# Copyright © 2025 Omnia Abdelkariem 
# Egypt Japan University for Science and Technology 

echo "Activate environment..."


DATA_DIR=./build/data
WEIGHTS=./build/float
DATASET=AIDER
GPU_ID=0
#QUANT_DIR=${QUANT_DIR:-quantized}
QUANT_DIR=./build/quantized
export PYTHONPATH=${PWD}:${PYTHONPATH}

echo "Conducting Quantization"

# 1) Post-training calibration (PTQ)
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py --data_root ./build/data/AIDER --backbone resnet18 \--quant_mode calib --fp32_ckpt ./build/float/AIDER_resnet18_best.pth --quant_dir ./build/quantized --calib_images 800

# 1) Post-training calibration (PTQ)  "Fix test" 
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py --data_root ./build/data/AIDER --backbone resnet18 \--quant_mode test --fp32_ckpt ./build/float/AIDER_resnet18_best.pth --quant_dir ./build/quantized --calib_images 800

# 1) Post-training calibration (PTQ)  " deploy"
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py --data_root ./build/data/AIDER --backbone resnet18 \--quant_mode test --fp32_ckpt ./build/float/AIDER_resnet18_best.pth --quant_dir ./build/quantized --calib_images 800 --deploy --device cpu 


#############################################
## Quanutization Aware Training 
#########################################

# 1) QAT (fine-tune quantized model)
#CUDA_VISIBLE_DEVICES=${GPU_ID}python  ./code/test.py --data_root ./build/data/AIDER --backbone resnet18  --quant_mode qat \
# --fp32_ckpt ./build/float/AIDER_resnet18_best.pth --quant_dir ./build/quantized  --qat_epochs 8 \
#--qat_lr 1e-4


# 2) QAT (fine-tune quantized model)
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./code/test.py --data_root ./build/data/AIDER --backbone resnet18 \
    --quant_mode deploy  --fp32_ckpt ./build/float/AIDER_resnet18_best.pth --quant_dir ./build/quantized --calib_images 800
