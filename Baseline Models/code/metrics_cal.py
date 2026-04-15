import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
from ptflops import get_model_complexity_info  # pip install ptflops
import os
#===============================
# Model Selection 
#===============================
from my_models import *   
# from ResNet18 import res_aider    
# from emergencyNet import ACFFModel  

from emergencyNet2 import ACFFModel
""" Implemented models [ MobileNet_v2, MobileNet_v3, SqeezeNet1_0, VGG16, ShuffleNet_v2 <--
    EfficientNet_B0  ResNet50
"""
model_name = "SqeezeNet1_0"    # you have to specify ehich model you are goning to train , val , test ...
print(" my Model is ",model_name)

model = select_model(model_name, 5)
#===============================
#
#===============================
def model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def model_size(model):
    """Return model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all = (param_size + buffer_size) / 1024 ** 2
    return size_all

def compute_flops(model, input_res=(3, 224, 224)):
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
    # FLOPs = 2 * MACs
    flops = 2 * macs
    return flops

def evaluate_f1(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1 * 100

# Example usage:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#for model_name, model in [('ResNet18', resnet18_model), ('SqueezeNet1_0', squeezenet_model)]:
    #model = model.to(device)
params = model_parameters(model)
size = model_size(model)
flops = compute_flops(model)
f1 = evaluate_f1(model, test_loader, device)
    
print(f"Model: {model_name}")
print(f"Parameters: {params} ({params/1e6:.2f}M)")
print(f"Model Size: {size:.2f} MB")
print(f"F1-score: {f1:.2f}%")
print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
print("="*40)

