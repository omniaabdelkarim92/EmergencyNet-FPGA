import torch
from torch import nn
from torchvision import models
from my_models import *
# from emergencyNet import ACFFModel
from emergencyNet2 import ACFFModel
from my_utils.helper_fns import print_size_of_model, print_no_of_parameter
from thop import profile


"""
saved_model = '../results/model.pth'
model = torch.load(saved_model)
model.eval()
"""
""" Implemented models [ MobileNet_v2, MobileNet_v3, SqeezeNet1_0, VGG16, ShuffleNet_v2 <--
    EfficientNet_B0  ResNet50
"""
model_name = "ResNet18"
print(" my Model is ",model_name)

model = select_model(model_name, 5)


# print(model)
# print_no_of_parameter(model)

# Calculate FLOPs
# Input example (adjust the size and dimension according to your model)
input_size = (1, 3, 224, 224)
# Prepare a dummy input tensor
dummy_input = torch.randn(*input_size)
# Use the `profile` function to count FLOPs
flops, params = profile(model, inputs=(dummy_input,))
print(f"FLOPs: {flops / 1e6:.2f} M")
print(f"Number of parameters: {params / 1e3:.2f} K")
