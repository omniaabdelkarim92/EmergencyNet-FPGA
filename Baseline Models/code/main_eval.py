from sklearn.metrics import confusion_matrix, classification_report
import random
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from my_models import *
from my_utils.dataset import load_dataset
from my_utils.helper_fns import print_size_of_model, print_no_of_parameter
import numpy as np
import torch
from torchvision import transforms
#from emergencyNet import ACFFModel
#from emergencyNet2 import ACFFModel


""" Implemented models [ MobileNet_v2, MobileNet_v3, SqeezeNet1_0, VGG16, ShuffleNet_v2 <--
    EfficientNet_B0  ResNet50
"""
model_name = "SqeezeNet1_0"
print(" my Model is ",model_name)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def eval(model, dataloader, ):
    y_pred = []
    y_true = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # iterate over test data
    model.eval()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ('collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    conf_mtrx_filepath = '../results/' + model_name + 'confusionMatrix.png'
    plt.savefig(conf_mtrx_filepath)
    plt.clf()
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))


def main():
    INPUT_SIZE = 224
    data_dir = '../files/code/dataset/AIDER/'   # change this to your actuall dataset path 

    # LOAD DATA
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create dataloaders "train" and "val"
    train_loader, val_loader, my_datasets = load_dataset(data_dir, data_transforms)

 
    model = select_model(model_name, classes=5)
    saved_state_dict = torch.load('../results/SqeezeNet1_0_GPU_75_best.pth') 
    model.load_state_dict(saved_state_dict['state_dict'])

    print(model)
    # save entire model
    # torch.save(model, '../results/emergencyNet.pth')

    # Export / Load Model in TorchScript Format
    # model_scripted = torch.jit.script(model)  # Export to TorchScript
    # model_scripted.save('model_scripted.pt')  # Save

    print_size_of_model(model)
    print_no_of_parameter(model)

    # Evaluate the Model
    # eval(model, val_loader)


if __name__ == '__main__':
    main()
