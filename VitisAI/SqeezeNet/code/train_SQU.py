#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# ----------------------------
# Image size (AIDER images resized)
# ----------------------------
IMG_W = 224
IMG_H = 224

# ----------------------------
# SqueezeNet model with custom head
# ----------------------------
def squeezenet_aider(num_classes=5, pretrained=True):
    model = models.squeezenet1_1(pretrained=pretrained)
    # Replace the classifier (original has Dropout + Conv2d + ReLU + AvgPool)
    model.classifier = nn.Sequential(
       # nn.Dropout(p=0.4),
       nn.Conv2d(512, 256, kernel_size=1),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.2),
       nn.Conv2d(256, num_classes, kernel_size=1),
       nn.AdaptiveAvgPool2d((1, 1))
    )

    return model


# ----------------------------
# Train & Test functions
# ----------------------------
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")
    return acc


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='PyTorch SqueezeNet on AIDER')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data_root', type=str, default='./AIDER', help='root dir of AIDER dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save model path')
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    print(f"Training on {device} device.")

    # ----------------------------
    # Data transforms for AIDER
    # ----------------------------
    train_transform = transforms.Compose([
        transforms.Resize([IMG_H, IMG_W]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize([IMG_H, IMG_W]),
        transforms.ToTensor(),
    ])

    # ----------------------------
    # Load AIDER dataset
    # ----------------------------
    train_set = datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=train_transform)
    val_set   = datasets.ImageFolder(os.path.join(args.data_root, 'val'), transform=val_transform)

    num_classes = len(train_set.classes)
    print(f'# classes: {num_classes}')
    print(f'train num: {len(train_set)}')
    print(f'classes: {train_set.classes}')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    # ----------------------------
    # Model setup
    # ----------------------------
    model = squeezenet_aider(num_classes=num_classes).to(device)

    if args.resume:
        print(f"Resuming from checkpoint {args.resume}")
        model.load_state_dict(torch.load(args.resume))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)

    # ----------------------------
    # Training loop
    # ----------------------------
    best = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        cur_acc = test(model, device, test_loader)
        scheduler.step()

        if cur_acc > best and args.save_model:
            print(f"New best acc: {cur_acc:.2f}, saving model...")
            torch.save(model.state_dict(), os.path.join(args.save_dir,"AIDER_squeezenet_best.pth"))
            best = cur_acc

    # Final save
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.save_dir, "AIDER_squeezenet_last.pth"))


if __name__ == "__main__":
    main()

