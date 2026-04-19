#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train + PTQ / QAT flow for ResNet on AIDER using Vitis-AI (pytorch_nndct).

Usage examples (inside Vitis-AI env):
# 1) Train / finetune FP32 (regular)
python train_and_quant_resnet.py --data_root ./AIDER --backbone resnet18 --epochs 20 --batch-size 64 --save-model --save_dir ./build/float

# 2) Post-training calibration (PTQ)
python train_and_quant_resnet.py --data_root ./AIDER --backbone resnet18 \
    --quant_mode calib --fp32_ckpt ./build/float/AIDER_resnet18_last.pth --quant_dir ./build/quantized --calib_images 800

# 3) QAT (fine-tune quantized model)
python train_and_quant_resnet.py --data_root ./AIDER --backbone resnet18 \
    --quant_mode qat --fp32_ckpt ./build/float/AIDER_resnet18_last.pth --quant_dir ./build/quantized \
    --qat_epochs 8 --qat_lr 1e-4
 ResNEt_Accu: (96.58%)
"""


import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from my_models_v1 import *

# ----------------------------
# Image size and normalization
# ----------------------------
IMG_W = 224
IMG_H = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----------------------------
# Modified ResNet heads (keeps your custom head)
# ----------------------------
def resnet18_aider(num_classes=5, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

def resnet34_aider(num_classes=5, pretrained=True):
    model = models.resnet34(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

def resnet50_aider(num_classes=5, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model
def MobileNet_v2(num_classes=5, pretrained=True):
        try:
            # Newer API
            model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        except TypeError:
            # Older API
            model = models.mobilenet_v2(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        return model

def MobileNet_v3(num_classes=5, pretrained=True):
        try:
            model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        except TypeError:
            model = models.mobilenet_v3_small(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )
        return model
# ----------------------------
# Train / Test utilities
# ----------------------------
def train_one_epoch(model, device, train_loader, optimizer, epoch, log_interval=10, dry_run=False):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * data.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += data.size(0)

        if batch_idx % log_interval == 0:
            pbar.set_postfix(loss=running_loss / max(1, total), acc=100.0 * correct / max(1, total))
            if dry_run:
                break

    return running_loss / max(1, total), 100.0 * correct / max(1, total)

def evaluate(model, device, loader, deploy=False):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            loss_sum += float(loss.item()) * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            if deploy:
                break
    avg_loss = loss_sum / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc

# ----------------------------
# Safe checkpoint loading
# ----------------------------
def safe_load(model, ckpt_path, map_location='cpu'):
    ck = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ck, dict):
        sd = ck.get('state_dict', ck.get('model', ck))
    else:
        sd = ck
    # strip 'module.' prefix if present
    if isinstance(sd, dict):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train + PTQ/QAT for ResNet on AIDER (Vitis-AI)')
    # training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='./build/float')
    parser.add_argument('--resume', type=str, default='', help='resume training from checkpoint')

    # model / data
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'MobileNet_v2' ,'MobileNet_v3'])
    parser.add_argument('--data_root', type=str, required=True, help='AIDER root (contains train/ val/ calib/ optional)')
    parser.add_argument('--num-workers', type=int, default=4)

    # quantization
    parser.add_argument('--quant_mode', type=str, default='float',
                        choices=['float', 'calib', 'qat', 'deploy'],
                        help='float | calib (PTQ) | qat (QAT fine-tune) | deploy (export)')
    parser.add_argument('--quant_dir', type=str, default='./build/quantized')
    parser.add_argument('--fp32_ckpt', type=str, default='', help='path to FP32 checkpoint (recommended for QAT)')
    parser.add_argument('--calib_images', type=int, default=800)
    parser.add_argument('--calib_steps', type=int, default=500)
    parser.add_argument('--qat_epochs', type=int, default=8)
    parser.add_argument('--qat_lr', type=float, default=1e-4)
    parser.add_argument('--qat_step', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print("[INFO] device:", device)

    # data transforms (include ImageNet normalization)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # dataloaders
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    calib_dir = os.path.join(args.data_root, 'calib')  # optional folder for calibration images

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError("Make sure data_root contains 'train' and 'val' subfolders")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    calib_ds = datasets.ImageFolder(calib_dir, transform=val_tf) if os.path.isdir(calib_dir) else val_ds

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(train_ds.classes)
    print(f"[INFO] classes ({num_classes}): {train_ds.classes}")
    print(f"[INFO] train samples: {len(train_ds)}, val samples: {len(val_ds)}, calib samples: {len(calib_ds)}")

    # build model
    if args.backbone == 'resnet18':
        model = resnet18_aider(num_classes=num_classes, pretrained=True)
    elif args.backbone == 'resnet34':
        model = resnet34_aider(num_classes=num_classes, pretrained=True)
    elif args.backbone == 'resnet50':
        model = resnet50_aider(num_classes=num_classes, pretrained=True)
    elif args.backbone == 'MobileNet_v2':
        model = MobileNet_v2(num_classes=num_classes, pretrained=True)
    else:
        model = MobileNet_v3(num_classes=num_classes, pretrained=True)

    model = model.to(device)

    # optionally load FP32 checkpoint (helps QAT converge)
    if args.fp32_ckpt:
        print("[INFO] Loading FP32 checkpoint:", args.fp32_ckpt)
        safe_load(model, args.fp32_ckpt, map_location=device)
        print("[INFO] FP32 checkpoint loaded (partial allowed).")

    # standard training flow (if user wants to train/finetune FP32)
    if args.quant_mode == 'float':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
        best = 0.0
        os.makedirs(args.save_dir, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, device, train_loader, optimizer, epoch, log_interval=args.log_interval)
            val_loss, val_acc = evaluate(model, device, val_loader)
            scheduler.step()
            print(f"[FP32] Epoch {epoch} val_acc: {val_acc:.2f}%")
            if args.save_model and val_acc > best:
                best = val_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"AIDER_{args.backbone}_best.pth"))
                print("[INFO] saved best fp32 checkpoint.")

        if args.save_model:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"AIDER_{args.backbone}_last.pth"))
        return

    # From here: quantization modes (calib / qat / deploy)
    # import quantizer
    from pytorch_nndct.apis import torch_quantizer

    dummy_input = torch.randn([args.batch_size, 3, IMG_H, IMG_W], dtype=torch.float32).to(device)

    # 1) Calibration (PTQ) - collect activation stats
    if args.quant_mode == 'calib':
        print("[INFO] Running calibration (PTQ) ...")
        os.makedirs(args.quant_dir, exist_ok=True)
        quantizer = torch_quantizer(quant_mode='calib', module=model, input_args=(dummy_input,), device=device, output_dir=args.quant_dir)
        qmodel = quantizer.quant_model
        qmodel.eval()
        seen = 0
        with torch.no_grad():
            for imgs, _ in tqdm(calib_loader, total=min(len(calib_loader), args.calib_steps)):
                imgs = imgs.to(device)
                _ = qmodel(imgs)
                seen += imgs.size(0)
                if seen >= args.calib_images:
                    break
        print(f"[INFO] Calibration finished on {seen} images")
        try:
            quantizer.export_quant_config()
            print("[INFO] exported quant config to", args.quant_dir)
        except Exception as e:
            print("[WARN] export_quant_config failed:", e)
        return

    # 2) QAT flow
    if args.quant_mode == 'qat':
        print("[INFO] Starting QAT flow (calibration + train-aware quant)")
        os.makedirs(args.quant_dir, exist_ok=True)
        # calibration first to collect ranges
        calib_quantizer = torch_quantizer(quant_mode='calib', module=model, input_args=(dummy_input,), device=device, output_dir=args.quant_dir)
        calib_qmodel = calib_quantizer.quant_model
        calib_qmodel.eval()
        seen = 0
        with torch.no_grad():
            for imgs, _ in tqdm(calib_loader, total=min(len(calib_loader), args.calib_steps)):
                imgs = imgs.to(device)
                _ = calib_qmodel(imgs)
                seen += imgs.size(0)
                if seen >= args.calib_images:
                    break
        print(f"[INFO] Calibration done (seen {seen} images)")

        # create QAT model (train-aware). Some Vitis-AI versions use 'train' and others 'test'
        try:
            qat_quantizer = torch_quantizer(quant_mode='train', module=model, input_args=(dummy_input,), device=device, output_dir=args.quant_dir)
        except Exception:
            qat_quantizer = torch_quantizer(quant_mode='test', module=model, input_args=(dummy_input,), device=device, output_dir=args.quant_dir)

        qmodel = qat_quantizer.quant_model.to(device)
        qmodel.train()

        # optimizer for QAT fine-tuning
        optimizer = optim.SGD(qmodel.parameters(), lr=args.qat_lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.qat_step, gamma=0.1)
        best_acc = 0.0

        # QAT fine-tune loop
        for epoch in range(1, args.qat_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(qmodel, device, train_loader, optimizer, epoch, log_interval=args.log_interval)
            val_loss, val_acc = evaluate(qmodel, device, val_loader)
            scheduler.step()
            print(f"[QAT] Epoch {epoch} train_acc: {tr_acc:.2f}% val_acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(args.quant_dir, f"best_qat_{args.backbone}.pth")
                torch.save({'state_dict': qmodel.state_dict(), 'acc': best_acc}, save_path)
                print("[INFO] Saved best QAT state_dict to", save_path)
                # try to save a scripted version (optional)
                try:
                    scripted = torch.jit.script(qmodel.cpu().eval())
                    scripted.save(os.path.join(args.quant_dir, f"best_qat_{args.backbone}.pt"))
                    qmodel.to(device).train()
                except Exception as e:
                    print("[WARN] TorchScript save failed:", e)

        print("[INFO] QAT finished. Best val acc: {:.2f}%".format(best_acc))

        # export xmodel / fallback
        try:
            qat_quantizer.export_xmodel(output_dir=args.quant_dir)
            print("[INFO] export_xmodel succeeded to", args.quant_dir)
        except Exception as e:
            print("[WARN] export_xmodel failed:", e)
            torch.save(qmodel.state_dict(), os.path.join(args.quant_dir, "model_qat_state_dict.pth"))
            try:
                sample = torch.randn(1, 3, IMG_H, IMG_W).to(device)
                torch.onnx.export(qmodel, sample, os.path.join(args.quant_dir, "model_qat.onnx"), opset_version=11)
                print("[INFO] Fallback onnx exported")
            except Exception as ex:
                print("[WARN] ONNX export failed:", ex)
        return

    # 3) Deploy mode: quick calibration + export (no QAT)
    if args.quant_mode == 'deploy':
        print("[INFO] Deploy mode: quick calib + export")
        os.makedirs(args.quant_dir, exist_ok=True)
        quantizer = torch_quantizer(quant_mode='calib', module=model, input_args=(dummy_input,), device=device, output_dir=args.quant_dir)
        qmodel = quantizer.quant_model.eval()
        with torch.no_grad():
            for imgs, _ in tqdm(calib_loader, total=min(len(calib_loader), args.calib_steps)):
                imgs = imgs.to(device)
                _ = qmodel(imgs)
                break
        try:
            quantizer.export_quant_config()
            quantizer.export_xmodel(args.quant_dir, deploy_check=True)
            quantizer.export_torch_script(output_dir=args.quant_dir)
            quantizer.export_onnx_model(output_dir=args.quant_dir)
            print("[INFO] Deploy artifacts exported to", args.quant_dir)
        except Exception as e:
            print("[WARN] Deploy export failed:", e)
        return

if __name__ == '__main__':
    main()

