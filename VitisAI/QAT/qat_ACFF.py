#!/usr/bin/env python3
"""
QAT template for Vitis-AI (pytorch_nndct)

Usage example (inside Vitis-AI docker):
    python qat_vitisai.py \
      --model_class SqeezeNet1_0 \
      --model_file code/my_models_v1.py \
      --fp32_ckpt ./build/float/SqeezeNet1_0_VAI_SQU_best.pth \
      --data_root ./build/data/AIDER \
      --output_dir ./build/quantized \
      --batch_size 32 \
      --calib_steps 500 \
      --qat_epochs 8
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Vitis AI quantizer API
from pytorch_nndct.apis import torch_quantizer  # provided in Vitis-AI conda env

data_root = '/workspace/tutorials/Tutorials/PyTorch_AIDER-main/files/code/dataset/AIDER/'

# =========================
# Replace/import your model here (or pass module path)
# =========================
# Example: 
from emernet_v1 import ACFFModel
#from ShuffleNetV2 import ShuffleNet
#from train_efficientnet_aider import EffB0
'''
# The script also supports dynamic import of a model file if you prefer to pass file/class.
def import_model_class(module_path, class_name):
    # module_path: path like "code.emernet_v1" (python importable)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)
'''

# =========================
# Placeholder: user should replace with real dataloader implementation
# =========================
def build_train_val_loaders(data_root, batch_size, img_size=(224,224), num_workers=4):
    """
    Return: train_loader, val_loader, calib_loader
    - calib_loader should yield representative images for calibration (no labels required)
    Replace this stub with your actual dataset / transforms.
    """
    from torchvision import transforms, datasets
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # Match the preprocessing you used for FP32 training:
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=transform)
    calib_dataset = datasets.ImageFolder(os.path.join(data_root, "calib"), transform=transform) \
                    if os.path.exists(os.path.join(data_root, "calib")) \
                    else val_dataset  # fallback to val if no calib folder

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, calib_loader






# =========================
#   # import your model class directly
# =========================
from my_models_v1 import *
from emernet_v1 import ACFFModel
#from ShuffleNetV2 import ShuffleNet
#from train_efficientnet_aider import EffB0
# Create model instance
""" Implemented models [ MobileNet_v2, MobileNet_v3, SqeezeNet1_0, VGG16, ShuffleNet_v2 <--
    EfficientNet_B0  ResNet50
"""
model_name = "SqeezeNet1_0"
print(" my Model is ",model_name)

#model = ACFFModel(num_classes=5) 
#model = MobileNet_v2()
#model = ShuffleNet(5)
#model = EffB0(num_classes=5)

# =========================
# Utility: evaluate
# =========================
def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    loss_meter = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_meter += float(loss.item()) * imgs.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)
    acc = 100.0 * correct / (total + 1e-12)
    avg_loss = loss_meter / (total + 1e-12)
    return avg_loss, acc

#============================
#fp32_ckpt = "/files/build/float/emergencyNetv12_best.pt"
#fp32_ckpt = "/files/build/float/MobileNet_v2_VAI_MBN_best.pth"
fp32_ckpt = "/files/build/float/SqeezeNet1_0_VAI_SQU_best.pth"
#fp32_ckpt = "/files/build/float/shufflenet_aider.pt"
#fp32_ckpt = "/files/build/float/efficientnet_b0_aider.pt"
# =========================
# QAT flow
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # import model class
    #model_cls = import_model_class(args.model_file.replace('/', '.').rstrip('.py'), args.model_class)
    #model = model_cls(num_classes=args.num_classes)  # adjust if constructor differs
    # import model class
    #model_cls = import_model_class(args.model_file.replace('/', '.').rstrip('.py'), args.model_class)
   # model = model_cls(num_classes=args.num_classes)  # adjust if constructor differs
    model = select_model(args.model_class, classes=5)

    # load fp32 checkpoint
    if args.fp32_ckpt:
        ckpt = torch.load(args.fp32_ckpt, map_location='cpu')
        # allow checkpoints that contain dict wrapper: try common keys
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            try:
                model.load_state_dict(ckpt)
            except Exception as e:
                print("Warning: load_state_dict failed directly, trying loose matching...", e)
                # try filtering keys
                state = ckpt.get('state_dict', ckpt)
                new_state = {}
                for k, v in state.items():
                    key = k.replace('module.', '')  # remove DataParallel prefix if present
                    new_state[key] = v
                model.load_state_dict(new_state, strict=False)
        print("FP32 checkpoint loaded.")
    model.to(device)

    # build dataloaders
    train_loader, val_loader, calib_loader = build_train_val_loaders(args.data_root, args.batch_size)

    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)

    ############
    # 1) Calibration (collect stats)
    ############
    print("==> Calibration (PTQ) start")
    quantizer = torch_quantizer(quant_mode='calib', module=model, input_args=(dummy_input,), device=device, output_dir=args.output_dir)
    quant_model = quantizer.quant_model  # this is wrapped quant model (for calibration)
    quant_model.eval()

    # run a subset or full calib dataset
    n_calib = 0
    with torch.no_grad():
        for imgs, _ in tqdm(calib_loader, total=min(len(calib_loader), args.calib_steps)):
            imgs = imgs.to(device)
            _ = quant_model(imgs)
            n_calib += imgs.size(0)
            if n_calib >= args.calib_images:
                break
    print(f"Calibration finished on {n_calib} images.")

    # optional: export quant config (PTQ)
    quantizer.export_quant_config()

    ############
    # 2) Create QAT model (train-aware)
    ############
    print("==> Creating QAT model")
    # create quantizer in 'train' or 'qat' mode depending on Vitis-AI version
    # Vitis-AI older/newer may use quant_mode='test' or 'train' for QAT — try 'train' first and fallback to 'test' if not supported
    try:
        qat_quantizer = torch_quantizer(quant_mode='train', module=model, input_args=(dummy_input,), device=device, output_dir=args.output_dir)
    except Exception:
        qat_quantizer = torch_quantizer(quant_mode='test', module=model, input_args=(dummy_input,), device=device, output_dir=args.output_dir)

    model_qat = qat_quantizer.quant_model
    model_qat.train()
    model_qat.to(device)

    # optimizer, scheduler
    optimizer = optim.SGD(model_qat.parameters(), lr=args.qat_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.qat_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_qat_acc = 0.0
    for epoch in range(args.qat_epochs):
        model_qat.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{args.qat_epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model_qat(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * imgs.size(0)
            pbar.set_postfix(loss=running_loss/((pbar.n+1)*imgs.size(0)))

        scheduler.step()

        # validate quantized model
        val_loss, val_acc = evaluate(model_qat, val_loader, device)
        print(f"[QAT] Epoch {epoch+1} val loss: {val_loss:.4f}  acc: {val_acc:.3f}%")

        # save best
        if val_acc > best_qat_acc:
            best_qat_acc = val_acc
            save_path = os.path.join(args.output_dir, f"best_qat_{args.model_class}.pth")
            torch.save({'state_dict': model_qat.state_dict(), 'acc': best_qat_acc}, save_path)
            print("Saved best QAT model:", save_path)

    print("QAT finished. Best val acc: ", best_qat_acc)

    ############
    # 3) Export quantized artifacts for deploy
    ############
    print("==> Exporting quantized model artifacts")
    try:
        qat_quantizer.export_quant_config()
    except Exception:
        pass

    # Export to xmodel/onnx if supported:
    try:
        qat_quantizer.export_xmodel(output_dir=args.output_dir)
        print("xmodel exported to:", args.output_dir)
    except Exception as e:
        print("export_xmodel failed (maybe old/new API), trying export onnx/pt:", e)
        # fallback: save quantized .pt (int simulation)
        torch.save(model_qat.state_dict(), os.path.join(args.output_dir, "model_qat_state_dict.pth"))
        # attempt onnx export
        sample = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        torch.onnx.export(model_qat, sample, os.path.join(args.output_dir, "model_qat.onnx"), opset_version=11)
        print("Saved fallback pt/onnx in", args.output_dir)

    print("Done. Output dir:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True,
                        help="Python import path to your model file (e.g. code.emernet_v1)")
    parser.add_argument("--model_class", type=str, default ="emernet_v1",
                        help="Model class name to instantiate inside model_file (e.g. ACFFModel)")
    parser.add_argument("--fp32_ckpt", type=str, default=None, help="Path to FP32 checkpoint to load")
    parser.add_argument("--data_root", type=str, required=True, help="Root of dataset containing train/val folders")
    parser.add_argument("--output_dir", type=str, default="./build/quantized", help="Where to store quant artifacts")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--calib_images", type=int, default=800, help="How many images to use for calibration")
    parser.add_argument("--calib_steps", type=int, default=500, help="Number of batches for calibration fallback")
    parser.add_argument("--qat_epochs", type=int, default=8)
    parser.add_argument("--qat_lr", type=float, default=1e-4)
    parser.add_argument("--qat_step", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    main(args)

