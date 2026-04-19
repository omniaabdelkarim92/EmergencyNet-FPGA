#!/usr/bin/env python3
"""
Robust QAT script for Vitis-AI (pytorch_nndct).

Usage example (inside Vitis-AI docker):
python qat_vitisai_full.py \
  --model_file code/RESNet.py \
  --model_class res_aider \
  --fp32_ckpt ../build/float/resnet18_aider_best.pth \
  --data_root ../build/data/AIDER \
  --output_dir ../build/quantized \
  --batch_size 32 \
  --calib_steps 500 \
  --qat_epochs 8
"""
import os
import argparse
import importlib.util
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Vitis-AI quantizer API (available in Vitis-AI conda env)
from pytorch_nndct.apis import torch_quantizer

# -------------------------
# Helpers
# -------------------------
def dynamic_import_model(model_file, model_class, num_classes):
    """
    Import model class from a python file path (non-package).
    Expects the class constructor to accept num_classes (if it does).
    """
    spec = importlib.util.spec_from_file_location("user_model_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, model_class)
    # try to instantiate with num_classes if signature expects it, else no args
    try:
        model = cls(num_classes=num_classes)
    except TypeError:
        try:
            model = cls(num_classes)
        except TypeError:
            model = cls()
    return model

def load_checkpoint_into_model(model, ckpt_path, device):
    """
    Loads either .pth (state_dict) or .pt (TorchScript) into the provided model.
    If .pt (TorchScript) is provided, try to extract state_dict() and load
    so the model (nn.Module) can be used for QAT.
    Returns True if successful and the model is regular nn.Module ready for QAT.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # If it's a TorchScript file (.pt), try torch.jit.load and extract state_dict
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth.pt"):
        print("[INFO] Detected a TorchScript file (.pt). Attempting to extract state_dict()...")
        try:
            ts = torch.jit.load(ckpt_path, map_location=device)
            # Some TorchScript objects may expose state_dict()
            if hasattr(ts, "state_dict"):
                state = ts.state_dict()
                # clean keys (module. prefix)
                new_state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(new_state, strict=False)
                print("[INFO] Extracted state_dict from TorchScript and loaded into model.")
                return True
            else:
                print("[WARN] TorchScript object has no state_dict(). Cannot extract weights for QAT.")
                return False
        except Exception as e:
            print("[WARN] Failed to load/extract TorchScript:", e)
            return False

    # Else treat as .pth (or generic) checkpoint
    print(f"[INFO] Loading checkpoint (state_dict) from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # If ckpt is a dict wrapper, pull out common keys
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        # Strip 'module.' prefix from DataParallel if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=False)
            print("[INFO] Loaded checkpoint state_dict into model (strict=False).")
            return True
        except Exception as e:
            print("[WARN] load_state_dict strict load failed:", e)
            # Attempt best-effort load by matching keys
            model_state = model.state_dict()
            filtered = {}
            for k, v in state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered[k] = v
            model_state.update(filtered)
            model.load_state_dict(model_state)
            print("[INFO] Partial weights loaded (matching shapes).")
            return True
    else:
        # ckpt might be a ScriptModule
        if hasattr(ckpt, "state_dict"):
            print("[INFO] Loaded object from torch.load has state_dict(); extracting.")
            state = ckpt.state_dict()
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            return True
        else:
            print("[ERROR] Unknown checkpoint format; cannot load into model.")
            return False

def build_loaders(data_root, batch_size, img_size, num_workers, use_calib_folder=False):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tf)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf)
    if use_calib_folder and os.path.exists(os.path.join(data_root, "calib")):
        calib_ds = datasets.ImageFolder(os.path.join(data_root, "calib"), transform=tf)
    else:
        calib_ds = val_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, calib_loader

def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_sum += float(loss.item()) * imgs.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)
    acc = 100.0 * correct / (total + 1e-12)
    avg_loss = loss_sum / (total + 1e-12)
    return avg_loss, acc

# -------------------------
# Main QAT flow
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Dynamically import model class
    model = dynamic_import_model(args.model_file, args.model_class, args.num_classes)
    model = model.to(device)

    # Try loading provided checkpoint (pth or pt). If returns False and ckpt exists, abort
    if args.fp32_ckpt:
        ok = load_checkpoint_into_model(model, args.fp32_ckpt, device)
        if not ok:
            print("[ERROR] Failed to prepare model from checkpoint for QAT. Exiting.")
            return
        else:
            print("[INFO] FP32 checkpoint loaded into model.")

    # Build dataloaders
    train_loader, val_loader, calib_loader = build_loaders(args.data_root, args.batch_size, args.img_size, args.num_workers, use_calib_folder=True)

    # Prepare dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)

    # 1) Calibration (PTQ) - collect statistics
    print("==> Calibration (PTQ) start")
    quantizer = torch_quantizer(quant_mode='calib', module=model, input_args=(dummy_input,), device=device, output_dir=args.output_dir)
    quant_model = quantizer.quant_model  # wrapped model for calibration
    quant_model.eval()

    n_calib = 0
    with torch.no_grad():
        for imgs, _ in tqdm(calib_loader, total=min(len(calib_loader), args.calib_steps)):
            imgs = imgs.to(device)
            _ = quant_model(imgs)
            n_calib += imgs.size(0)
            if n_calib >= args.calib_images:
                break
    print(f"Calibration finished on {n_calib} images.")

    # Export quantization config (optional)
    try:
        quantizer.export_quant_config()
    except Exception:
        pass

    # 2) Create QAT model (train-aware)
    print("==> Creating QAT model for fine-tuning (QAT)")
    # try 'train' mode first, fallback to 'test' if API differs
    try:
        qat_quantizer = torch_quantizer(quant_mode='train', module=model, input_args=(dummy_input,), device=device, output_dir=args.output_dir)
    except Exception:
        qat_quantizer = torch_quantizer(quant_mode='test', module=model, input_args=(dummy_input,), device=device, output_dir=args.output_dir)

    model_qat = qat_quantizer.quant_model
    model_qat.train()
    model_qat.to(device)

    optimizer = optim.SGD(model_qat.parameters(), lr=args.qat_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.qat_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Training history
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    best_qat_acc = 0.0
    print("==> Starting QAT fine-tuning")
    for epoch in range(1, args.qat_epochs + 1):
        model_qat.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{args.qat_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_qat(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            pbar.set_postfix(loss=running_loss / max(1, total_train))

        scheduler.step()

        train_loss = running_loss / max(1, total_train)
        train_acc = 100.0 * correct_train / max(1, total_train)
        val_loss, val_acc = evaluate(model_qat, val_loader, device)

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"[QAT] Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.3f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.3f}%")

        # Save best QAT model (state_dict)
        if val_acc > best_qat_acc:
            best_qat_acc = val_acc
            save_path = os.path.join(args.output_dir, f"best_qat_{args.model_class}.pth")
            torch.save({'state_dict': model_qat.state_dict(), 'acc': best_qat_acc}, save_path)
            # also save scripted quant-aware model for inspection (optional)
            try:
                scripted = torch.jit.script(model_qat.cpu())
                scripted.save(os.path.join(args.output_dir, f"best_qat_{args.model_class}.pt"))
                model_qat.to(device)
            except Exception:
                # ignore scripting issues
                model_qat.to(device)
            print(f"[INFO] Saved best QAT model: {save_path} (Acc={best_qat_acc:.3f}%)")

    print("QAT finished. Best val acc: ", best_qat_acc)

    # 3) Export quantized artifacts
    print("==> Exporting quantized artifacts (xmodel/onnx/pt fallback)")
    try:
        # primary: try to export xmodel (vai_c_xir expected later)
        qat_quantizer.export_xmodel(output_dir=args.output_dir)
        print("[INFO] xmodel exported to:", args.output_dir)
    except Exception as e:
        print("[WARN] export_xmodel failed, fallback saving state_dict and ONNX:", e)
        torch.save(model_qat.state_dict(), os.path.join(args.output_dir, "model_qat_state_dict.pth"))
        # try ONNX export
        try:
            sample = torch.randn(1, 3, args.img_size, args.img_size).to(device)
            model_qat.eval()
            torch.onnx.export(model_qat, sample, os.path.join(args.output_dir, "model_qat.onnx"), opset_version=11,
                              input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
            print("[INFO] Fallback ONNX exported.")
        except Exception as ex2:
            print("[WARN] ONNX export also failed:", ex2)

    # Save training curves
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot([x/100.0 for x in train_acc_hist], label="Train Acc")
        plt.plot([x/100.0 for x in val_acc_hist], label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid()
        plt.subplot(1,2,2)
        plt.plot(train_loss_hist, label="Train Loss")
        plt.plot(val_loss_hist, label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "qat_training_curves.png"))
        print("[INFO] Training curves saved.")
    except Exception as e:
        print("[WARN] Could not save training curves:", e)

    print("Done. Output dir:", args.output_dir)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to python file containing model class (e.g. code/RESNet.py)")
    parser.add_argument("--model_class", type=str, required=True,
                        help="Name of model class inside the file (e.g. res_aider)")
    parser.add_argument("--fp32_ckpt", type=str, default=None, help="Path to FP32 checkpoint (.pth or .pt)")
    parser.add_argument("--data_root", type=str, required=True, help="Root path of dataset containing train/val[/calib]")
    parser.add_argument("--output_dir", type=str, default="./build/quantized", help="Where to store quant artifacts")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--calib_images", type=int, default=800, help="Number of images to use for calibration")
    parser.add_argument("--calib_steps", type=int, default=500, help="Number of calibration batches to run (fallback)")
    parser.add_argument("--qat_epochs", type=int, default=8)
    parser.add_argument("--qat_lr", type=float, default=1e-4)
    parser.add_argument("--qat_step", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    main(args)

