#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch SqueezeNet Test Script with Vitis-AI Quantization/Deployment Support.

Expects your training script to provide `squeezenet_aider(num_classes, pretrained=True/False)`.
This script:
 - builds the same SqueezeNet architecture used for training
 - loads a checkpoint (handles simple dict wrappers)
 - optionally runs Vitis-AI quantizer in calib/test modes
 - runs evaluation and can export artifacts for deploy
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

# Import the model builder used in training
# Ensure train_SQU.py defines: def squeezenet_aider(num_classes, pretrained=True)
from train_SQU import squeezenet_aider

# Image size (ImageNet default)
IMG_W = np.short(224)
IMG_H = np.short(224)


def safe_load_checkpoint_into_model(model, ckpt_path, map_location="cpu"):
    """
    Load checkpoint into model. Handles:
      - raw state_dict saved with torch.save(model.state_dict())
      - dict wrappers like {'state_dict': ..., 'acc': ...}
    Returns True if loaded successfully.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ck, dict):
        # common wrappers
        if "state_dict" in ck:
            sd = ck["state_dict"]
        elif "model" in ck:
            sd = ck["model"]
        else:
            sd = ck
        # strip DataParallel prefix if present
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        try:
            model.load_state_dict(sd, strict=True)
            return True
        except Exception as e:
            # try non-strict fallback (best-effort)
            try:
                model.load_state_dict(sd, strict=False)
                print("[WARN] Loaded checkpoint with strict=False (partial match).")
                return True
            except Exception as e2:
                print("[ERROR] Failed to load checkpoint into model:", e2)
                return False
    else:
        # maybe the checkpoint itself is a state_dict-like object
        try:
            model.load_state_dict(ck, strict=True)
            return True
        except Exception:
            try:
                model.load_state_dict(ck, strict=False)
                print("[WARN] Loaded checkpoint object with strict=False.")
                return True
            except Exception as e:
                print("[ERROR] Unknown checkpoint format and failed to load:", e)
                return False


def test(model, device, test_loader, deploy=False):
    """Run evaluation"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            # squeeze conv output if necessary (squeezenet returns N x C x 1 x 1)
            if out.ndim == 4:
                out = out.view(out.size(0), -1)
            test_loss += criterion(out, target).item() * out.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            if deploy:
                # for deploy checks we only need to verify one forward pass
                break

    if total > 0:
        test_loss /= total
        acc = 100.0 * correct / total
    else:
        test_loss, acc = 0.0, 0.0

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({acc:.3f}%)\n")
    return test_loss, acc


def main():
    parser = argparse.ArgumentParser(description="PyTorch SqueezeNet Example (test / quant)")
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", type=str, default="", help="path to trained model checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, default="./build/data", help="dataset root (contains val/)")
    parser.add_argument("--quant_dir", default="./build/quantized")
    parser.add_argument("--quant_mode", default="float", type=str, choices=["float", "calib", "test"],
                        help="float | calib | test (train-aware QAT uses 'train' but script uses calib/test modes)")
    parser.add_argument("--device", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("--deploy", action="store_true", help="export deploy artifacts (xmodel/pt/onnx)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Testing on {device} device.")

    # Data loader
    if args.deploy:
        args.test_batch_size = 1

    val_tf = transforms.Compose([transforms.Resize([IMG_H, IMG_W]), transforms.ToTensor()])
    val_dir = os.path.join(args.data_root, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val/ folder not found in data_root: {args.data_root}")

    test_set = datasets.ImageFolder(val_dir, transform=val_tf)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,
                                              shuffle=False, drop_last=False, num_workers=4)

    print("Test samples:", len(test_set))
    print("Classes:", test_set.classes)

    # Build model (must match training definition)
    num_classes = len(test_set.classes)
    print(f"Building SqueezeNet with {num_classes} classes...")
    model = squeezenet_aider(num_classes=num_classes, pretrained=False)
    model = model.to(device)

    # Load checkpoint if provided
    if args.resume:
        print("Loading checkpoint:", args.resume)
        ok = safe_load_checkpoint_into_model(model, args.resume, map_location=device)
        if not ok:
            raise RuntimeError("Failed to load checkpoint. Aborting.")
        else:
            print("Checkpoint loaded successfully.")

    # Quantization modes (Vitis-AI pytorch_nndct)
    if args.quant_mode != "float":
        from pytorch_nndct.apis import torch_quantizer
        dummy_input = torch.randn([1, 3, IMG_H, IMG_W], dtype=torch.float32).to(device)

        # create quantizer in requested mode
        print(f"Creating quantizer in mode '{args.quant_mode}' (output_dir={args.quant_dir})")
        os.makedirs(args.quant_dir, exist_ok=True)
        quantizer = torch_quantizer(args.quant_mode, model, (dummy_input,), output_dir=args.quant_dir, device=device)
        model = quantizer.quant_model  # wrapped model for calib/test

    # Run test (deploy flag causes single-batch forward-check)
    test(model, device, test_loader, deploy=args.deploy)

    # If calibration, export quant config
    if args.quant_mode == "calib":
        print("Exporting quant config...")
        try:
            quantizer.export_quant_config()
            print("Quant config exported to", args.quant_dir)
        except Exception as e:
            print("export_quant_config failed:", e)

    # If deploy requested, try to export xmodel / scripted / onnx
    if args.deploy:
        print("Exporting deploy artifacts...")
        try:
            quantizer.export_xmodel(args.quant_dir, deploy_check=True)
        except Exception as e:
            print("export_xmodel failed:", e)
        try:
            quantizer.export_torch_script(output_dir=args.quant_dir)
        except Exception as e:
            print("export_torch_script failed:", e)
        try:
            quantizer.export_onnx_model(output_dir=args.quant_dir)
        except Exception as e:
            print("export_onnx_model failed:", e)

    print("Done.")


if __name__ == "__main__":
    main()

