# EmergencyNet-FPGA

EmergencyNet-FPGA is a computer-vision project for emergency scene classification, with both:

- baseline PyTorch training/evaluation code, and
- FPGA deployment workflows using AMD Vitis AI (quantization + compilation for DPU targets).

The repository includes multiple model families (EmergencyNet/ACFF, ResNet18, SqueezeNet, and others), pretrained checkpoints, and target artifacts for boards such as ZCU104.

## Highlights

- End-to-end training and evaluation pipelines in PyTorch.
- Custom EmergencyNet-style architecture (`ACFFModel`) with atrous + fusion blocks.
- Vitis AI post-training quantization and export (`.xmodel`, TorchScript, ONNX).
- Target-specific compile scripts for ZCU102/ZCU104/VCK190/V70/VEK280/VCK5000.
- Separate experiment folders for EmergencyNet, ResNet18, SqueezeNet, and QAT variants.

## Repository Structure

```text
EmergencyNet-FPGA/
	Baseline Models/
		code/                  # Baseline PyTorch training/eval/model utilities
		results/               # Baseline trained weights

	VitisAI/
		EmergencyNet/          # Main Vitis AI flow for EmergencyNet/ACFF
			code/                # Train/test/quant scripts
			scripts/             # clean/train/test/quant/compile shell runners
			build/               # float, quantized, compiled outputs
			target/              # target-side xmodel artifacts

		ResNet18/              # Vitis AI flow for ResNet18
		SqeezeNet/             # Vitis AI flow for SqueezeNet
		QAT/                   # Quantization-aware training scripts/checkpoints
		New folder/            # Additional experimental branch/scripts

	real-time deployment/    # Deployment-side assets (in progress)
```

## Models

`Baseline Models/code/my_models.py` exposes model selection for:

- `MobileNet_v2`
- `MobileNet_v3`
- `SqeezeNet1_0`
- `VGG16`
- `ShuffleNet_v2`
- `EfficientNet_B0`
- `ResNet50`
- `ResNet18`

The custom `ACFFModel` (EmergencyNet variant) is implemented in:

- `Baseline Models/code/emergencyNet2.py`
- `VitisAI/EmergencyNet/code/emergencyNet2.py`

## Dataset Format

This repo uses `torchvision.datasets.ImageFolder` conventions.

### Baseline training/eval

`Baseline Models/code/main.py` and `main_eval.py` expect a class-folder dataset path (set in `data_dir`), for example:

```text
dataset/
	collapsed_building/
	fire/
	flooded_areas/
	normal/
	traffic_incident/
```

The baseline loader automatically creates a train/validation split internally.

### Vitis AI testing/quant flow

`VitisAI/EmergencyNet/code/test_E.py` reads validation data from:

```text
<data_root>/val/<class_name>/*.jpg
```

Example:

```text
VitisAI/EmergencyNet/build/data/AIDER/val/fire/...
```

## Environment Setup

A baseline `requirements.txt` is provided at repository root for Python dependencies.

### Baseline PyTorch environment

Typical Python packages used across baseline scripts:

- `torch`, `torchvision`
- `numpy`, `matplotlib`
- `scikit-learn`, `pandas`, `seaborn`
- `thop`
- `opencv-python`, `Pillow`, `scipy`

Example:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas seaborn thop opencv-python Pillow scipy
```

### Vitis AI environment

For quantization/compilation, use a Vitis AI environment that includes:

- `pytorch_nndct` (quantizer)
- `vai_c_xir` (compiler)
- target DPU `arch.json` files

The shell scripts under `VitisAI/*/scripts` assume this environment.

## Key Algorithm Implementation Map

Core EmergencyNet/ACFF components and where they are implemented:

- `SeparableConvBlock`: depthwise-separable conv block in `Baseline Models/code/emergencyNet2.py`.
- `FusionBlock`: feature fusion strategies (`add`, `max`, `con`, `avg`) in `Baseline Models/code/emergencyNet2.py`.
- `AtrousBlock`: multi-dilation atrous feature extraction + fusion in `Baseline Models/code/emergencyNet2.py`.
- `ACFFModel`: end-to-end classifier architecture in `Baseline Models/code/emergencyNet2.py`.
- Vitis AI counterpart used by quant/deploy scripts: `VitisAI/EmergencyNet/code/emergencyNet2.py`.

## Quick Start

### 1) Baseline training

```bash
cd "Baseline Models/code"
python main.py
```

Notes:

- Update `data_dir`, `model_name`, and training parameters in `main.py`.
- Best checkpoints are stored under `Baseline Models/results/`.

### 2) Baseline evaluation

```bash
cd "Baseline Models/code"
python main_eval.py
```

### 3) Vitis AI EmergencyNet flow

```bash
cd VitisAI/EmergencyNet
sh scripts/run_train_ACFF.sh
sh scripts/run_test_ACFF.sh
sh scripts/run_quant_ACFF.sh
sh scripts/run_compile.sh zcu104 zcu104_ACFFModel_int.xmodel
```

Notes:

- Set dataset and weights paths inside scripts as needed.
- Quantized artifacts are generated in `build/quantized/`.
- Compiled artifacts are generated in `build/compiled_<target>/`.

### 4) Other model pipelines

Equivalent script sets are available for:

- `VitisAI/ResNet18/scripts/`
- `VitisAI/SqeezeNet/scripts/`

### 5) Vitis AI ResNet18 deployment flow

```bash
cd VitisAI/ResNet18
sh scripts/run_test.sh
sh scripts/run_quant.sh
sh scripts/run_compile.sh zcu104 zcu104_train_resnet18_AIDER.xmodel
```

Notes:

- Floating-point test script: `scripts/run_test.sh`.
- Quantization/deploy export script: `scripts/run_quant.sh`.
- Compiler output path: `build/compiled_<target>/`.
- Example target artifact is also tracked under `target/zcu104_train_resnet18_AIDER.xmodel`.

### 6) Vitis AI SqueezeNet deployment flow

```bash
cd VitisAI/SqeezeNet
sh scripts/run_test_SQU.sh
sh scripts/run_quant_SQU.sh
sh scripts/run_compile_SQU.sh zcu104 zcu104_train_squeeze_AIDER.xmodel
```

Notes:

- Floating-point test script: `scripts/run_test_SQU.sh`.
- Quantization/deploy export script: `scripts/run_quant_SQU.sh`.
- Compiler output path: `build/compiled_<target>/`.
- Example target artifacts are tracked under `target/zcu104_train_squeeze_AIDER.xmodel` and `target/zcu104_train_squeezeD4D3_AIDER.xmodel`.

## Deployment Summary Table

| Model | Working Folder | Float Test | Quantization / Deploy Export | Compile | Example Output Artifact |
|---|---|---|---|---|---|
| EmergencyNet (ACFF) | `VitisAI/EmergencyNet` | `sh scripts/run_test_ACFF.sh` | `sh scripts/run_quant_ACFF.sh` | `sh scripts/run_compile.sh zcu104 zcu104_ACFFModel_int.xmodel` | `VitisAI/EmergencyNet/build/compile/zcu104_ACFFModel_int.xmodel` |
| ResNet18 | `VitisAI/ResNet18` | `sh scripts/run_test.sh` | `sh scripts/run_quant.sh` | `sh scripts/run_compile.sh zcu104 zcu104_train_resnet18_AIDER.xmodel` | `VitisAI/ResNet18/target/zcu104_train_resnet18_AIDER.xmodel` |
| SqueezeNet | `VitisAI/SqeezeNet` | `sh scripts/run_test_SQU.sh` | `sh scripts/run_quant_SQU.sh` | `sh scripts/run_compile_SQU.sh zcu104 zcu104_train_squeeze_AIDER.xmodel` | `VitisAI/SqeezeNet/target/zcu104_train_squeeze_AIDER.xmodel` |

Notes:

- Replace `zcu104` with your actual target board (`zcu102`, `vck190`, `v70`, `vek280`, `vck5000`) when compiling.
- Generated compiler outputs are saved under each model's `build/compiled_<target>/` directory.

## Results

### Table 1. AIDER Dataset Split Summary (70% / 15% / 15%)

| Class | Train | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| Collapsed Building | 490 | 105 | 105 | 700 |
| Fire/Smoke | 518 | 111 | 111 | 740 |
| Flood | 490 | 105 | 105 | 700 |
| Traffic Accident | 490 | 105 | 105 | 700 |
| Normal | 3990 | 855 | 855 | 5700 |
| Total | 5978 | 1281 | 1281 | 8540 |

### Table 2. Resource Utilization on ZCU104 FPGA Platform

| Resource | Available | Used | Utilization (%) |
|---|---:|---:|---:|
| DSP | 1728 | 710 | 41.09 |
| BRAM | 312 | 255 | 81.73 |
| LUT | 230400 | 51342 | 22.28 |
| FF | 460800 | 98932 | 21.47 |

### Table 3. Accuracy Comparison Between FP32 (CPU) and INT8 (FPGA)

| Model | FP32 CPU Acc. (%) | INT8 QAT (%) | INT8 PTQ (%) | Accuracy Change (QAT - FP32) | Accuracy Change (PTQ - FP32) |
|---|---:|---:|---:|---:|---:|
| EmergencyNet | 95.70 | 97.20 | 54.40 | +1.50 | -41.30 |
| ResNet-18 | 96.99 | 96.97 | 96.68 | -0.03 | -0.31 |
| SqueezeNet | 97.00 | 96.656 | 83.44 | -0.344 | -13.56 |

### Table 4. Accuracy, Latency, Throughput, and Energy Efficiency (CPU vs FPGA)

| Model | Platform | Precision | Acc. (%) | Latency (ms) | FPS | FPS/W |
|---|---|---|---:|---:|---:|---:|
| EmergencyNet | CPU | FP32 | 95.70 | 18.40 | 54 | 1.31 |
| EmergencyNet | FPGA (ZCU104) | INT8 | 97.20 | 3.75 | 267 | 29.88 |
| ResNet-18 | CPU | FP32 | 96.99 | 22.56 | 44 | 1.07 |
| ResNet-18 | FPGA (ZCU104) | INT8 | 96.97 | 5.89 | 169 | 14.90 |
| SqueezeNet | CPU | FP32 | 97.00 | 13.59 | 73 | 1.78 |
| SqueezeNet | FPGA (ZCU104) | INT8 | 96.656 | 2.86 | 349 | 50.10 |

Notes:

- `QAT` = quantization-aware training, `PTQ` = post-training quantization.
- Accuracy change is reported against FP32 CPU baseline for each model.

## Important Paths

- Baseline training entry: `Baseline Models/code/main.py`
- Baseline evaluation entry: `Baseline Models/code/main_eval.py`
- EmergencyNet quant/test entry: `VitisAI/EmergencyNet/code/test_E.py`
- EmergencyNet compile script: `VitisAI/EmergencyNet/scripts/run_compile.sh`
- ResNet18 deploy scripts: `VitisAI/ResNet18/scripts/`
- SqueezeNet deploy scripts: `VitisAI/SqeezeNet/scripts/`

## Transparency Checklist (Journal-Oriented)

This section maps repository contents to typical transparency requirements requested during manuscript screening.

- Full source code in public repository: covered (`EmergencyNet-FPGA` repository).
- Open-source project link in manuscript/abstract: add your final GitHub URL in the paper.
- DOI for source release: pending (mint DOI via Zenodo/Figshare and update links below).
- Dataset disclosure with README: partially covered (dataset format described here; see `datasets/README.md` for release template).
- Dependencies and requirements: covered (`requirements.txt` and Environment Setup section).
- Key algorithm description and implementation links: covered (Models + implementation paths).
- Testing protocol/data split disclosure: covered in the Reproducibility section below.
- Citation format mentioning article title and journal: covered (`CITATION.cff` + citation block below).

## Open-Source And DOI

Update this section once archiving is complete:

- GitHub repository: `https://github.com/omniaabdelkarim92/EmergencyNet-FPGA`
- Code archive DOI (Zenodo/Figshare): `https://doi.org/10.5281/zenodo.19651017`
- Release tag linked to manuscript: `vX.Y.Z`

Tip: connect GitHub to Zenodo, create a release, then paste the generated DOI here and in the manuscript abstract/system-submission form.

## Reproducibility And Testing Data

- Baseline training uses `ImageFolder` + internal train/val split from `Baseline Models/code/my_utils/dataset.py`.
- Baseline evaluation entrypoint: `Baseline Models/code/main_eval.py`.
- Vitis AI evaluation/quantization entrypoint: `VitisAI/EmergencyNet/code/test_E.py`.
- Default emergency classes: `collapsed_building`, `fire`, `flooded_areas`, `normal`, `traffic_incident`.
- Checkpoints and generated artifacts are stored under model-specific `results/` and `build/` folders.

For manuscript reproducibility tables, report:

- dataset version/source URL,
- train/val/test split strategy,
- exact checkpoint filename,
- hardware target (GPU + DPU board),
- software stack versions (PyTorch, Vitis AI image tag).

## Known Notes

- Some directory names use the spelling `SqeezeNet` (kept as-is in this repo).
- Paths in scripts may need adaptation to your local dataset and checkpoint names.
- Shell scripts are intended for Linux/Vitis AI environments.

## Citation

Use this format in the GitHub description/readme and manuscript supplementary material:

```bibtex
@article{<key>,
	title   = {Hardware-Aware Deployment of EmergencyNet for Real-Time UAV Disaster Recognition},
	author  = {Omnia Abdel Karim and Sameh Sherif and Rami Zewail and Koji Inoue and Mohammed S. Sayed},
	journal = {The Visual Computer},
	year    = {<YEAR>},
	doi     = {10.5281/zenodo.19651017}
}
```

Also keep `CITATION.cff` at repository root updated with final title, authors, journal, and DOI.