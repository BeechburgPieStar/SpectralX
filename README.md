# SpectralX: A Unified Spectral-Aware Architecture for Radio Signal Analysis

Official PyTorch implementation of the IEEE Communications Letters paper
**"SpectralX: A Unified Spectral-Aware Architecture for Radio Signal Analysis"**.

SpectralX is a **unified, lightweight** backbone for three heterogeneous radio-signal tasks:

| Task | Full name | Benchmark | Folder |
|------|-----------|-----------|--------|
| **AMC** | Automatic Modulation Classification | RML2018.01a | [`AMC/`](AMC) |
| **RFF** | Radio Frequency Fingerprinting | ORACLE | [`RFF/`](RFF) |
| **WS**  | Wireless Sensing (joint Activity Recognition + Indoor Localization) | ARIL | [`JARIL/`](JARIL) |

The same architecture (`MSTM → ASR`) is reused across all three tasks with only task-specific
input dimensions and classification heads, and comes in three sizes **S / M / L**.

---

## Highlights

- **Unified serial architecture.** A single backbone handles AMC, RFF and WS by changing only the input channels and the output head.
- **Time–frequency joint modeling.** Multi-Scale Temporal Modeling (MSTM) captures multi-resolution temporal dependencies; Adaptive Spectral Refinement (ASR) performs instance-adaptive spectral filtering via an FFT-based learnable mask.
- **Lightweight.** SpectralX-S reaches competitive accuracy with as few as ~30K parameters, several times smaller than transformer-based baselines.
- **Plug-and-play spectral module.** The ASR block adds < 0.3% parameters and < 0.5% MACs, yet consistently improves accuracy and cross-domain robustness (see ablations).

---

## Architecture Overview

```
Input signal  x  ∈ R^{C×L}
      │
      ▼
┌──────────────────────────────────────────┐
│ MSTM  —  Multi-Scale Temporal Modeling    │   XceptionBlock
│  • stacked XceptionTime modules           │   (multi-scale depthwise
│  • MS-DSConv kernels {41,21,11} + MaxPool │    separable convs + residual)
│  • residual connections, channels 8d→32d  │
└──────────────────────────────────────────┘
      │  z ∈ R^{32d×L}
      ▼
┌──────────────────────────────────────────┐
│ ASR  —  Adaptive Spectral Refinement      │   Adaptive_Spectral_Block
│  • LN → real FFT                          │   (a.k.a. ASB in code)
│  • global path : learnable complex weight │
│  • selective path : energy-median mask +  │
│    straight-through learnable threshold τ │
│  • inverse FFT back to time domain        │
└──────────────────────────────────────────┘
      │  ẑ ∈ R^{32d×L}
      ▼
  Bottleneck (AAP → PConv 32d→16d→8d) → Classifier (PConv → GAP)
      │
      ▼
   logits
```

> **Naming note.** The *Adaptive Spectral Refinement (ASR)* module described in the paper is
> implemented in code as the `Adaptive_Spectral_Block` class, and is toggled through the
> `use_asb` / `--no_asb` flags. Ablation weights are suffixed `_wo_ASB`.

Model sizes are controlled by the base channel width `nf`:

| Variant | `nf` | CLI `--model_size` |
|---------|------|--------------------|
| SpectralX-S | 4  | `S` |
| SpectralX-M | 8  | `M` |
| SpectralX-L | 16 | `L` |

---

## Repository Structure

```
SpectralX/
├── AMC/                       # Automatic Modulation Classification (RML2018.01a)
│   ├── main.py
│   └── utils/
│       ├── SpectralX.py       # model (single classification head)
│       └── get_dataset.py     # loads ./dataset/{train,val,test}_snr_*.h5
├── RFF/                       # Radio Frequency Fingerprinting (ORACLE)
│   ├── main.py
│   └── utils/
│       ├── SpectralX.py       # model returns (embedding, logits)
│       └── get_dataset.py     # loads ./dataset/run{1,2}/x_*_{ft}ft.npy
├── JARIL/                     # Wireless Sensing: joint AR + IL (ARIL)
│   ├── main.py
│   └── utils/
│       ├── SpectralX.py       # model with dual heads (activity + location)
│       └── get_dataset.py     # loads ./datasets/*_data_split_amp.mat
└── README.md
```

Each task folder is **self-contained** and is meant to be run with its own folder as the
working directory (paths inside the code are relative to that folder).

---

## Requirements

- Python ≥ 3.9
- PyTorch **2.5.1** (as used in the paper; other recent 2.x versions should also work)
- CUDA-capable GPU (experiments were run on an NVIDIA RTX 3090)

Install dependencies:

```bash
pip install torch==2.5.1 torchvision numpy h5py scipy scikit-learn timm torchsummary
```

| Package | Used by |
|---------|---------|
| `torch`, `torchsummary` | all tasks |
| `timm` | `trunc_normal_` init inside the ASR block |
| `h5py` | AMC (`.h5` datasets) |
| `scipy` | JARIL (`.mat` datasets) |
| `scikit-learn` | RFF / JARIL (train/val split) |
| `torchvision` | imported in RFF `main.py` |

---

## Datasets and Pretrained Weights

The datasets and pretrained weights are **not** included in this repository. Download them from
Baidu Netdisk and place them under each task folder as described below.

> **百度网盘 / Baidu Netdisk**
> 链接 (Link): https://pan.baidu.com/s/1iQazU9mAGP5vL6ynLIAnYQ
> 提取码 (Code): `8mg9`
> 分享文件 (Shared folder): `code_SpectralX`

After downloading, arrange the files so each task sees its data and weights at the expected
relative paths.

### AMC (RML2018.01a)

`get_dataset.py` reads per-SNR HDF5 shards from `./dataset/`:

```
AMC/
├── dataset/
│   ├── train_snr_*.h5     # each file has datasets 'X' (N,128,2) and 'Y' (N,)
│   ├── val_snr_*.h5
│   └── test_snr_-20.h5 ... test_snr_30.h5   # step of 2 dB
└── weights/               # create this folder; put SpectralX_{S,M,L}.pth here
```

### RFF (ORACLE)

`get_dataset.py` reads `.npy` arrays organized by capture run:

```
RFF/
├── dataset/
│   ├── run1/
│   │   ├── x_train_2ft.npy   y_train_2ft.npy
│   │   └── x_test_2ft.npy    y_test_2ft.npy
│   └── run2/
│       ├── x_train_2ft.npy   y_train_2ft.npy
│       └── x_test_2ft.npy    y_test_2ft.npy
└── weights/                  # create this folder; whole model is saved here
```

`run1` is the **source domain (S)** and `run2` is the **target domain (T)** used to measure
cross-time robustness. This is controlled by `--sd_time_ft` and `--td_time_ft`
(default `[1 2]` and `[2 2]`, i.e. `[run, ft]`).

### WS / JARIL (ARIL)

`get_dataset.py` reads MATLAB `.mat` files from `./datasets/`:

```
JARIL/
├── datasets/
│   ├── train_data_split_amp.mat   # keys: train_data, train_activity_label, train_location_label
│   └── test_data_split_amp.mat    # keys: test_data,  test_activity_label,  test_location_label
└── weights/                       # auto-created by main.py; put SpectralX_{S,M,L}.pth here
```

> **Note on the `weights/` folder.** `JARIL/main.py` calls `os.makedirs("weights", exist_ok=True)`
> automatically. For **AMC** and **RFF** you should create the `weights/` folder manually
> (`mkdir weights`) before training, or before running `--mode test` with downloaded weights.

Expected input shapes per task (also used by `torchsummary`):

| Task | Input `(C, L)` | #Classes |
|------|----------------|----------|
| AMC  | `(2, 128)`  | 24 |
| RFF  | `(2, 6000)` | 16 |
| WS   | `(52, 192)` | 6 (activity) + 16 (location) |

---

## Usage

Run each task from **inside its own folder** so the relative `dataset/` and `weights/` paths resolve.

### AMC — RML2018.01a

```bash
cd AMC
mkdir -p weights

# Train + test SpectralX-M
python main.py --mode train_test --model_size M --cuda 0

# Test only (requires weights/SpectralX_M.pth)
python main.py --mode test --model_size M --cuda 0

# Ablation: without the ASR/ASB module (weights/SpectralX_M_wo_ASB.pth)
python main.py --mode train_test --model_size M --no_asb --cuda 0
```

Testing sweeps SNR from **−20 dB to +30 dB** (step 2 dB) and prints per-SNR accuracy.

Key arguments (`AMC/main.py`): `--batch_size 512`, `--epochs 100`, `--lr 1e-4`,
`--wd 1e-5`, `--num_classes 24`, `--patience 5` (early stopping), `--seed 2023`.

### RFF — ORACLE

```bash
cd RFF
mkdir -p weights

# Train on source (run1) and evaluate on both source (S) and target (T = run2)
python main.py --mode train_test --model_size S --cuda 0 \
    --sd_time_ft 1 2 --td_time_ft 2 2
```

Key arguments (`RFF/main.py`): `--batch_size 32`, `--epochs 200`, `--lr 1e-3`,
`--num_classes 16`. The model is saved as a **full model object** (`torch.save(model, ...)`),
and testing reports accuracy on both the source and target domains.

### WS — ARIL (joint Activity Recognition + Indoor Localization)

```bash
cd JARIL

# Train + test SpectralX-L (weights/ is created automatically)
python main.py --mode train_test --model_size L --cuda 0
```

Key arguments (`JARIL/main.py`): `--batch_size 128`, `--epochs 200`, `--lr 5e-3`,
`--in_channels 52`, `--num_classes_act 6`, `--num_classes_loc 16`,
`--loss_weight_act 0.5 --loss_weight_loc 0.5` (multi-task loss weighting),
with a `MultiStepLR` schedule (`gamma=0.5`). Testing reports both AR and IL accuracy.

> The default `--cuda` index is `0` for AMC and `1` for RFF/JARIL — set `--cuda` to a valid GPU id on your machine.

## Citation

If you find this work useful, please cite:

```bibtex
@article{huang2025spectralx,
  title   = {SpectralX: A Unified Spectral-Aware Architecture for Radio Signal Analysis},
  author  = {Huang, Hao and Wang, Yu and Shi, Zheng},
  journal = {Submitted to IEEE Communications Letters},
  year    = {2026}
}
```
