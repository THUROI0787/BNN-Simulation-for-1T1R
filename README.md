# Binary Neural Network with 1T1R Memristor Crossbar

This repository contains the simulation and training code for our Nature Communications paper on **binary neural networks (BNNs) deployed on 1T1R memristor arrays**. The work explores binarizing a Network-in-Network (NIN) backbone for image classification (CIFAR-10, MNIST), also replacing the final classification layer with a 0/1-weight memristor crossbar, and systematically evaluating robustness under noise, quantization, and sparsity.

## Project Structure

```
.
├── README.md
├── train.py                 # Unified training script
├── test.py                  # Unified evaluation script
├── add_noise.py             # Noise injection for robustness experiments
├── simulation.py            # Standalone 1T1R array circuit simulation
├── models/
│   ├── __init__.py
│   └── nin.py               # All model architectures
└── utils/
    ├── __init__.py
    ├── binop.py             # Binary weight operator (XNOR-Net style)
    ├── data_loaders.py      # CIFAR-10 / MNIST dataset builders
    └── functions.py         # Seed, logger, TET loss utilities
```

## Available Architectures

### CIFAR-10 (32×32, 3 channels, 10 classes)

| Architecture | Description | Params |
|---|---|---|
| `Net_00` | First & last layers full-precision (FP) | ~970K |
| `Net_01` | First FP, last binarized | ~970K |
| `Net_11` | First & last binarized | ~970K |
| `Net_small` | Compact VGG-style BNN | ~500K |
| `Net_small2` | Ultra-lightweight BNN | ~150K |
| `Net_mid` | Medium NIN-style BNN | ~330K |
| `Net_mid2` | NIN + 10×10 1T1R FC (2-stage) | ~662K |
| `Net_mid3` | NIN + 192×10 1T1R FC (2-stage) | ~661K |

### MNIST (28×28, 1 channel, 10 classes)

| Architecture | Description | Params |
|---|---|---|
| `Net_00_MNIST` | First & last FP | ~961K |
| `Net_01_MNIST` | First FP, last binarized | ~961K |
| `Net_11_MNIST` | First & last binarized | ~961K |

## Installation

```bash
pip install torch torchvision numpy
```

## Training

All training is done through `train.py`. The script supports both single-stage (standard BNN) and two-stage (1T1R memristor FC) training.

### CIFAR-10 Examples

**Train a standard BNN (Net_01):**
```bash
python train.py \
    --dataset cifar10 \
    --arch Net_01 \
    --epochs 200 \
    --lr 0.01 \
    --batch_size 128 \
    --log my_net01_exp
```

**Train Net_mid2 with 2-stage 1T1R training:**
```bash
python train.py \
    --dataset cifar10 \
    --arch Net_mid2 \
    --epochs 200 \
    --epochs_finetune 50 \
    --lr 0.01 \
    --lr_finetune 0.01 \
    --log mid2_full
```

**Skip Stage 1 and only finetune the 1T1R FC layer (requires a pretrained Stage 1 checkpoint):**
```bash
python train.py \
    --dataset cifar10 \
    --arch Net_mid2 \
    --skip_stage1 \
    --pretrained ckpt/mid2_full_stage1_best.pth \
    --epochs_finetune 50 \
    --lr_finetune 0.01 \
    --log mid2_s2_only
```

### MNIST Examples

```bash
python train.py \
    --dataset mnist \
    --arch Net_01_MNIST \
    --epochs 20 \
    --lr 0.001 \
    --batch_size 128 \
    --log mnist_exp
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10` or `mnist` |
| `--arch` | `Net_01` | Model architecture (see table above) |
| `--epochs` | `200` | Stage 1 training epochs |
| `--lr` | `0.01` | Stage 1 initial learning rate |
| `--batch_size` | `128` | Training batch size |
| `--weight_decay` | `1e-5` | L2 weight decay |
| `--skip_stage1` | `False` | Skip CNN training (requires `--pretrained`) |
| `--epochs_finetune` | `50` | Stage 2 FC finetuning epochs |
| `--lr_finetune` | `0.01` | Stage 2 learning rate |
| `--pretrained` | `None` | Path to pretrained checkpoint |
| `--log` | `default_exp` | Experiment name for logs and checkpoints |
| `--save_dir` | `ckpt` | Directory for checkpoints and logs |
| `--seed` | `42` | Random seed |

### Checkpoint Output

After training, checkpoints are saved to `--save_dir` (default: `ckpt/`):

- **Single-stage models**: `ckpt/{log}_stage1_best.pth`
- **Two-stage models**: `ckpt/{log}_stage1_best.pth` (CNN) and `ckpt/{log}_stage2_best.pth` (full model with trained FC)

Each checkpoint contains:
- `state_dict`: model weights
- `acc`: best test accuracy achieved
- `epoch`: epoch at which the checkpoint was saved
- `config` (Stage 2 only): training arguments for reproducibility

Training logs are saved to `ckpt/{log}.txt`.

## Testing & Evaluation

All evaluation is done through `test.py` with six modes.

### 1. Standard Evaluation

```bash
python test.py \
    --dataset cifar10 \
    --arch Net_01 \
    --ckpt ckpt/my_net01_exp_stage1_best.pth
```

### 2. 1T1R FC Evaluation (with Identity Baseline)

Evaluate the 10×10 memristor mapping and compare against an identity baseline:

```bash
# With learned 1T1R weights
python test.py \
    --dataset cifar10 \
    --arch Net_mid2 \
    --ckpt ckpt/mid2_full_stage2_best.pth \
    --mode fc

# Force identity (baseline: what accuracy without the 1T1R layer?)
python test.py \
    --dataset cifar10 \
    --arch Net_mid2 \
    --ckpt ckpt/mid2_full_stage2_best.pth \
    --mode fc \
    --force_identity
```

### 3. Confusion Matrix

Generates a 10×10 confusion matrix and saves it as CSV:

```bash
python test.py \
    --dataset cifar10 \
    --arch Net_mid2 \
    --ckpt ckpt/mid2_full_stage2_best.pth \
    --mode confusion
```

Output: `confusion_matrix_{ckpt_name}.csv`

### 4. Noise Robustness Sweep

Adds Gaussian noise to binarized weights and sweeps standard deviation:

```bash
python test.py \
    --dataset cifar10 \
    --arch Net_01 \
    --ckpt ckpt/my_model.pth \
    --mode noise \
    --noise_std_max 0.2 \
    --noise_steps 25
```

### 5. Quantization Evaluation

Quantizes the first convolutional layer to N-bit precision:

```bash
python test.py \
    --dataset cifar10 \
    --arch Net_01 \
    --ckpt ckpt/my_model.pth \
    --mode quant \
    --n_bits 4 \
    --clip_percentile 0.25
```

### 6. Sparsity Measurement

Measures the ratio of +1 weights in each binarized convolutional layer:

```bash
python test.py \
    --dataset cifar10 \
    --arch Net_01 \
    --ckpt ckpt/my_model.pth \
    --mode sparsity
```

### Key Test Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10` or `mnist` |
| `--arch` | *(required)* | Model architecture |
| `--ckpt` | *(required)* | Path to checkpoint |
| `--mode` | `standard` | `standard`, `noise`, `quant`, `sparsity`, `fc`, `confusion` |
| `--noise_std_max` | `0.2` | Max noise std for sweep |
| `--noise_steps` | `25` | Number of noise levels |
| `--n_bits` | `4` | Quantization bit-width |
| `--force_identity` | `False` | Force 1T1R FC to identity (baseline) |

## Noise Injection (add_noise.py)

For generating noisy checkpoints to simulate memristor device faults:

```bash
python add_noise.py \
    --base_ckpt ckpt/mid2_full_stage1_best.pth \
    --save_dir ckpt/noisy_models
```

Supported noise types (edit `configs` in the script):
- `flip`: bit-flip (weight × -1)
- `drop`: stuck-at-0 (weight → 0)
- `stuck`: stuck-at-1 (weight → |weight| + 0.1)

Target layers: `all`, `shallow`, `mid`, `deep`.

## 1T1R Array Simulation (simulation.py)

`simulation.py` is a standalone circuit-level simulation of a 1T1R memristor crossbar array. It models:

- **IR drop**: voltage drop across parasitic wire resistance for the farthest cell
- **Sneak path**: leakage current through unselected cells on the same bitline
- **Write disturb**: analysis of half-selected cell disturbance

Run it directly:
```bash
python simulation.py
```

The simulation uses realistic device parameters (LRS ~10kΩ, HRS ~21MΩ, transistor Ron = 100Ω) and supports configurable array sizes (10×10 and 100×100).

## Two-Stage Training Explained

For `Net_mid2` and `Net_mid3`, training proceeds in two stages:

1. **Stage 1 (CNN training)**: The 1T1R FC layer is set to identity mode and frozen. Only the NIN convolutional backbone is trained with standard BNN binarization (XNOR-Net style via `BinOp`).

2. **Stage 2 (FC finetuning)**: The CNN backbone is frozen. The 1T1R FC layer is unfrozen and trained with sigmoid-constrained 0/1 weights (via `STE_SigmoidRound`), simulating the memristor crossbar's binary conductance states.

This two-stage approach ensures the CNN learns good features before the memristor-compatible classifier is optimized.

## Citation

If you use this code, please cite our Nature Communications paper (forthcoming).
