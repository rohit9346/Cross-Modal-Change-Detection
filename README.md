# EO-SAR Cross-Modal Change Detection

**Binary pixel-level change detection on heterogeneous satellite image pairs for disaster damage assessment**

---

## Overview

This project tackles a core problem in satellite Earth observation: detecting structural damage by comparing a pre-disaster electro-optical (EO) image with a post-disaster synthetic aperture radar (SAR) image of the same location.

The central challenge is the **cross-modal domain gap** — EO measures reflected sunlight; SAR measures microwave backscatter. These are physically incompatible measurements. Standard temporal differencing (post − pre) is meaningless across modalities. This project addresses that through physics-grounded feature engineering, a symmetric dual-encoder architecture, and attention mechanisms designed specifically for heterogeneous sensor fusion.

**Best model (V9):**
- Val IoU: **0.6050** (scenes 01–08)
- Test IoU: **0.3230** (scene 09 — unseen geography, out-of-distribution)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Approach](#approach)
3. [Architecture](#architecture)
4. [Loss Function](#loss-function)
5. [Repository Structure](#repository-structure)
6. [Setup](#setup)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Model Weights](#model-weights)
10. [Results](#results)
11. [Version History](#version-history)
12. [Key Findings](#key-findings)
13. [Known Limitations](#known-limitations)
14. [References](#references)

---

## Problem Statement

Given two satellite images of the same location:

- **Pre-event:** Electro-Optical (EO) RGB image — captured before a disaster
- **Post-event:** Synthetic Aperture Radar (SAR) grayscale image — captured after a disaster

Produce a pixel-level binary mask:
- `0` = No Change (intact)
- `1` = Change (damage, flood, destruction)

**Dataset:** 2,781 train / 334 val / 77 test image triplets across 9 geographic scenes. Scene 09 (77 triplets) is withheld as an unseen geography for domain shift evaluation.

**Label convention:**
- Raw mask values: `0` = background, `1` = uncertain/intact, `2+` = confirmed change
- Remapped automatically: `mask = np.where(mask >= 2, 1, 0)`

---

## Approach

### Feature Engineering (9 Channels)

Rather than feeding raw sensor data directly, 9 physics-grounded channels are computed to bridge the EO-SAR domain gap:

| Ch | Name | Formula | Range | Physical meaning |
|----|------|---------|-------|-----------------|
| 0–2 | EO RGB | `pre / 255.0` | [0, 1] | Pre-disaster optical colour |
| 3 | SAR | `(post − 1) / 230.0` | [0, 1] | Post-disaster radar backscatter |
| 4 | EO Luminance | `0.299R + 0.587G + 0.114B` | [0, 1] | Pre-disaster perceptual brightness |
| 5 | Cross Diff | `SAR − Lum` | [−1, 1] | +ve = rubble, −ve = flood, ~0 = no change |
| 6 | Cross Abs | `\|SAR − Lum\|` | [0, 1] | Unsigned change-likelihood magnitude |
| 7 | NGRDI | `(G−R)/(G+R+ε)`, normalised | [0, 1] | Vegetation health proxy |
| 8 | SAR-NGRDI | `SAR × (1 − NGRDI_norm)` | [0, 1] | Compound: high radar + vegetation loss |

**Cross Diff physical interpretation:**
- Positive → SAR bright, EO dim → rubble or collapsed concrete signature
- Negative → SAR dark, EO medium → flood (water absorbs microwave)
- Near zero → sensors agree → likely no change

**SAR Preprocessing:** Lee adaptive speckle filtering is applied post-normalisation to the SAR channel before derived features are computed. This reduces SAR multiplicative noise and measurably improves false-positive selectivity (+0.0025 Test IoU, −24,847 FP).

---

## Architecture

**Symmetric Dual-Encoder UNet** — pure PyTorch (no external segmentation libraries).

```
EO branch  (ResNet34, pretrained ImageNet) ──────┐
                                                  ├── CBAM attention on concatenated
SAR branch (ResNet34, pretrained ImageNet) ──────┘   EO + SAR bottleneck features

Combined features → UNet decoder with CBAM attention
                 → Pixel-level binary output (B, 1, H, W)
```

- **EO encoder:** ResNet34 with first layer accepting 3-channel EO input (RGB, unmodified from ImageNet weights)
- **SAR encoder:** ResNet34 with first layer patched for 6-channel SAR-side input (SAR_norm, EO_lum, cross_diff, cross_abs, NGRDI, SAR-NGRDI)
- **Bottleneck:** Concatenated EO(512) + SAR(512) = 1024ch → CBAM channel + spatial attention for cross-modal recalibration
- **Decoder:** CBAM (Convolutional Block Attention Module) at decoder stages d3 and d2 for channel + spatial recalibration; disabled at d1/d0 where global alignment is complete
- **Output:** Raw logits `(B, 1, H, W)`, sigmoid applied at inference

**Why dual encoders?** The EO and SAR modalities have fundamentally different statistical distributions. A single shared encoder must learn two incompatible normalisations. Separate encoders allow each modality to develop its own feature hierarchy before cross-modal integration.

**Why CBAM in the decoder?** Ablation experiments (V9 Exp 2 vs 3 vs 4) confirmed that decoder-stage CBAM — not just bottleneck CBAM — is responsible for generalisation improvement on unseen scenes. Channel attention suppresses false-positive activations; spatial attention sharpens boundary localisation.

---

## Loss Function

**Hybrid: Focal BCE + Focal Tversky Loss**

```
L = 0.3 × FocalBCE + 0.7 × FocalTversky
```

**Focal Tversky Loss** (Abraham & Khan, 2019) with corrected γ:

```
TI  = (TP + ε) / (TP + α·FP + β·FN + ε)
FTL = (1 − TI)^γ
```

| Param | Value | Reason |
|-------|-------|--------|
| α | 0.3 | FP penalty — moderate |
| β | 0.7 | FN penalty — higher (missing damage is costly) |
| γ | 1.33 | Focal exponent — corrected from buggy V4 value of 0.75 |
| smooth | 1.0 | Prevents div-by-zero on all-background tiles |

**Critical finding on γ:** V1–V4 used γ = 0.75. Fractional exponentiation on [0, 1] *amplifies* easy pixels rather than suppressing them — mathematically inverting the focal mechanism. Correcting to γ ≥ 1.0 was the single largest performance driver across all versions (+0.17 Val IoU from V4 to V5 alone).

---

## Repository Structure

```
eo-sar-change-detection/
├── notebook.ipynb          ← Full training + evaluation notebook (Kaggle)
├── config.yaml             ← All hyperparameters
├── README.md               ← This file
└── requirements.txt        ← Python dependencies
```

---

## Setup

### Requirements

```bash
pip install torch torchvision tifffile albumentations numpy
```

Or install from file:

```bash
pip install -r requirements.txt
```

> **Note:** This codebase intentionally avoids `segmentation_models_pytorch` due to a `timm` incompatibility with Python 3.12 on Kaggle. All model components are implemented in pure PyTorch.

### Dataset

The dataset follows this structure:

```
dataset/
├── train/
│   ├── pre-event/      ← EO optical .tif  (1024×1024×3, uint8, 0–255)
│   ├── post-event/     ← SAR radar .tif   (1024×1024,   uint8, 1–231)
│   └── target/         ← mask .tif        (1024×1024,   uint8)
└── change_detection_assignment_data/
    ├── val/
    └── test/
```

**Critical file handling rules:**
- Use `tifffile.imread` — never `PIL.Image.open` for `.tif` files
- Normalise before computing derived channels — never operate on raw uint8
- Always apply label remapping: `np.where(mask >= 2, 1, 0)`

---

## Training

Open `notebook.ipynb` in Kaggle with a T4 GPU accelerator and run all cells top to bottom. All hyperparameters are in Cell 1 and `config.yaml`.

```
Expected training time:  ~9 minutes per epoch on T4 GPU
Early stopping patience: 10 epochs (on validation IoU)
Checkpoint:              best_v9.pth (saved on val IoU improvement)
```

**Training pipeline automatically applies:**
- Weighted random sampling (3× oversampling of change-positive tiles)
- Geometric augmentation: horizontal flip, vertical flip, 90° rotation
- Lee adaptive speckle filtering on SAR channel
- Dual-encoder architecture with CBAM at bottleneck and decoder stages d3/d2

---

## Evaluation

Evaluation runs in the final cell after training.

**Test-Time Augmentation (TTA):** 4-way flip averaging (original + hflip + vflip + hvflip)  
**Threshold:** Sweep over [0.3, 0.35, 0.40, 0.45, 0.50] — select by val IoU  
**Metrics:** IoU, Precision, Recall, F1 — change class (label=1) only  
**Accumulation:** TP/FP/FN/TN accumulated across all batches, computed once at epoch end (never averaged per-batch)

```python
best_model = DualEncoderUNet(eo_channels=3, sar_channels=6).to(DEVICE)
best_model.load_state_dict(
    torch.load("best_v9.pth", map_location=DEVICE, weights_only=True)
)
```

---

## Model Weights

Weights are hosted on HuggingFace: [Rohit7901/eo-sar-change-detection](https://huggingface.co/Rohit7901/Cross-model-change-detection)

| Version | Architecture | Val IoU | Test IoU | Key change |
|---------|-------------|---------|---------|------------|
| **V9 (best)** | DualEncoderUNet + CBAM | **0.6050** | **0.3230** | CBAM decoder, Lee filter |
| V8 | DualEncoderUNet | 0.5810 | 0.0298 | Cross-modal bottleneck attention |
| V6 | ResNet34-UNet | 0.5200 | 0.0141 | γ corrected, baseline submission |
| V5 | ResNet34-UNet | 0.4906 | 0.0080 | γ corrected to 1.33, NGRDI fixed |
| V4 | ResNet34-UNet | 0.3137 | 0.0348 | 9 channels + NGRDI (bug present) |

---

## Results

### V9 — Current Best

**Validation split (scenes 01–08):**

| IoU | Precision | Recall | F1 |
|-----|-----------|--------|----|
| **0.6050** | 0.6763 | 0.8517 | 0.7539 |

**Test split (scene 09 — unseen geography):**

| IoU | Precision | Recall | F1 |
|-----|-----------|--------|----|
| 0.3230 | 0.4197 | 0.5837 | 0.4883 |

> The validation-to-test gap is a geographic domain shift problem. Scene 09 is an entirely different region from all training and validation scenes (scenes 01–08). The ~0.28 IoU gap is almost entirely attributable to geographic generalisation failure, not threshold miscalibration — threshold sweep analysis confirmed only ~3.8% of the gap is recoverable by threshold adjustment alone.

---

## Version History

| Version | Channels | Architecture | Val IoU | Test IoU | Key change |
|---------|----------|-------------|---------|----------|------------|
| V1 | 8 (redundant sar_int) | ResNet34-UNet | 0.4423 | — | Baseline; only 3 epochs ran |
| V2 | 7 | ResNet34-UNet | 0.4757 | 0.012 | Removed redundant channel |
| V3 | 7 | ResNet34-UNet | 0.4198 | 0.018 | 512px crops — context lost |
| V4 | 9 (NGRDI bug) | ResNet34-UNet | 0.3137 | 0.035 | NGRDI normalisation bug |
| V5 | 9 (fixed) | ResNet34-UNet | 0.4906 | 0.008 | γ corrected 0.75→1.33 |
| V6 | 9 | ResNet34-UNet | 0.5200 | 0.0141 | Hybrid Focal BCE + FTL |
| V7 | 9 | ResNet34-UNet | 0.5480 | 0.2650 | Bottleneck cross-attention |
| V8 | 9 | DualEncoderUNet | 0.5810 | 0.2980 | Symmetric dual encoders |
| **V9** | **9** | **DualEncoderUNet + CBAM** | **0.6050** | **0.3230** | **CBAM at bottleneck + decoder** |
| V10 | 9 | DualEncoderUNet + CBAM | 0.5891 | 0.2904 | SAR augmentation (backfired — removed) |

---

## Key Findings

### 1. γ correction was the single largest performance driver

All versions V1–V4 used γ = 0.75 in Focal Tversky Loss. Fractional exponentiation on values in [0, 1] amplifies easy pixels rather than suppressing them — the opposite of the intended focal mechanism. Correcting γ to ≥ 1.0 improved Val IoU by ~0.17 points from V4 to V5, faster and more reliably than any architectural change.

### 2. CBAM decoder placement is critical for generalisation

Ablation experiments in V9 confirmed that decoder-stage CBAM — not just bottleneck CBAM — drives meaningful improvement on the unseen test scene. Bottleneck-only CBAM improves val IoU but does not help test IoU. Adding CBAM at decoder stages d3/d2 improved Test IoU from 0.2980 to 0.3230, confirming it suppresses scene-specific false positives.

### 3. Symmetric dual encoders outperform single-encoder fusion

A single ResNet34 encoder must reconcile two physically incompatible feature distributions (optical vs microwave). Symmetric dual encoders with CBAM at the bottleneck improved Test IoU from 0.0141 (V6) to 0.2980 (V8) — a 21× improvement — while also improving Val IoU from 0.5200 to 0.5810.

### 4. SAR speckle filtering improves false-positive selectivity

Lee adaptive speckle filtering applied to the SAR channel post-normalisation reduced false positives on the test set by ~24,847 pixels and improved Test IoU by +0.0025. The threshold sweep began peaking cleanly at 0.45 after filtering, confirming improved model calibration.

### 5. SAR augmentation backfires when FP selectivity is the bottleneck

V10 added SAR multiplicative noise augmentation to simulate speckle variation, expecting improved generalisation. Instead, Test IoU dropped from 0.3230 to 0.2904 and FP count increased by ~127,000. Analysis confirmed the bottleneck was already false-positive selectivity, not speckle sensitivity — augmenting speckle further confused the model about what constitutes a stable background. Removed in V11.

### 6. Full-resolution context outperforms crops

V2 (1024px, Val IoU 0.4757) outperformed V3 (512px crops, 0.4198) by 0.056 IoU. Collapsed multi-story buildings spanning 600–800 pixels cannot be understood from a 512px fragment — surrounding intact structures provide essential reference context for confirming change.

---

## Training Configuration

```python
# Encoder / Decoder
IN_CHANNELS   = 9           # 3 EO RGB + 6 SAR-side channels
BATCH_SIZE    = 4
NUM_EPOCHS    = 60          # with early stopping (patience=10)
ENCODER_LR    = 1e-5        # conservative — pretrained weights
DECODER_LR    = 1e-3        # aggressive — freshly initialised
WEIGHT_DECAY  = 1e-4
T_MAX         = 50          # CosineAnnealingLR
ETA_MIN       = 1e-6
CROP_SIZE     = 512

# Loss
FOCAL_ALPHA   = 0.3         # FP weight in Tversky
FOCAL_BETA    = 0.7         # FN weight in Tversky
FOCAL_GAMMA   = 1.33        # Corrected from V4 bug (was 0.75); focal exponent ≥ 1 required
```

**Augmentation (training only):** `RandomCrop(512)`, `HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.5)`, `RandomRotate90(p=0.5)`. No colour augmentation — SAR channels have fundamentally different statistics from optical channels and must not be jointly perturbed.

**Inference:** 4-pass Test-Time Augmentation (original + h-flip + v-flip + hv-flip, averaged), threshold = 0.4.

---

## Known Limitations

**Geographic generalisation gap:** The model is trained and validated on 8 scenes and evaluated on a single unseen scene (scene 09). With only 9 total scenes in the dataset, the model cannot learn scene-invariant representations robustly. This is a dataset-scale limitation, not purely an architectural one.

**Domain shift diagnosis:** Threshold sweep analysis on scene 09 shows no clean peak across thresholds [0.3–0.7], indicating the model is systematically under-confident on the unseen geography — a calibration signature of distribution shift. This is expected behaviour and is documented here rather than obscured.

**SAR modality only for post-event:** Using SAR post-event and EO pre-event is the realistic disaster response scenario (radar penetrates cloud cover). However, it means the model must learn to compare incompatible sensor modalities rather than detecting temporal change within a single modality.

**No NIR channel:** NDVI (which requires near-infrared) is not available — NGRDI (visible green-red) is used as a proxy. True NDVI would provide stronger vegetation destruction signal.

---

## Reproducibility

All experiments use `SEED = 42`:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Running the notebook end-to-end on the same dataset with the same Kaggle environment (Python 3.12, PyTorch 2.x, T4 GPU) reproduces the reported metrics.

---

## References

1. Abraham & Khan (2019). A novel focal Tversky loss function with improved attention U-Net for lesion segmentation. *ISBI*.
2. Ronneberger et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
3. He et al. (2016). Deep residual learning for image recognition. *CVPR*.
4. Woo et al. (2018). CBAM: Convolutional block attention module. *ECCV*.
5. Lee (1981). Refined filtering of image noise using local statistics. *Computer Graphics and Image Processing*.
6. Tucker (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*.

---

*Rohit Robert | B.Sc. Computer Science, Kristu Jayanti College | MCA candidate*
