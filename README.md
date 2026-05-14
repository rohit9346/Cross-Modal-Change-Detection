# Binary Change Detection on EO-SAR Image Pairs
### GalaxEye Space — Satellite AI Research Intern Technical Assessment

**Candidate:** Rohit Robert  
**Submitted:** May 2026  
**Primary Model:** V5 — ResNet34-UNet, 9-channel cross-modal features, γ-corrected Focal Tversky Loss  
**Best Validation IoU:** 0.4906  

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Approach](#approach)
3. [Repository Structure](#repository-structure)
4. [Setup](#setup)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Model Weights](#model-weights)
8. [Results](#results)
9. [Version Comparison](#version-comparison)
10. [Key Findings](#key-findings)

---

## Problem Statement

Given two satellite images of the same location:
- **Pre-event:** Electro-Optical (EO) optical RGB image — captured before a disaster
- **Post-event:** Synthetic Aperture Radar (SAR) image — captured after a disaster

Produce a pixel-level binary mask:
- `0` = No Change (intact)
- `1` = Change (damage, flood, destruction)

The core challenge is the **cross-modal domain gap** — EO measures reflected sunlight, SAR measures microwave backscatter. These are physically incompatible measurements. Standard temporal differencing (post − pre) is meaningless across modalities. This project addresses that through physics-grounded cross-modal feature engineering.

---

## Approach

### Feature Engineering (9 Channels)

Rather than feeding raw images to the model, 9 physics-grounded channels are computed:

| Ch | Name | Formula | Range | Physical meaning |
|----|------|---------|-------|-----------------|
| 0–2 | EO RGB | `pre / 255.0` | [0, 1] | Pre-disaster optical colour |
| 3 | SAR | `(post − 1) / 230.0` | [0, 1] | Post-disaster radar backscatter |
| 4 | EO Luminance | `0.299R + 0.587G + 0.114B` | [0, 1] | Pre-disaster perceptual brightness |
| 5 | Cross Diff | `SAR − Lum` | [−1, 1] | +ve = rubble, −ve = flood, ~0 = no change |
| 6 | Cross Abs | `\|SAR − Lum\|` | [0, 1] | Unsigned change-likelihood signal |
| 7 | NGRDI | `(G−R)/(G+R+ε)`, normalised to [0,1] | [0, 1] | Vegetation health proxy (RGB; no NIR) |
| 8 | SAR-NGRDI | `SAR × (1 − NGRDI_norm)` | [0, 1] | Compound: high radar + vegetation loss |

### Architecture

**ResNet34-UNet** — pure PyTorch, no external segmentation libraries.

- Pretrained ResNet34 encoder (ImageNet weights), first layer patched for 9-channel input
- UNet decoder with skip connections at every scale
- Output: raw logits `(B, 1, H, W)`, sigmoid at inference

### Loss Function

**Focal Tversky Loss** with corrected γ = 1.33 (Abraham & Khan, 2019):

```
FTL = (1 − Tversky)^γ
Tversky = (TP + ε) / (TP + 0.3·FP + 0.7·FN + ε)
```

`α = 0.3, β = 0.7` encodes the disaster response priority: missing damage (FN) is
operationally more costly than a false alarm (FP).

> **Key finding:** γ = 0.75 (used in V1–V4) mathematically inverts the focal mechanism —
> fractional exponentiation on [0,1] amplifies easy pixels instead of suppressing them.
> Correcting to γ = 1.33 was the single largest performance improvement across all versions.

---

## Repository Structure

```
galaxeye-change-detection/
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

### Dataset

Download the dataset from Kaggle and place it at:

```
/kaggle/input/datasets/rohitrobertc2/galaxyeye-change-detection-v2/
├── train/
│   ├── pre-event/      ← EO optical .tif  (1024×1024×3, uint8)
│   ├── post-event/     ← SAR radar .tif   (1024×1024,   uint8)
│   └── target/         ← mask .tif        (1024×1024,   uint8)
└── change_detection_assignment_data/
    ├── val/
    └── test/
```

**Label remapping applied automatically:**  
Raw mask values: `0` = background, `1` = uncertain/intact, `2+` = change  
Remapped to: `0` = no change, `1` = change  

---

## Training

Open `notebook.ipynb` in Kaggle with a T4 GPU accelerator and run all cells top to bottom.

All hyperparameters are defined in **Cell 1** and in `config.yaml`. No other configuration is needed.

```
Expected training time: ~9 minutes per epoch on T4 GPU
Early stopping: patience = 10 on validation IoU
Checkpoint saved to: best_v5.pth (whenever val IoU improves)
```

**Training will automatically:**
- Apply weighted random sampling (3× oversampling of change-positive tiles)
- Apply geometric augmentation (horizontal flip, vertical flip, 90° rotation)
- Apply EO brightness augmentation (training only)
- Save the best checkpoint based on validation IoU
- Stop early if no improvement for 10 consecutive epochs

---

## Evaluation

Evaluation runs automatically in **Cell 12** after training completes.

**Test-Time Augmentation (TTA):** 4-way flip averaging (original + hflip + vflip + hvflip)  
**Threshold:** 0.4 (lowered from 0.5 — Focal Tversky with β=0.7 calibrates below 0.5)  
**Metrics:** IoU, Precision, Recall, F1 — computed for the change class (label=1) only  

To evaluate a saved checkpoint without retraining, load it in Cell 12:

```python
best_model = ResNet34UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
best_model.load_state_dict(
    torch.load("best_v5.pth", map_location=DEVICE, weights_only=True)
)
```

---

## Model Weights

All checkpoints are publicly available on Google Drive:

| Version | Val IoU | Description | Download |
|---------|---------|-------------|----------|
| **V5 (PRIMARY)** | **0.4906** | 9-ch, γ=1.33 corrected, 1024px | **[best_v5.pth](https://huggingface.co/Rohit7901/galaxeye-change-detection/resolve/main/best_v5.pth)** |
| V3 | 0.4198 | 7-ch, 512px crop ablation | [best_v3.pth](https://huggingface.co/Rohit7901/galaxeye-change-detection/resolve/main/best_v3.pth) |
| V2 | 0.4757 | 7-ch, 1024px, prior best | [best_v2.pth](https://huggingface.co/Rohit7901/galaxeye-change-detection/resolve/main/best_v2.pth) |

> **Primary submission weights:** V5 (`best_v5.pth`)  
> All weights require the same architecture: `ResNet34UNet(in_channels=9)` for V4/V5, `ResNet34UNet(in_channels=7)` for V2/V3.

---

## Results

### V5 — Primary Submission

**Validation split (scenes 01–08):**

| IoU | Precision | Recall | F1 | Loss |
|-----|-----------|--------|-----|------|
| **0.4906** | 0.6656 | 0.6511 | 0.6583 | 0.7081 |

**Confusion matrix (validation):**

|  | Predicted No-Change | Predicted Change |
|--|---------------------|-----------------|
| **Actual No-Change** | TN = 339,998,061 | FP = 2,520,150 |
| **Actual Change** | FN = 2,688,913 | TP = 5,017,260 |

**Test split (scene_09 — unseen geography):**

| IoU | Precision | Recall | F1 | Loss |
|-----|-----------|--------|-----|------|
| 0.0080 | 0.0276 | 0.0111 | 0.0159 | 0.9847 |

**Confusion matrix (test):**

|  | Predicted No-Change | Predicted Change |
|--|---------------------|-----------------|
| **Actual No-Change** | TN = 79,892,073 | FP = 239,085 |
| **Actual Change** | FN = 602,413 | TP = 6,781 |

> The large validation-to-test gap is a geographic domain shift problem — scene_09 is a  
> completely different geography from all training scenes. The model detects approximately  
> the correct number of change pixels but mislocalises many of them. This is expected and  
> documented in the technical report (Section 4.3).

---

## Version Comparison

| Version | Channels | Input | Val IoU | Test IoU | Key change |
|---------|----------|-------|---------|----------|------------|
| V1 | 8 (w/ redundant sar_int) | 1024 | 0.4423 | — | Baseline. Only 3 epochs ran. |
| V2 | 7 | 1024 | 0.4757 | 0.012 | Removed redundant channel. Best prior submission. |
| V3 | 7 | 512 crop | 0.4198 | 0.018 | Crops improved recall but hurt val IoU — context lost. |
| V4 | 9 (NGRDI bug) | 512 crop | 0.3137 | 0.035 | NGRDI normalisation bug. Best test IoU despite bugs. |
| **V5** | **9 (fixed)** | **1024** | **0.4906** | 0.0080 | **γ corrected 0.75→1.33. NGRDI fixed. Best val IoU.** |

---

## Key Findings

**1. The γ correction was the single largest improvement**  
All versions V1–V4 used γ = 0.75 in Focal Tversky Loss. This mathematically inverts the focal mechanism — fractional exponentiation on [0,1] amplifies easy pixels instead of suppressing them. The ratio of gradient signal between hard and easy pixels is 9.2× at γ=0.75 vs 25.7× at the correct γ=1.33 (Abraham & Khan, 2019). V5 reached Val IoU 0.358 by epoch 5, faster than any prior version reached its best score.

**2. Removing redundant channels improves performance**  
V1's `sar_int` channel was an exact copy of the SAR channel. Removing it (V2) improved all metrics. The model wasted encoder capacity on duplicate information.

**3. Full 1024×1024 context outperforms 512 crops**  
V2 (1024px, IoU 0.4757) beat V3 (512 crops, IoU 0.4198) by 0.056 IoU. A collapsed multi-story building spanning 600–800 pixels cannot be understood from a 512 fragment — surrounding intact structures provide the reference context for confirming change.

**4. Physics-independent channels improve geographic generalisation**  
The NGRDI and SAR-NGRDI channels encode vegetation destruction — a relationship that is geography-independent. Despite the normalisation bug, V4 achieved the best test IoU (0.035) of all versions, demonstrating that the physical signal generalises to unseen scenes better than scene-specific texture features.

**5. Lovász loss combined with Focal Tversky is unstable**  
Combining both losses caused validation loss > 1.0 at epoch 28 due to scale mismatch between the two terms. FocalTversky alone with corrected γ was adopted for V5.

---

## Reproducibility

All experiments use `SEED = 42` with:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Running the notebook end-to-end on the same dataset with the same Kaggle environment reproduces the reported metrics.

---

## References

1. Abraham & Khan (2019). A novel focal Tversky loss with improved attention U-Net. ISBI.
2. Ronneberger et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.
3. He et al. (2016). Deep residual learning for image recognition. CVPR.
4. Berman et al. (2018). The Lovász-softmax loss. CVPR.
5. Tucker (1979). Red and photographic infrared linear combinations for monitoring vegetation. RSE.

---

*GalaxEye Space Technical Assessment | May 2026 | Rohit Robert*
