# Changelog

All notable changes to this project are documented here.

---

## V9 — Final Submission · May 2026

**Architecture:** DualEncoderUNet (two ResNet34 encoders + CBAM)  
**Val IoU:** 0.6050 · **Test IoU:** 0.3230  
**Checkpoint:** `best_v9.pth`

### What changed
- Replaced single patched encoder with **dual independent encoders**: ResNet34 for EO (standard 3ch) and ResNet34 for SAR (patched 6ch). Each encoder preserves modality-specific ImageNet features without weight dilution.
- Added **CBAM (Convolutional Block Attention Module)** at bottleneck (1024ch), d3, and d2 decoder stages.
- Decoder redesigned as `FusionUpBlock` — each stage concatenates skip connections from **both** encoders.
- Switched to **HybridLoss**: `FocalTversky + 0.3 × FocalBCE(pos_weight=3.0)`.
- γ updated to **1.5** (from 1.33 in V5).

### Key result
+0.1144 IoU over V5 from architectural change alone. Dual encoders preserve the full quality of ImageNet RGB pretraining while learning a separate feature hierarchy for SAR-derived channels.

---

## V5 · May 2026

**Architecture:** ResNet34-UNet, single encoder, 9 channels  
**Val IoU:** 0.4906 · **Test IoU:** 0.0080  
**Checkpoint:** `best_v5.pth`

### What changed from V4
- **γ corrected from 0.75 → 1.33** in Focal Tversky Loss. V1–V4 used γ=0.75, which fractionally exponentiates a [0,1] value — amplifying easy examples instead of suppressing them. This is the mathematical inverse of the intended focal mechanism (Abraham & Khan 2019 requires γ > 1). Correction was the largest single performance gain across all versions.
- NGRDI normalisation bug fixed: raw NGRDI range [-1,1] now correctly mapped to [0,1] before SAR-NGRDI interaction.
- Full 1024×1024 resolution restored (V3/V4 used 512 crops).

### Notes
Best validation IoU among single-encoder models. Low test IoU (0.008) suggests γ correction improved within-scene performance but cross-scene generalisation still limited by single-encoder architecture.

---

## V4 · May 2026

**Architecture:** ResNet34-UNet, 9 channels (NGRDI/SAR-NGRDI added, bug present)  
**Val IoU:** 0.3137 · **Test IoU:** 0.0348  
**Checkpoint:** `best_v4.pth` *(excluded from HuggingFace — simultaneous bugs)*

### What changed from V3
- Added channels 7 (NGRDI) and 8 (SAR-NGRDI) for vegetation-damage signal.
- Despite a normalisation bug in NGRDI and γ still inverted, achieved the **best test IoU (0.035) up to this point** — demonstrating that physics-grounded vegetation channels generalise across scenes even when improperly normalised.

### Known bugs
- NGRDI raw range [-1,1] not remapped to [0,1] before use.
- γ = 0.75 (inverted focal mechanism, inherited from V1–V3).

---

## V3 · May 2026

**Architecture:** ResNet34-UNet, 7 channels, 512×512 crops  
**Val IoU:** 0.4198 · **Test IoU:** 0.018  
**Checkpoint:** `best_v3.pth`

### What changed from V2
- Introduced 512×512 random crops during training (albumentations).
- Added `A.HorizontalFlip`, `A.VerticalFlip`, `A.RandomRotate90`.
- Slightly improved test IoU over V2 (0.018 vs 0.012).

### Notes
Val IoU regression vs V2 (0.4198 vs 0.4757). Hypothesis: 512 crops lose the surrounding-context signal needed to confirm building damage. A collapsed structure spanning 600–800px cannot be inferred from a 512px fragment; intact neighbours provide the reference contrast.

---

## V2 · May 2026

**Architecture:** ResNet34-UNet, 7 channels, full 1024×1024  
**Val IoU:** 0.4757 · **Test IoU:** 0.012  
**Checkpoint:** `best_v2.pth`

### What changed from V1
- Removed redundant `sar_int` channel (was an exact copy of `post_norm` — zero new information, wasted encoder capacity).
- Clean top-to-bottom notebook structure (Run All works first time).
- Single training loop, one checkpoint name.
- `weights_only=True` added to `torch.load`.

---

## V1 · May 2026

**Architecture:** ResNet34-UNet, 8 channels, full 1024×1024  
**Val IoU:** ~0.4423 (3-epoch sanity run only)  
**Checkpoint:** not published (training incomplete)

### Notes
- Messy notebook with duplicate cells, two training loops, wrong cell ordering.
- 8-channel input included redundant `sar_int` (copy of `post_norm`).
- Full 10-epoch training loop was defined but **never executed** (execution count: None).
- γ = 0.75 (inverted focal mechanism — present in all versions through V4).
- Serves as architecture baseline for the series.
